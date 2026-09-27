"""A recorded run, read back as the conversation that produced it.

WHAT IS AND IS NOT ON DISK.  ``runs/<arm>/decisions.jsonl`` holds, per turn, the
new observation and the completion it drew.  It does **not** hold the two things
that framed them:

* the **system block and action grammar** -- where the size rule, the hedge
  disclosure, the admitted families and the DSL are *stated to the policy*.
  Built by ``state_space.system_block(config)`` at run time and thrown away.
  Recovered here from ``manifest.json``'s ``env_config`` via
  ``EnvConfig.from_dict``, and the recovery is checked against the manifest's
  own ``env_fingerprint`` before it is used.
* the **episode header** -- the ``EPI`` line and, from the second episode on,
  the restatement of the book carried across the month boundary.  Rebuilt here
  by calling the environment's own ``episode_block`` against the book restored
  from ``state/<previous episode>.json``; see :func:`episode_header` for why it
  is that file and not the first observation.

Training on the completions without these would teach a student to produce
answers whose instructions it never saw.  That failure is silent: the resulting
corpus is well-formed, loads cleanly, and trains.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.  It does not tokenize, does not pad,
does not template, and does not know what a trainer is.  It emits messages with
a trainable flag; turning those into token ids with a loss mask is the backend's
job, because the chat template is a property of the model and the framework, not
of the data.  The one number it does compute is a token count, and only because
an episode that will not fit the training sequence length has to be found here
rather than after it has been silently truncated.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from ..env.book import BookError, BookState
from ..env.pseudonyms import Pseudonyms, anonymize
from ..env.spec import EnvConfig
from ..env.statespace import EpisodeContext, build_state_space
from ..env.tokens import HeuristicTokenCounter, TokenCounter

__all__ = [
    "CorpusError",
    "Message",
    "TrainingEpisode",
    "build_episodes",
    "carried_state",
    "episode_header",
    "load_env_config",
    "read_decision_rows",
    "system_message",
    "write_chat_jsonl",
    "write_chat_parquet",
]

#: What distinguishes a *decision* turn from a *quote* turn in
#: ``decisions.jsonl``.  ``step_block`` always opens with the ``T`` header;
#: ``quote_block`` emits the bare ``R`` rows and never does.
#:
#: The recorded ``turn`` field is **not** usable for this: it labels what the
#: completion did, so the row marked ``"quote"`` is the one carrying the full
#: state and the row marked ``"act"`` carries the price block.  Keying on it
#: inverts the test.
_STEP_PREFIX = "T "


class CorpusError(ValueError):
    """A run directory cannot be read back as a conversation."""


@dataclass(frozen=True)
class Message:
    """One wire turn.

    ``trainable`` is the loss mask in its framework-independent form: true on
    what the policy produced, false on what it was given.  Kept as a per-message
    flag rather than as "assistant turns are trainable" because the two stop
    agreeing the moment an arm replays a teacher's reasoning back into the
    prompt -- that is an assistant-role turn the student must not be trained to
    reproduce.
    """

    role: str
    content: str
    trainable: bool = False


@dataclass(frozen=True)
class TrainingEpisode:
    """One month of one run, as an append-only conversation."""

    arm: str
    episode_id: str
    messages: tuple[Message, ...]
    env_fingerprint: str
    state_space_id: str
    model: str
    #: Turns the policy took.  Not ``len(messages)``: the system message and the
    #: header-bearing first observation are not decisions.
    decisions: int
    tokens: int
    #: Present only when the run recorded one.  A non-empty value means at least
    #: one turn of this episode failed at the provider, and a dead run looks
    #: exactly like a healthy one, so it is carried rather than dropped.
    errors: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)

    def as_chat_record(self) -> dict[str, object]:
        """The generic form: OpenAI-style messages plus a parallel loss mask.

        Generic on purpose.  Miles, veRL and trl all read some dialect of this,
        and every one of them differs in where the mask goes; emitting the union
        and letting the backend project it is cheaper than emitting three files.

        The mask is therefore emitted **twice**, in both dialects:

        *   ``loss_mask``, a parallel list of bools, which is what veRL's
            ``MultiTurnSFTDataset`` reads;
        *   ``step_loss_mask``, an int **inside each message dict**, which is
            what Miles reads -- ``miles/utils/mask_utils.py`` consults
            ``message.get("step_loss_mask", 1)`` and knows nothing about a
            top-level column, so a corpus carrying only the veRL dialect is
            silently accepted and silently masked by Miles' own default
            (assistant turns trainable, everything else not).

        Two consequences of Miles' spelling are worth stating where the writer
        lives.  First, the override is **subtractive only**: a ``0`` can switch
        a turn off, but no value switches on a turn whose role is not
        ``assistant``, so a corpus that needs a trainable non-assistant turn
        cannot be expressed to Miles at all -- no config fixes it, which is why
        the tests assert the invariant instead of a config documenting it.
        Second, the flag is written on **every** message and not only
        where it is zero: parquet gives a struct column one schema for the whole
        file, so a field present on some messages arrives as ``None`` on the
        others, and ``None != 1`` is exactly the condition that zeroes a mask.
        Omitting the flag where it is redundant would therefore silently
        untrain those turns on the parquet path only.

        ROW IDENTITY LIVES UNDER ``metadata`` FOR THE SAME CLASS OF REASON.
        ``Dataset.__init__`` builds ``Sample(prompt, label, metadata,
        multimodal_inputs)`` and reads nothing else off the row, so a top-level
        key is loaded from the file and then discarded.  SFT does not notice --
        ``sft_rollout`` touches only ``prompt`` and ``metadata["tools"]`` and
        hard-sets ``sample.reward = 0``, so identity is not an input to the
        computation -- but a GRPO reward is ``custom_rm(args, sample) -> float``
        and ``sample.metadata`` is the only per-row channel it gets.  Miles
        routes its own ``--opd-teacher-key`` through that channel, so this is
        the framework's intended extension point rather than a workaround.
        """
        return {
            "messages": [
                {"role": m.role, "content": m.content, "step_loss_mask": int(m.trainable)}
                for m in self.messages
            ],
            "loss_mask": [m.trainable for m in self.messages],
            # Every key present on every row, for the reason the mask is: a
            # struct column has one schema per parquet file, so a key written
            # on only some rows comes back ``None`` on the others.
            "metadata": {
                "arm": self.arm,
                "episode_id": self.episode_id,
                "state_space_id": self.state_space_id,
            },
            # Deliberately NOT moved.  The fingerprint is one value for the
            # whole corpus -- every selected run shares an environment -- so it
            # is a property of the file, and ``MANIFEST.json`` is where a
            # file-level fact belongs.  It stays at the top level of the row too
            # so that one line of the jsonl is self-describing to a reviewer.
            "env_fingerprint": self.env_fingerprint,
            "teacher_model": self.model,
            "decisions": self.decisions,
            "tokens": self.tokens,
            **({"errors": list(self.errors)} if self.errors else {}),
            **dict(self.provenance),
        }


# --------------------------------------------------------------------------
# reading a run directory
# --------------------------------------------------------------------------


def read_manifest(run_dir: Path | str) -> Mapping[str, object]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise CorpusError(
            f"{run_dir}: no manifest.json. A run killed mid-flight leaves its "
            "decisions behind without one; such a directory is a torso, not a "
            "short run, and must not be read as a complete episode."
        )
    return json.loads(path.read_text())


def load_env_config(run_dir: Path | str) -> EnvConfig:
    """The config this run executed under, verified against its own hash.

    The verification is the whole point.  ``from_dict`` fills a missing key with
    the field default, which is the right behaviour for reading an older
    manifest and the wrong behaviour for building a prompt -- a defaulted field
    silently rewrites the rulebook.  Comparing the rebuilt fingerprint against
    the recorded one catches missing, extra and altered keys in one comparison,
    and it is cheap, so it is not optional.
    """
    manifest = read_manifest(run_dir)
    payload = manifest.get("env_config")
    if not isinstance(payload, Mapping):
        raise CorpusError(f"{run_dir}: manifest has no env_config object")
    config = EnvConfig.from_dict(payload)
    recorded = manifest.get("env_fingerprint")
    if recorded and config.fingerprint() != recorded:
        raise CorpusError(
            f"{run_dir}: the config rebuilt from the manifest fingerprints "
            f"{config.fingerprint()[:12]}... but the run recorded "
            f"{str(recorded)[:12]}.... The system block this would generate is "
            "not the one the policy was given, so the trajectory cannot be "
            "used as a label."
        )
    return config


def read_decision_rows(run_dir: Path | str) -> Iterator[Mapping[str, object]]:
    path = Path(run_dir) / "decisions.jsonl"
    if not path.exists():
        raise CorpusError(f"{run_dir}: no decisions.jsonl")
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------
# rebuilding the two blocks that were never written down
# --------------------------------------------------------------------------


def system_message(config: EnvConfig) -> str:
    """System block and action grammar, joined the way the runner joins them.

    ``"\\n\\n"`` is not a guess: it is the separator that reproduces, byte for
    byte, the ``system`` string the policy was reset with.  Pinned by test
    rather than by comment, because a one-character drift here shifts every
    token position in every training example and would show up only as a
    slightly worse model.
    """
    space = build_state_space(config)
    return f"{space.system_block(config)}\n\n{space.action_grammar(config)}"


def carried_state(run_dir: Path | str, previous_episode_id: str) -> BookState:
    """The book as it arrives at an episode, from the previous episode's handoff.

    ``run_arm`` writes ``state/<episode>.json`` as the last thing it does in an
    episode, and builds the next episode's header before calling ``env.reset``.
    Nothing runs in between, so the restored book here *is* the book that
    rendered that header.
    """
    path = Path(run_dir) / "state" / f"{previous_episode_id}.json"
    if not path.exists():
        raise CorpusError(
            f"{run_dir}: no state/{previous_episode_id}.json, so the book "
            "carried into the next episode cannot be restated. The run was "
            "executed without --state-dir; its later episodes are missing the "
            "half of the header that says what the policy already owned."
        )
    try:
        return BookState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except BookError as exc:  # checksum, or a NAV that does not add up
        raise CorpusError(f"{path}: {exc}") from exc


def episode_header(
    config: EnvConfig,
    *,
    episode_id: str,
    start_date: date,
    end_date: date,
    decision_points: int,
    carried: BookState | None = None,
) -> str:
    """The ``EPI`` block, rebuilt through the environment's own renderer.

    WHY THE HANDOFF FILE AND NOT THE FIRST OBSERVATION.  The obvious shortcut is
    to lift the ``A`` and ``P`` blocks out of the episode's first observation:
    ``episode_block`` and ``step_block`` share ``_account_row`` and
    ``_position_rows``, so the two look like they must agree.  They do not, for
    two reasons, and both were found by comparing against a live environment
    rather than by reading the code:

    1.  ``runner._run_episode`` builds the header *before* ``env.reset``, which
        advances to the first grid point -- settling expiries, accruing interest
        and re-marking.  The header therefore describes the book at the previous
        episode's close.  A dte one day stale is the visible symptom; a mark
        moved by a weekend is the invisible one.
    2.  Under ``flags.anonymize`` the labels are redrawn per episode, so the
        previous episode's rows are masked under a *different* bijection than
        the header's.  Copying them forward would relabel the whole book.

    So the carried book is restored from ``state/<previous>.json`` -- the
    handoff the runner itself writes -- rendered by ``episode_block``, and
    masked with this episode's own draw.  Every byte comes from the environment;
    this function chooses the inputs and nothing else.
    """
    space = build_state_space(config)
    text = space.episode_block(
        EpisodeContext(
            episode_id=episode_id,
            config=config,
            start_date=start_date,
            end_date=end_date,
            decision_points=decision_points,
            # ``as_of`` reaches only ``Position.view``, which does not read it;
            # the dte on the row is the one stored at the last mark.  Passing
            # the handoff's own stamp keeps the call honest rather than useful.
            carried_in=carried.restore().view(carried.as_of) if carried else None,
        )
    )
    if not config.flags.anonymize:
        return text
    # ``trade_date=None`` exactly as ``OptionsEnv.episode_header`` passes it:
    # the span is a span, not a step, so it is not given the sawtooth.
    pseudo = Pseudonyms.draw(episode_id, config.universe.observed)
    return anonymize(text, pseudo, trade_date=None)


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def build_episodes(
    run_dir: Path | str,
    *,
    counter: TokenCounter | None = None,
    max_tokens: int | None = None,
) -> list[TrainingEpisode]:
    """Every episode of one run, as a conversation.

    ``max_tokens`` drops -- and reports, via ``provenance`` on the survivors --
    episodes that will not fit the training sequence length.  It is ``None`` by
    default because dropping is a sampling decision: applied at the tail of a
    corpus whose head was admitted under a different budget, it silently
    reweights the sample.  The caller has to ask for it.
    """
    run_dir = Path(run_dir)
    counter = counter or HeuristicTokenCounter()
    manifest = read_manifest(run_dir)
    config = load_env_config(run_dir)
    system = system_message(config)

    order = [
        str(e["episode_id"])
        for e in manifest.get("episodes", ())  # type: ignore[union-attr]
        if isinstance(e, Mapping)
    ]
    spans = {
        str(e["episode_id"]): e
        for e in manifest.get("episodes", ())  # type: ignore[union-attr]
        if isinstance(e, Mapping)
    }
    if not spans:
        raise CorpusError(f"{run_dir}: manifest lists no episodes")

    # File order is wire order -- the ledger is append-only -- and that matters
    # because a quoted step writes two rows under one ``step_index``.  Sorting
    # on ``step_index`` alone would leave their order to the sort's stability;
    # sorting on the timestamp would break on a run where both land in the same
    # second.  Neither is necessary: the file already has them in the order the
    # policy saw them.
    rows_by_episode: dict[str, list[Mapping[str, object]]] = {}
    for row in read_decision_rows(run_dir):
        rows_by_episode.setdefault(str(row["episode_id"]), []).append(row)

    arm = str(manifest.get("arm", run_dir.name))
    model = str(manifest.get("policy_model", ""))
    fingerprint = str(manifest.get("env_fingerprint", ""))

    out: list[TrainingEpisode] = []
    dropped: list[str] = []
    for position, episode_id in enumerate(order):
        rows = rows_by_episode.get(episode_id)
        if not rows:
            continue
        span = spans[episode_id]
        if span.get("terminated"):
            raise CorpusError(
                f"{run_dir}: episode {episode_id!r} terminated early "
                f"({span['terminated']!r}). Its grid ran past its last decision, "
                "so the span on the header cannot be recovered from the rows, "
                "and a header that understates the month is a header the policy "
                "never saw."
            )
        # ``steps=`` on the header is the count of decision *grid points*, which
        # is not ``len(rows)``: a quoted step writes a proposal row and a fill
        # row under one point, and it is not the manifest's ``decisions`` field
        # either, which counts policy turns and so includes the proposals.
        steps = [r for r in rows if str(r["observation"]).startswith(_STEP_PREFIX)]
        if not steps:
            raise CorpusError(
                f"{run_dir}: episode {episode_id!r} has no turn carrying a step "
                "block, so neither its span nor its decision count can be read"
            )
        dates = [date.fromisoformat(str(r["step_ts"])[:10]) for r in steps]
        header = episode_header(
            config,
            episode_id=episode_id,
            start_date=dates[0],
            end_date=dates[-1],
            decision_points=len(steps),
            # The first episode of a run starts flat and the runner passes
            # ``carried_in=False``; every later one restates the book it was
            # handed, which is the previous episode's checkpoint.
            carried=None if position == 0 else carried_state(run_dir, order[position - 1]),
        )

        messages = [Message("system", system, trainable=False)]
        errors: list[str] = []
        for index, row in enumerate(rows):
            observation = str(row["observation"])
            # The header is prepended to the first observation rather than sent
            # as its own message, because that is how the runner sends it: one
            # user turn carrying header then state.  Splitting it into two would
            # add a message boundary -- and, under any chat template, extra
            # control tokens -- that the recorded conversation never had.
            content = f"{header}\n{observation}" if index == 0 else observation
            messages.append(Message("user", content, trainable=False))
            messages.append(
                Message("assistant", str(row["completion"]), trainable=True)
            )
            if row.get("error"):
                errors.append(f"step {row['step_index']}: {row['error']}")

        tokens = sum(counter.count(m.content) for m in messages)
        episode = TrainingEpisode(
            arm=arm,
            episode_id=episode_id,
            messages=tuple(messages),
            env_fingerprint=fingerprint,
            state_space_id=str(manifest.get("state_space_id", config.state_space_id)),
            model=model,
            decisions=len(rows),
            tokens=tokens,
            errors=tuple(errors),
            provenance={"token_counter": counter.identity},
        )
        if max_tokens is not None and tokens > max_tokens:
            dropped.append(episode_id)
            continue
        out.append(episode)

    # The drop is recorded on the survivors, not merely performed.  A corpus
    # built under a cap and one built without it are otherwise byte-comparable
    # files with different row counts, and the difference is a *sampling*
    # difference: the longest episodes are systematically the months with the
    # most trading days, so a cap reweights the corpus by calendar rather than
    # at random.  ``max_tokens`` is carried even when nothing was dropped,
    # because "nothing exceeded the cap" and "no cap was applied" are different
    # facts about a file somebody else will train on.
    if max_tokens is not None:
        for index, episode in enumerate(out):
            out[index] = replace(
                episode,
                provenance={
                    **episode.provenance,
                    "max_tokens": max_tokens,
                    "dropped_over_max_tokens": tuple(dropped),
                },
            )
    return out


def write_chat_jsonl(
    episodes: Sequence[TrainingEpisode], path: Path | str
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.as_chat_record()) + "\n")
    return len(episodes)


def write_chat_parquet(
    episodes: Sequence[TrainingEpisode], path: Path | str
) -> int:
    """The same records as parquet, because that is what veRL can open.

    ``MultiTurnSFTDataset`` reads its input with
    ``pandas.read_parquet(parquet_file, dtype_backend="pyarrow")``
    (``verl/utils/dataset/multiturn_sft_dataset.py:146``) -- there is no jsonl
    path, and a ``.jsonl`` handed to it fails *after* Ray has started.  Miles'
    format is unknown, which is the other reason the jsonl stays: it is the
    reviewable, diffable, grep-able artifact, and this is a derived file.

    ``messages`` becomes a list of structs and ``loss_mask`` a list of bools;
    both are derived from the same turn sequence in
    :meth:`TrainingEpisode.as_chat_record`, so they cannot slip against each
    other here.  What *can* go wrong is the schema: parquet has no place for an
    absent column, so an episode without errors and one with them must agree
    before the row groups will concatenate.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise CorpusError(
            "write_chat_parquet needs pandas + pyarrow. The jsonl writer has no "
            "such dependency, so build the jsonl here and convert on the machine "
            "that has the trainer."
        ) from exc

    records = []
    for episode in episodes:
        record = dict(episode.as_chat_record())
        record.setdefault("errors", [])
        records.append(record)
    if not records:
        raise CorpusError(
            "refusing to write an empty parquet. veRL reads the file, finds no "
            "rows, and reports a division by zero inside the sampler."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_parquet(path, index=False)
    return len(records)
