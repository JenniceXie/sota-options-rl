"""Re-emit a recorded decision stream so an arm can be varied without the model.

Every number this project compares across arms is downstream of a sampler that
does not repeat.  Three runs over the same window — ``traj_v2_nohedge``,
``traj_v2_volhedge``, ``traj_v3_volhedge`` — sent a byte-identical step-0
observation, which the provider tokenized to 1,301 tokens in all three cases,
at ``temperature=0`` and ``seed=0``, and opened three different books.  Hosted
inference batches requests, floating-point addition is not associative, so the
reduction order behind every logit depends on whose request shared the batch;
``argmax`` is then a discontinuous function of a value that moves in its last
bits.  Across the ~10,000 sequential argmaxes in one step's reasoning, agreement
is the unlikely outcome, not the likely one.

That noise is not fixable from here and does not need to be.  It is only fatal
because it sits *between* the thing being varied and the thing being measured.
No flag name enters an *observation* — so for two configurations disclosed
alike, the policy cannot know which arm it is in, and every difference in what
it *chose* is noise by construction.  Comparing two live runs measures the
configuration plus a sampler; this measures the configuration.  (Some rules
*are* disclosed, in the system block, which is not an observation and which no
anchor covers; that is the subject of the amendment below, and it does not
weaken the paragraph above — it bounds which flags it covers.)

So the arm comparison stops being two samples across an unmeasured noise floor
and becomes a paired diff: record the decisions once, replay the same actions
under each configuration, and attribute the difference to the configuration
because nothing else was allowed to vary.

**The policy asserts its own input.**  ``act`` refuses to answer an observation
that differs from the one recorded at that position in the stream.  That is the
whole safety property.  A replay that emitted recorded actions against drifted
state would look like a successful run and would silently be measuring a
different experiment — orders placed for a book that no longer exists, priced
against a chain that moved.  Because it refuses instead, a completed replay is
positive evidence that the environment reproduced every observation exactly,
which is the premise the paired diff rests on.

**A configuration that trades is only replayable against itself.**  The flag
name stays out of the prompt but its *consequences* do not: the account row
carries NAV, cash, buying power and net delta, so the first fill a resolver
places under one setting and not the other puts the two arms in different
states, and every later recorded action belongs to only one of them.  Replaying
``traj_v3_volhedge`` at ``--hedge volatility`` reproduced its ledger exactly;
the same recording at ``--hedge none`` refused at decision 2, on the AM row
after the first hedge fill, with cash 937 against a recorded 939 (thousands).

That is the tool working, not failing.  What it rules out is the cheap reading
of a paired replay — "run the recording under both settings and difference the
NAV curves" — for any flag that changes fills.  The paired diff is exact only
up to the first divergent fill, and the refusal is what dates it.  Past that
point the honest comparison is two live runs with the sampler measured, not one
recording pretending to cover both books.

**AMENDED 2026-09-24 — a second tolerance, and why it is not "turn the check
off".**  Everything above still holds for the default.  What it left with no
instrument is the whole *contribution* half of the ablation matrix: the sizing
arms and the hedge-contribution arms hold the action sequence fixed and vary
what the resolvers do with it, which changes a fill, which changes the account
row, which changes the next observation.  Under ``exact`` those arms cannot be
run at all — not "run imprecisely", not run.  The alternative was a GPU
re-generation of trajectories whose actions are already on disk and are meant to
stay fixed, which would measure the sampler again, which is the thing this
module exists to remove.

So the weakening is scoped rather than global.  An observation is not one
string, it is blocks, and they divide cleanly by causation:

    anchor    T header, M market, N news     the arm cannot reach these
    tolerated A account, P positions, R results   the arm's own fills

``state_tolerant`` drops the comparison on the second set and keeps it, byte for
byte, on the first.  Replaying month one's actions into month two still refuses
— the header moves.  A dropped or inserted decision point still refuses.  A
different universe, a different anonymization draw, a state space with a block
removed: all still refuse, and the last of those matters because it means the
A1 contribution arms remain *correctly* un-replayable and still need inference.

Two further guards exist only because tolerance opens the door to them:

*Turn kind is checked on every decision, in both modes.*  A ``Q`` completion
produces two turns and a non-``Q`` completion produces one, so a configuration
under which a proposal resolves to nothing would consume one fewer turn than the
recording and silently shift every later action by one — while each individual
observation still looked plausible.  Bytes made that check redundant before;
without bytes it is the only thing holding the streams in step.

*Divergence is dated rather than hidden.*  Under ``exact`` the refusal is what
tells you where the arms stopped being comparable.  Tolerance removes the
refusal, so it must supply the date itself: ``replay_stats`` carries the count
of drifted decisions and the identity of the first one, and the runner puts them
in the manifest.  A tolerant replay that reports zero divergences is exactly as
strong as an exact one; a tolerant replay that reports its first divergence at
decision 2 is telling you the same thing the old ``ReplayMismatch`` did, and the
reader of the results table can see it without rerunning anything.

The block split is verified lossless over 5,405 recorded observations from 60
runs of the astra corpus: every line attributes to a block and rejoining them
reproduces the text.  ``state_tolerant`` is opt-in, never a fallback, and is
recorded in the manifest, because a run that tolerated drift and a run that
proved there was none must not look the same afterwards.

**WHAT TOLERANCE STILL CANNOT SEE, AND WHY THAT IS THE MEASUREMENT.**  Since
2026-09-23 the hedged family set, the band, the band-edge rule and the size rule
are rendered into the *system block*.  The system block is not an observation,
so no anchor covers it: a tolerant replay across ``--hedge`` or ``--size-rule``
re-emits decisions that were taken while reading a different rulebook, and
nothing in ``act`` will object.

That is not a hole in the check, it is the definition of the quantity.  A
contribution arm asks what the recorded *behaviour* is worth under a different
mechanism; holding the behaviour fixed is the point, and the disclosure is part
of what produced the behaviour.  The arm that lets the policy respond to the new
rulebook is the adaptation arm, and it is a retrain, not a replay.  The two
numbers are meant to differ — that gap is the result.

What would be dishonest is a reader taking a tolerant replay for "same policy,
different resolver".  So ``reset`` hashes the system block it is handed and
``replay_stats`` reports it.  The recording side did not write such a hash — the
system block is not persisted in ``runs/`` and has to be rebuilt from
``EnvConfig`` — so this does not compare, it *states*: two tolerant replays with
different ``system_sha`` values were run under different disclosures, and that
is visible from the manifests alone instead of requiring the prompt to be
reconstructed months later.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..statespace import Observation
from . import PolicyResponse

__all__ = [
    "ReplayPolicy",
    "ReplayMismatch",
    "RecordedDecision",
    "read_decisions",
    "EXACT",
    "STATE_TOLERANT",
    "TOLERANCES",
    "ANCHOR_BLOCKS",
    "split_blocks",
]

#: Refuse any observation that is not byte-identical to the recording.  The
#: default, and the only setting under which a completed replay is by itself
#: proof that the environment reproduced the recorded run.
EXACT = "exact"

#: Refuse any observation whose *anchor* blocks differ, and permit the blocks
#: the replayed arm's own fills are allowed to move.  Required by any arm that
#: changes a fill -- sizing, hedging -- and reported in the manifest so that a
#: number produced under it is never mistaken for one produced under ``EXACT``.
STATE_TOLERANT = "state_tolerant"

TOLERANCES = (EXACT, STATE_TOLERANT)

#: Block order as ``statespace_v1.step_block`` emits it.  The order is what
#: makes the split unambiguous: each block is a contiguous run of lines and the
#: leading letter only ever moves forward, so a continuation line that happens
#: to begin with a block letter already passed stays with its own block.
_BLOCK_ORDER = ("T", "M", "A", "P", "N", "R")

#: Blocks no arm can reach: the grid header, the market rows, the news rows.
#: Every one is a function of the date, the session and the universe, so a
#: difference here is a difference in *which experiment is running*, never a
#: consequence of the flag under test.  ``?`` is here as a fail-closed bucket --
#: a line that attributes to no block is unexplained, and unexplained text is
#: compared strictly rather than waved through.
ANCHOR_BLOCKS = ("T", "M", "N", "?")


def split_blocks(text: str) -> dict[str, str]:
    """Split a rendered observation back into the blocks it was joined from.

    ``Observation`` carries its blocks, but a *recording* carries only the
    joined text, and the comparison needs both sides in the same shape.  Parsing
    the recorded side is therefore unavoidable; what makes it safe is that the
    join is ``"\\n".join`` over an insertion-ordered dict, so the inverse is a
    single forward pass with no lookahead and no ambiguity.

    Verified lossless over 5,405 observations from 60 runs of the astra corpus
    (2026-09-24): every line attributed to a block, and rejoining reproduced the
    original text in every case.
    """
    rank = {key: index for index, key in enumerate(_BLOCK_ORDER)}
    blocks: dict[str, list[str]] = {}
    current = ""
    highest = -1
    for line in text.splitlines():
        key = line.split(" ", 1)[0] if line else ""
        if key in rank and rank[key] > highest:
            current, highest = key, rank[key]
            blocks[current] = []
        blocks.setdefault(current or "?", []).append(line)
    return {key: "\n".join(lines) for key, lines in blocks.items()}


def observation_kind(text: str) -> str:
    """Which of the two shapes of observation this is.

    ``step_block`` always opens with a ``T`` header; ``quote_block`` renders the
    result block alone and never does.  That is the whole discriminator, and it
    reads off the text rather than off a recorded label, so it means the same
    thing on the live side and the recorded side.
    """
    return "state" if "T" in split_blocks(text) else "quote_answer"


def anchor_mismatch(recorded: str, actual: str) -> str | None:
    """The first anchor block that differs, or ``None`` if the anchors agree.

    A block present on one side and absent on the other counts as a difference:
    an observation that lost its news block is a different observation, and
    comparing only the blocks they happen to share would be exactly the kind of
    check that cannot fail.
    """
    left = split_blocks(recorded)
    right = split_blocks(actual)
    for key in ANCHOR_BLOCKS:
        if left.get(key) != right.get(key):
            return key
    return None


class ReplayMismatch(RuntimeError):
    """The replay was asked for an action the recording does not cover.

    Fatal rather than a warning, and fatal on the *first* divergence.  The
    alternative is a run that keeps going and produces a ledger, which is the
    expensive failure: the artifact is indistinguishable from a good one, and
    the difference only shows up as PnL nobody can source.
    """


@dataclass(frozen=True, slots=True)
class RecordedDecision:
    """One row of a recorded ``decisions.jsonl``, reduced to what replay needs."""

    step_ts: str
    episode_id: str
    step_index: int
    observation: str
    completion: str
    reasoning: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    @property
    def observation_kind(self) -> str:
        """``"state"`` or ``"quote_answer"``, derived from the text itself.

        Deliberately *not* read from the row's ``turn`` field, which labels
        something else: ``turn`` says what the completion did, so the row marked
        ``"quote"`` is the one whose answer asked for prices and whose
        observation is the ordinary full state, while the row that follows it is
        marked ``"act"`` and is the one carrying the bare price block.  Keying
        the alignment check on ``turn`` inverts it, which is how this was first
        written and what ``test_replaying_a_quoted_run_with_the_verb_on...``
        caught.

        The text is unambiguous where the label is not: ``step_block`` always
        opens with a ``T`` header and ``quote_block`` never does.
        """
        return observation_kind(self.observation)


def read_decisions(path: Path | str) -> tuple[RecordedDecision, ...]:
    """Load a decision stream in the order it was written.

    Order is the alignment key.  ``step_ts`` is carried for error messages only:
    matching on it would let a replay skip a decision point the recording does
    not have and call the result aligned, when a missing decision point is
    exactly the kind of divergence this is meant to catch.
    """
    rows: list[RecordedDecision] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                RecordedDecision(
                    step_ts=str(row.get("step_ts", "")),
                    episode_id=str(row.get("episode_id", "")),
                    step_index=int(row.get("step_index", -1)),
                    observation=str(row.get("observation", "")),
                    completion=str(row.get("completion", "")),
                    reasoning=str(row.get("reasoning") or ""),
                    model=str(row.get("model") or ""),
                    prompt_tokens=int(row.get("prompt_tokens") or 0),
                    completion_tokens=int(row.get("completion_tokens") or 0),
                )
            )
    return tuple(rows)


class ReplayPolicy:
    """Emits the recorded action for each decision point, in order."""

    name = "replay"

    def __init__(
        self,
        decisions: Sequence[RecordedDecision],
        *,
        source: str = "",
        tolerance: str = EXACT,
    ) -> None:
        if tolerance not in TOLERANCES:
            raise ValueError(
                f"unknown replay tolerance {tolerance!r}; expected one of "
                f"{', '.join(TOLERANCES)}. A misspelling must not silently "
                f"select {EXACT!r}: an arm that needs {STATE_TOLERANT!r} would "
                "then fail at its first fill and look like a broken environment "
                "rather than a mis-typed flag."
            )
        self._decisions = tuple(decisions)
        self._source = source
        self._tolerance = tolerance
        self._cursor = 0
        self._divergent = 0
        self._first_divergence: dict[str, object] | None = None
        self._system_sha = ""
        #: Carried so the manifest can state what was replayed and how much of
        #: it.  A replay that stopped early because the window was shorter than
        #: the recording is a different artifact from one that consumed it all.
        self.model = f"replay:{self._decisions[0].model}" if self._decisions else "replay"

    @classmethod
    def from_path(cls, path: Path | str, *, tolerance: str = EXACT) -> ReplayPolicy:
        return cls(read_decisions(path), source=str(path), tolerance=tolerance)

    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """No context to build.

        ``tools`` is ignored, and harmlessly: a recording already contains the
        completions, so the channel they were *asked* for on has no bearing on
        what this policy returns.  What does matter is that the observations
        still match byte-for-byte, which is ``act``'s check and is why the tool
        channel was built to leave ``observation.text`` untouched.

        Deliberately *not* a cursor reset.  The recording is one stream across
        every episode in the run, and restarting it at each episode boundary
        would replay month one's actions into month two while every observation
        still matched position-for-position within the episode -- the one drift
        the text check cannot see.

        ``system`` and ``grammar`` are hashed rather than used.  Under
        ``STATE_TOLERANT`` the rulebook the recorded decisions were taken under
        can legitimately differ from the one this run discloses -- that gap is
        the contribution measurement -- and the hash is what makes the
        difference legible in the manifest instead of inferable only by
        rebuilding both prompts from their configs.
        """
        self._system_sha = hashlib.sha256(
            f"{system}\n{grammar}".encode()
        ).hexdigest()[:12]
        return None

    def act(self, observation: Observation) -> PolicyResponse:
        if self._cursor >= len(self._decisions):
            raise ReplayMismatch(
                f"the recording has {len(self._decisions)} decisions and the run "
                f"asked for one more. The replayed window is longer than the "
                f"recorded one, so the arms do not cover the same dates."
                + (f" Source: {self._source}" if self._source else "")
            )

        recorded = self._decisions[self._cursor]
        where = (
            f"decision {self._cursor} ({recorded.episode_id} step "
            f"{recorded.step_index}, {recorded.step_ts})"
        )

        # Checked in both tolerances, and first.  A step whose completion asked
        # for prices consumes two turns and one that did not consumes one, so a
        # configuration under which a proposal resolves to nothing consumes one
        # fewer turn than the recording and shifts every later action by one --
        # while each individual observation still looks reasonable.  Bytes made
        # this redundant before; without them it is the only thing holding the
        # two streams in step.
        kind = observation_kind(observation.text)
        if kind != recorded.observation_kind:
            raise ReplayMismatch(
                f"the environment produced a {kind!r} observation where the "
                f"recording has a {recorded.observation_kind!r} one, at {where}. "
                "The two streams have come out of step -- most likely a proposal "
                "that resolved to nothing under this configuration and so never "
                "bought its prices -- so the recorded action belongs to a "
                "different decision than the one being asked for."
            )

        if self._tolerance == EXACT:
            if observation.text != recorded.observation:
                raise ReplayMismatch(
                    "the environment produced a different observation than the "
                    f"one recorded at {where}. Replaying the recorded action "
                    "here would trade against a state that did not occur.\n"
                    f"{_first_difference(recorded.observation, observation.text)}"
                    f"\n  If this arm is expected to change fills, it needs "
                    f"--replay-tolerance {STATE_TOLERANT} rather than a fix."
                )
        else:
            drifted = anchor_mismatch(recorded.observation, observation.text)
            if drifted is not None:
                raise ReplayMismatch(
                    f"the {drifted!r} block differs at {where}, and that block is "
                    "an anchor: no sizing or hedging flag can reach the header, "
                    "the market rows or the news rows. So this is not the arm "
                    "drifting, it is a different window, universe or state "
                    f"space, and {STATE_TOLERANT} does not cover it.\n"
                    f"{_first_difference(recorded.observation, observation.text)}"
                )
            if observation.text != recorded.observation:
                self._note_divergence(recorded, observation.text, where)

        self._cursor += 1
        return PolicyResponse(
            text=recorded.completion,
            reasoning=recorded.reasoning,
            # Zero, and not the recorded counts.  A replay makes no API call, so
            # carrying them would let a replayed arm report a cost it did not
            # pay -- and the evaluation report sums exactly this field per arm.
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=0.0,
            # Prefixed rather than copied, so a replayed arm cannot be mistaken
            # for a live one in a manifest or a results table.
            model=self.model,
            finish_reason="replay",
            extra={
                "replayed_from": self._source,
                "replayed_step_ts": recorded.step_ts,
                "recorded_prompt_tokens": recorded.prompt_tokens,
                "recorded_completion_tokens": recorded.completion_tokens,
            },
        )

    def _note_divergence(
        self, recorded: RecordedDecision, actual: str, where: str
    ) -> None:
        """Record that the book drifted here, and keep the first one forever.

        The first divergence is the load-bearing number.  It is the decision at
        which the paired diff stops being exact, which is precisely what the
        ``ReplayMismatch`` used to report by refusing -- so tolerating the drift
        without recording where it started would not be a weaker check, it would
        be no check, and the results table would carry no way to tell a clean
        arm from one that diverged on its second decision.
        """
        self._divergent += 1
        if self._first_divergence is not None:
            return
        left = split_blocks(recorded.observation)
        right = split_blocks(actual)
        self._first_divergence = {
            "decision_index": self._cursor,
            "episode_id": recorded.episode_id,
            "step_index": recorded.step_index,
            "step_ts": recorded.step_ts,
            "blocks": tuple(
                key
                for key in sorted(set(left) | set(right))
                if left.get(key) != right.get(key)
            ),
            "detail": _first_difference(recorded.observation, actual),
            "where": where,
        }

    @property
    def replay_stats(self) -> dict[str, object]:
        """What the manifest records about the replay.

        Named for the replay rather than called ``stats`` because the runner
        finds it with ``hasattr``: a generic name would be claimed by the next
        policy that wants to report something, and it would be claimed silently.

        ``tolerance`` is unconditional, including when it is the default.  A run
        whose manifest is silent about which comparison it made is a run whose
        numbers cannot be graded later, and "the key is absent so it must have
        been exact" is the kind of inference that is right until the day
        somebody changes the default.
        """
        return {
            "source": self._source,
            "recorded_decisions": len(self._decisions),
            "replayed_decisions": self._cursor,
            "consumed_all": self._cursor == len(self._decisions),
            "tolerance": self._tolerance,
            # Zero by construction under ``EXACT`` -- the run would have raised.
            # Emitted anyway so the two tolerances produce the same shape and a
            # reader can compare arms without special-casing.
            "divergent_decisions": self._divergent,
            "first_divergence": self._first_divergence,
            # The rulebook *this* run disclosed, not the one the recording was
            # taken under.  See the module docstring: under STATE_TOLERANT those
            # are allowed to differ, and this is what says so out loud.
            "system_sha": self._system_sha,
        }

    def __iter__(self) -> Iterator[RecordedDecision]:
        return iter(self._decisions)


def _first_difference(recorded: str, actual: str) -> str:
    """The first line that differs, which is almost always the whole story.

    Observations are line-oriented -- one line per name, per position, plus the
    account row -- so the first differing line names the field that moved.
    """
    recorded_lines = recorded.splitlines()
    actual_lines = actual.splitlines()
    for index in range(max(len(recorded_lines), len(actual_lines))):
        was = recorded_lines[index] if index < len(recorded_lines) else "<absent>"
        now = actual_lines[index] if index < len(actual_lines) else "<absent>"
        if was != now:
            return f"  line {index}\n  recorded: {was!r}\n  replayed: {now!r}"
    return "  the texts differ only in trailing whitespace"
