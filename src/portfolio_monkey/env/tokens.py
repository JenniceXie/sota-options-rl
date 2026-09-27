"""Counting the context budget with the tokenizer that will actually hold it.

WHY THIS EXISTS.  ``estimate_tokens`` in ``statespace`` is ``len(text) // 4``.
Its docstring is honest about being crude, and for a long time that was fine
because the number was only ever *reported* -- ``context_pressure`` has no
caller.  It stopped being fine when the episode budget became a training
constraint: ``--max-seq-len`` for the student has to hold the whole append-only
conversation, and a budget checked with the wrong ruler is not checked.

HOW WRONG.  Measured 2026-09-24 over the 378 episodes of the 126-run
``V+ASR+MDD`` astra corpus (``scratch/qwen3_seq_len.py``):

    chars // 4   mean 14,118   max 16,594
    Qwen3-4B     mean 26,125   max 32,763      ratio 1.85x

The crude count says the worst episode fills half the 32,768 budget.  The
tokenizer that will train on it says the same episode fills 99.98% of it.  Both
numbers are "the episode fits"; only one of them is a fact about the run.

THE DESIGN RULE HERE IS: NEVER FALL BACK SILENTLY.  A counter that is asked for
a real tokenizer and quietly returns the heuristic produces a run that looks
checked and is not -- the same failure shape as a policy that was handed tools
it cannot speak and returned a healthy-looking run with zero quotes.  So an
unreadable tokenizer, a missing dependency, and an unrecognized spec are all
:class:`ValueError` at construction time, before a single step runs.

Every counter carries an ``identity`` that goes into the run manifest, and for
a file-backed tokenizer that identity includes a hash of the file.  Swapping the
tokenizer under a fixed path is otherwise invisible, and the budget a run was
checked against is part of what the run means.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "FileTokenizerCounter",
    "ContextBudgetExceeded",
    "build_token_counter",
    "enforce_context_budget",
    "HEURISTIC_SPEC",
]

#: The spec string that selects the legacy ``len(text) // 4`` behaviour.  Named
#: rather than spelled ``None`` so that a manifest always states which ruler was
#: used, and "nobody chose" is distinguishable from "the crude one was chosen".
HEURISTIC_SPEC = "heuristic"


class ContextBudgetExceeded(RuntimeError):
    """The conversation grew past ``max_context_tokens``.

    Fatal, and fatal on the step that crossed -- not at the end of the episode.
    The alternative is an episode that completes and is written to ``runs/``
    looking exactly like every other episode, whose only defect is that no
    student with this context window can ever be trained on it.  That defect
    would surface as a truncation during training, thousands of GPU-hours
    downstream of the run that caused it and with nothing pointing back.
    """


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can say how many tokens a string is worth."""

    #: Stable string recorded in the manifest.  Two runs with the same identity
    #: were checked against the same ruler; two runs with different identities
    #: were not, whatever else their configs say.
    identity: str

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """``len(text) // 4``, preserved exactly, and now forced to say its name.

    This is the behaviour of every run written before 2026-09-24, so it stays
    available and stays the default: changing the arithmetic under an in-flight
    campaign would split its corpus across two rulers.  What changes is that a
    run using it now records ``heuristic.chars4`` in its manifest, so a
    trajectory that was never checked against a real tokenizer can be found
    later instead of being assumed fine.
    """

    identity = "heuristic.chars4"

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class FileTokenizerCounter:
    """A real tokenizer, loaded from a HuggingFace ``tokenizer.json``.

    Deliberately the *file* rather than a model id: resolving an id needs
    network egress, and on this cluster a login node throttles external
    transfers by ~190x, so a job that resolves an id is a job whose runtime
    depends on where it was launched.  A path is also what makes ``identity``
    meaningful -- the file can be hashed.

    ``tokenizers`` is imported lazily and its absence is an error, never a
    fallback.  It is ~3 MB and carries none of ``transformers``' weight, which
    is the dependency ``estimate_tokens`` was written to avoid.
    """

    def __init__(self, path: Path | str) -> None:
        resolved = Path(path)
        if resolved.is_dir():
            resolved = resolved / "tokenizer.json"
        if not resolved.is_file():
            raise ValueError(
                f"no tokenizer at {resolved}. Pass the path to a HuggingFace "
                f"tokenizer.json (or a directory containing one), or "
                f"{HEURISTIC_SPEC!r} to keep the len//4 estimate"
            )
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised by monkeypatch
            raise ValueError(
                f"a real token count was requested ({resolved}) but the "
                f"'tokenizers' package is not importable: {exc}. Install it "
                f"(pip install tokenizers) or pass {HEURISTIC_SPEC!r} and accept "
                "that the context budget is then estimated at len//4, which "
                "under-reads Qwen3 by ~1.85x on this state space"
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(resolved))
        self.path = resolved
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:12]
        self.identity = f"tokenizer.json:{digest}"
        # The conversation is append-only, so ``context_estimate`` re-counts
        # every earlier turn on every step: 36 turns re-tokenized 36 times is
        # quadratic in a loop that already costs real seconds.  Memoizing on the
        # text is exact -- the counter is a pure function -- and turns it linear.
        self._memo: dict[str, int] = {}

    def count(self, text: str) -> int:
        hit = self._memo.get(text)
        if hit is None:
            hit = len(self._tokenizer.encode(text, add_special_tokens=False).ids)
            self._memo[text] = hit
        return hit


def enforce_context_budget(
    *,
    used: int,
    budget: int,
    counter: TokenCounter,
    where: str,
    episode_id: str = "",
) -> None:
    """Raise if the conversation no longer fits the budget it was run under.

    Called on both halves of a step -- after the observation is appended and
    after the completion is -- because the two crossings mean different things.
    Crossing on the observation says the *state* no longer fits and no model
    could have answered it; crossing on the completion says the answer is what
    tipped it, which is a budget set too tight for the deliberation the arm
    asked for.  ``where`` is carried into the message so that the two are still
    distinguishable when a traceback is all that survives.

    ``budget <= 0`` disables the check rather than failing everything, so a
    caller that genuinely wants no ceiling can say so in one place.
    """
    if budget <= 0 or used <= budget:
        return
    raise ContextBudgetExceeded(
        f"the conversation is {used:,} tokens after the {where} and the budget "
        f"is {budget:,}"
        + (f" ({episode_id})" if episode_id else "")
        + f". Counted with {counter.identity}. This is not recoverable by "
        "trimming: the prefix is append-only and provider-cached, so dropping a "
        "turn rewrites every later prompt. Shorten the episode, shrink the "
        "state space, or raise --max-context-tokens deliberately."
    )


@lru_cache(maxsize=8)
def _cached_file_counter(path: str) -> FileTokenizerCounter:
    """One tokenizer object per path per process.

    Loading ``tokenizer.json`` is ~11 MB of parsing.  Episodes are driven one
    per process here, but the eval harness builds several policies in one, and
    paying that per policy is pure waste.
    """
    return FileTokenizerCounter(path)


def build_token_counter(spec: str | None) -> TokenCounter:
    """Turn a CLI string into the ruler the run will be measured with.

    ``None`` and ``"heuristic"`` both give the legacy estimate -- ``None``
    because that is what an un-passed flag looks like, and keeping the two
    identical means adding the flag changes nothing for callers that do not use
    it.  Anything else is treated as a path and must resolve, because the only
    other options are to guess and to fall back, and both of those end in a run
    that claims a check it did not perform.
    """
    if spec is None or spec == HEURISTIC_SPEC:
        return HeuristicTokenCounter()
    return _cached_file_counter(str(spec))
