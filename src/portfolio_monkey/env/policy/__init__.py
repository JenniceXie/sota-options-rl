"""Policies: whatever turns an observation into an action line.

The environment does not know or care what is on the other side of this
interface.  That is what lets the same ``OptionsEnv``, the same resolvers and
the same ledger produce every arm in ``docs/evaluation_protocol.md`` section 6 —
the LLM arm, the rule arms and the do-nothing control differ only in which
object is passed in, so a difference in their results cannot be a difference in
their environments.

``reset`` exists separately from ``act`` because of ``R7``: the LLM policy keeps
an **append-only** conversation so that DeepSeek's prefix cache hits on every
step after the first.  The system block and the episode header are written once
at ``reset``; each step appends one user turn and one assistant turn and never
edits what came before.  A policy that rewrote its history would pay full price
for every token on every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Mapping, Protocol, runtime_checkable

from ..statespace import Observation

__all__ = ["PolicyResponse", "Policy", "HoldPolicy"]


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    """One action, plus everything the ledger wants to know about producing it.

    ``reasoning`` is stored and never fed back into the conversation.  It is
    what the run is for — the user asked for the model's thinking to be
    saved — but replaying it into the next turn would spend the context budget
    on the policy's own deliberation rather than on market state, and it would
    break the byte-identical prefix that makes caching work.
    """

    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""
    finish_reason: str = ""
    error: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    """The whole policy interface."""

    name: str

    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Start a new episode's context.  The book carries over; this does not.

        ``tools`` carries the state space's machine-readable declarations -- the
        quote schema today -- for an arm running the tool channel, and is empty
        otherwise.  It travels the same path as ``grammar`` because it *is*
        grammar: the state space chooses between printing the ``Q`` verb and
        declaring the tool, and a policy that built its own schema would be
        describing a vocabulary it does not own.

        A policy that cannot speak the tool channel must **refuse** a non-empty
        ``tools`` rather than ignore it.  Ignoring it is the one failure that
        looks healthy: under the tool channel the ``Q`` lines are gone from the
        prompt, so the arm would run a whole month with no way to ask a price
        and nothing in the ledger saying the capability was dropped.  Policies
        that ignore ``system`` and ``grammar`` too -- hold, the rule arms,
        replay -- may ignore this as well, since for them it is not a capability
        at all.
        """
        ...

    def act(self, observation: Observation) -> PolicyResponse:
        """One action line for one decision point."""
        ...


class HoldPolicy:
    """Does nothing, forever.  The control arm of section 6.

    Worth running for real rather than assuming: a book that holds cash still
    earns ``r_f``, still pays nothing in spread, and its log return is the
    number every other arm has to beat.  It is also the fastest way to prove the
    environment's plumbing works without spending an API call.
    """

    name = "hold"

    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        return None

    def act(self, observation: Observation) -> PolicyResponse:
        return PolicyResponse(text="H", model=self.name)
