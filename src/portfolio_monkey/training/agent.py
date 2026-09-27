"""The rollout session: this repo's environment, driven one turn at a time.

WHY THIS IS AN INVERSION AND NOT A NEW HARNESS.  There are already two harnesses
in this project -- the eval runner and the per-name SFT job -- and their
divergence is a recorded problem, not a feature.  A third one, written to suit
an RL framework, would make the RL arm incomparable to the evaluation it is
scored by, which is the single most expensive mistake available here.

So this module adds **no** episode logic.  It runs the real
:func:`portfolio_monkey.env.runner.run_arm` on a worker thread, with a
:class:`QueuePolicy` standing where the LLM policy normally stands.  ``run_arm``
renders the observation, runs the quote round, applies the anonymisation mask,
resolves orders, marks the book and writes the ledger exactly as it does for
every other arm; the only difference is that the policy's ``act`` hands the
prompt out through a queue and blocks for the framework's text instead of
calling an API.  If the environment changes, the RL rollout changes with it,
because it *is* the environment.

THE SHAPE THE FRAMEWORKS WANT.  veRL asks an interaction for
``generate_response(instance_id, messages) -> (should_terminate, content,
turn_score, extra)`` (``verl/interactions/base.py``), and Miles' equivalent is
unknown.  Both reduce to start / respond / finalize, so that is the interface
here and the veRL adapter at the bottom is thirty lines.  Nothing in this file
imports a framework.

THE TRAP THIS MODULE EXISTS TO AVOID.  ``run_arm`` catches its own exceptions
and records them as a ``failure`` on the result -- it does not re-raise.  On a
worker thread that means a crashed environment looks exactly like a finished
episode: the queue simply stops producing turns.  :class:`RolloutSession`
therefore treats "the worker stopped without a result" as an error, and
surfaces ``RunResult.failure`` as :class:`RolloutError` rather than returning a
trajectory whose NAV series stops in the middle of the month.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RolloutError",
    "RolloutResult",
    "RolloutSession",
    "Turn",
    "PortfolioMonkeyInteraction",
]

#: A turn that is never answered would hang a rollout worker forever, and a
#: hung worker is indistinguishable from a slow one.  Seconds.
DEFAULT_TIMEOUT = 600.0

_SENTINEL = object()


class RolloutError(RuntimeError):
    """The environment could not produce, or could not finish, a trajectory."""


@dataclass(frozen=True)
class Turn:
    """One prompt the policy is expected to answer.

    ``system`` and ``header`` are populated on the **first** turn only.  The
    conversation is append-only by design -- the state space guarantees those
    two blocks are byte-identical across steps so a prefix cache bills only the
    delta -- and re-sending them per turn would both cost tokens and break that
    guarantee.
    """

    index: int
    kind: str
    text: str
    system: str = ""
    grammar: str = ""
    header: str = ""
    tools: tuple[Mapping[str, Any], ...] = ()
    missing: tuple[str, ...] = ()
    token_estimate: int = 0

    @property
    def is_first(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class RolloutResult:
    """What the trajectory earned, in the form :mod:`..reward` consumes."""

    arm: str
    start_nav: float
    end_nav: float
    log_return: float
    #: ``{"episode_id", "step_ts", "nlv", "transaction_cost"}`` per decision.
    nav_marks: tuple[Mapping[str, Any], ...]
    turn_kinds: tuple[str, ...]
    terminated: str | None = None
    run_dir: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def reward_extra_info(self) -> dict[str, Any]:
        """Exactly what :func:`..reward.compute_score` needs, and nothing else."""
        return {
            "arm": self.arm,
            "start_nav": self.start_nav,
            "nav_marks": [dict(m) for m in self.nav_marks],
            "terminated": self.terminated,
        }


class QueuePolicy:
    """A :class:`~portfolio_monkey.env.policy.Policy` whose brain is elsewhere.

    It satisfies the protocol structurally -- ``name``, ``reset``, ``act`` --
    so ``run_arm`` cannot tell it apart from the Bedrock or OpenRouter policies.
    ``reset`` captures the system block, grammar and header rather than
    discarding them, because the framework needs them to build the first
    message and only the environment knows what they say.

    A non-empty ``tools`` is **carried**, not ignored: under the tool quote
    channel the ``Q`` verb is absent from the grammar, so a policy that dropped
    the schema would run a whole month unable to ask a price, with nothing in
    the ledger recording that the capability was lost.
    """

    name = "rollout_queue"

    def __init__(self, outbound: queue.Queue, inbound: queue.Queue, timeout: float):
        self._out = outbound
        self._in = inbound
        self._timeout = timeout
        self._index = 0
        self.system = ""
        self.grammar = ""
        self.header = ""
        self.tools: tuple[Mapping[str, Any], ...] = ()
        self.turn_kinds: list[str] = []

    def reset(self, *, system: str, grammar: str, episode_header: str,
              tools: Sequence[Mapping[str, Any]] = ()) -> None:
        self.system = system
        self.grammar = grammar
        self.header = episode_header
        self.tools = tuple(tools)

    def act(self, observation) -> Any:
        from ..env.policy import PolicyResponse

        kind = "quote" if self._is_quote(observation) else "act"
        turn = Turn(
            index=self._index,
            kind=kind,
            text=observation.text,
            system=self.system if self._index == 0 else "",
            grammar=self.grammar if self._index == 0 else "",
            header=self.header if self._index == 0 else "",
            tools=self.tools,
            missing=tuple(observation.missing),
            token_estimate=int(observation.token_estimate or 0),
        )
        self.turn_kinds.append(kind)
        self._index += 1
        self._out.put(turn)
        text = self._in.get(timeout=self._timeout)
        if text is _SENTINEL:
            # The caller abandoned the rollout. Raising here unwinds run_arm,
            # which records it as a failure and lets the worker thread exit
            # instead of leaking for the life of the process.
            raise RolloutError("rollout session closed while a turn was pending")
        return PolicyResponse(text=str(text), model=self.name)

    @staticmethod
    def _is_quote(observation) -> bool:
        """Which round this is, asked of the observation rather than a counter.

        Not read off a recorded ``turn`` field: in the ledger that label does
        not sit on the observation it describes, and a rollout that mislabels
        its turns misaligns every per-turn reward after the first quote round.
        """
        blocks = getattr(observation, "blocks", {}) or {}
        provenance = getattr(observation, "provenance", {}) or {}
        if "turn" in provenance:
            return str(provenance["turn"]) == "quote"
        return "quotes" in blocks or "R" in blocks


class RolloutSession:
    """Start an episode, answer its turns, collect the trajectory.

    Typical use, and the only supported ordering::

        session = RolloutSession(env_factory, episodes, arm="rl_rollout_0")
        turn = session.start()
        while turn is not None:
            turn = session.respond(model_text(turn))
        result = session.result()

    ``env_factory`` is a callable rather than an environment because a group of
    ``G`` rollouts shares nothing: two sessions holding one ``OptionsEnv`` would
    trade against each other's book.
    """

    def __init__(
        self,
        env_factory,
        episodes: Sequence[Any],
        *,
        arm: str,
        track: str = "full",
        ledger_factory=None,
        state_dir: Path | str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        run_kwargs: Mapping[str, Any] | None = None,
    ):
        self._env_factory = env_factory
        self._episodes = list(episodes)
        self._arm = arm
        self._track = track
        self._ledger_factory = ledger_factory
        self._state_dir = state_dir
        self._timeout = timeout
        self._run_kwargs = dict(run_kwargs or {})
        self._out: queue.Queue = queue.Queue()
        self._in: queue.Queue = queue.Queue()
        self._policy = QueuePolicy(self._out, self._in, timeout)
        self._thread: threading.Thread | None = None
        self._run_result: Any = None
        self._error: BaseException | None = None
        self._env: Any = None
        self._started = False
        self._finished = False

    # -- driving ---------------------------------------------------------

    def start(self) -> Turn | None:
        if self._started:
            raise RolloutError("session already started")
        self._started = True
        self._thread = threading.Thread(target=self._work, name=f"rollout-{self._arm}",
                                        daemon=True)
        self._thread.start()
        return self._next_turn()

    def respond(self, text: str) -> Turn | None:
        if not self._started:
            raise RolloutError("respond() before start()")
        if self._finished:
            raise RolloutError("respond() after the episode ended")
        self._in.put(text)
        return self._next_turn()

    def close(self) -> None:
        """Abandon a rollout without leaking the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            self._in.put(_SENTINEL)
            self._thread.join(timeout=self._timeout)
        self._finished = True

    # -- result ----------------------------------------------------------

    def result(self) -> RolloutResult:
        if not self._finished:
            raise RolloutError("result() before the episode ended")
        if self._error is not None:
            raise RolloutError(f"{self._arm}: rollout crashed: {self._error}") from self._error
        run = self._run_result
        if run is None:
            # run_arm records its own exceptions instead of raising, so a dead
            # worker and a finished one look the same from out here.
            raise RolloutError(
                f"{self._arm}: the worker produced no RunResult. This is what a "
                "crashed environment looks like from the outside -- it is not "
                "an empty trajectory."
            )
        failure = getattr(run, "failure", None)
        if failure:
            raise RolloutError(f"{self._arm}: environment failure: {failure}")
        return self._collect(run)

    # -- internals -------------------------------------------------------

    def _next_turn(self) -> Turn | None:
        while True:
            try:
                item = self._out.get(timeout=self._timeout)
            except queue.Empty as exc:
                self.close()
                raise RolloutError(
                    f"{self._arm}: no turn within {self._timeout}s. A hung worker "
                    "is indistinguishable from a slow one, so this is an error."
                ) from exc
            if item is _SENTINEL:
                self._finished = True
                return None
            return item

    def _work(self) -> None:
        from ..env.runner import run_arm

        try:
            self._env = self._env_factory()
            ledger = self._ledger_factory() if self._ledger_factory is not None else None
            self._run_result = run_arm(
                self._env,
                self._policy,
                self._episodes,
                arm=self._arm,
                track=self._track,
                ledger=ledger,
                state_dir=self._state_dir,
                **self._run_kwargs,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in result()
            self._error = exc
        finally:
            self._out.put(_SENTINEL)

    def _collect(self, run) -> RolloutResult:
        marks = tuple(self._marks(run))
        end_nav = marks[-1]["nlv"] if marks else float(getattr(run, "start_nav", 0.0))
        return RolloutResult(
            arm=self._arm,
            start_nav=float(run.start_nav),
            end_nav=float(end_nav),
            log_return=float(run.log_return),
            nav_marks=marks,
            turn_kinds=tuple(self._policy.turn_kinds),
            terminated=getattr(run, "terminated", None),
            run_dir=str(self._state_dir or ""),
            extra={"episodes": len(self._episodes)},
        )

    def _marks(self, run):
        """Decision marks, from the ledger the run just wrote.

        Taken from the NAV panel rather than from per-position profits: a trip's
        profit double-counts a roll and ignores the cost of the leg that
        replaced it, and the account mark is the only quantity that closes.
        """
        rows = getattr(run, "nav_rows", None)
        if rows is None and self._state_dir:
            from .reward import load_trajectory

            traj = load_trajectory(self._state_dir, track=self._track)
            return [
                {"episode_id": e, "step_ts": t, "nlv": n, "transaction_cost": c}
                for e, t, n, c in traj.marks
            ]
        out = []
        for row in rows or ():
            if not row.get("is_decision_point"):
                continue
            out.append({
                "episode_id": row.get("episode_id", ""),
                "step_ts": row.get("step_ts", ""),
                "nlv": float(row["nlv"]),
                "transaction_cost": float(row.get("cost_half_spread") or 0.0)
                + float(row.get("cost_fees") or 0.0),
            })
        return out


# --------------------------------------------------------------------------
# veRL adapter
# --------------------------------------------------------------------------


class PortfolioMonkeyInteraction:
    """veRL's ``BaseInteraction`` shape, satisfied structurally.

    It does **not** subclass ``verl.interactions.base.BaseInteraction``, because
    veRL is not importable on the machine where this package is developed and an
    import that only works inside the training container would make every test
    here unrunnable.  ``initialize_interactions_from_config`` constructs the
    class by dotted path and calls it with ``config=`` -- no ``isinstance``
    check is involved -- so structural compliance is sufficient.  If a future
    veRL adds that check, this class grows a subclass shim and nothing else
    moves.

    ``session_factory`` is injected rather than built here: what an episode
    *is* -- which dates, which universe, which sizing rule -- belongs to the
    experiment matrix, not to a framework adapter.
    """

    def __init__(self, config: Mapping[str, Any] | None = None, session_factory=None):
        self.config = dict(config or {})
        self.name = self.config.get("name", "portfolio_monkey")
        self._session_factory = session_factory
        self._sessions: dict[str, RolloutSession] = {}
        self._results: dict[str, RolloutResult] = {}

    async def start_interaction(self, instance_id: str, **kwargs: Any) -> str:
        if self._session_factory is None:
            raise RolloutError(
                "PortfolioMonkeyInteraction has no session_factory. Constructing "
                "a default episode here would invent the experiment's dates."
            )
        session = self._session_factory(instance_id=instance_id, **kwargs)
        self._sessions[instance_id] = session
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> tuple[bool, str, float, dict[str, Any]]:
        session = self._sessions[instance_id]
        text = _last_assistant(messages)
        turn = session.start() if text is None else session.respond(text)
        if turn is not None:
            return False, turn.text, 0.0, {"turn_kind": turn.kind}
        result = session.result()
        self._results[instance_id] = result
        # turn_score stays 0.0: the trajectory reward is computed by
        # reward.compute_score from the NAV path, and returning a per-turn PnL
        # here as well would have the framework count the same dollars twice.
        return True, "", 0.0, result.reward_extra_info()

    async def calculate_score(self, instance_id: str, **kwargs: Any) -> float:
        result = self._results.get(instance_id)
        return 0.0 if result is None else float(result.log_return)

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        session = self._sessions.pop(instance_id, None)
        if session is not None:
            session.close()
        self._results.pop(instance_id, None)


def _last_assistant(messages: Sequence[Mapping[str, Any]]) -> str | None:
    for message in reversed(list(messages)):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return None
