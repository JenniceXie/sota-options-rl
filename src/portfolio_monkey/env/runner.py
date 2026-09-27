"""The episode driver: one policy, one book, a sequence of monthly episodes.

This is the only place that knows the *shape* of a run, and it is deliberately
thin.  Everything it does is sequencing; every number it touches was computed by
the environment or reported by the policy.  That matters because
``docs/evaluation_protocol.md`` section 6 compares arms that differ only in the
``Policy`` object handed to :func:`run_arm` — if the driver did any arithmetic,
the arms would differ in the driver too.

Three things happen here that happen nowhere else:

**The episode boundary.**  ``policy.reset`` is called once per episode and
``env.reset`` is called once per episode, but the *book* crosses the boundary
untouched (``docs/env_contract.md`` section 1.4).  The context resets; the
positions, the cash and the realized PnL do not.  This asymmetry is the entire
content of the monthly-episode decision: a fresh book each month would make the
run a sequence of independent bets whose log returns no longer sum to the run's.

**The handoff is written even though it is not needed in-process.**  ``env``
holds one ``Book`` for the whole run, so ``BookState`` is redundant when the run
completes in one process.  It is written at every boundary anyway because on PSC
a six-month run is one job that can be preempted, and a resumable run is the
difference between losing a month and losing an hour.  ``BookState.restore``
asserts NAV rather than restating it, so a corrupt handoff fails loudly.

**Failure is recorded, not raised.**  A provider outage becomes a hold with the
error in the ledger (see ``policy/deepseek.py``); a ``DataGap`` ends the run and
is written to the manifest.  A run that dies on step 400 with an exception and
no manifest is a run whose 399 good steps cannot be reported on.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Any

from .book import BookState
from .environment import DataGap, Episode, OptionsEnv, StepView
from .ledger import DecisionRow, LedgerWriter
from .policy import Policy, PolicyResponse
from .resolvers import resolver_versions
from .statespace import Observation

__all__ = ["RUNNER_VERSION", "EpisodeResult", "RunResult", "run_arm"]

#: Bumped 2026-09-24, and the bump is load-bearing rather than cosmetic.  The
#: quote round now goes through ``env.quote_observation`` instead of rendering
#: straight off the state space, so a de-identified arm no longer puts real
#: tickers in the ``R`` price block.  That changes the prompt the policy sees
#: and therefore what the run means -- but it does **not** move
#: ``env_fingerprint``, because the mask lives outside ``EnvConfig``.  Without a
#: version on the manifest, a run recorded before the fix and one recorded after
#: would be indistinguishable by any field, and the two must never be pooled.
RUNNER_VERSION = "runner.v2"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """What one month did, in the units the run is scored in."""

    episode_id: str
    steps: int
    decisions: int
    start_nav: float
    end_nav: float
    terminated: str | None
    state_path: Path | None

    @property
    def log_return(self) -> float:
        if self.start_nav <= 0 or self.end_nav <= 0:
            return 0.0
        return log(self.end_nav / self.start_nav)


@dataclass(slots=True)
class RunResult:
    """The run's own summary.  Metrics come from the ledger, not from here.

    ``log_return`` is included because it is the reward and it must be checkable
    against the panel built from T2: the two are computed from different objects
    by different code, and if they disagree, one of them is wrong.
    """

    arm: str
    track: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    start_nav: float = 0.0
    end_nav: float = 0.0
    terminated: str | None = None
    failure: str | None = None
    wall_seconds: float = 0.0

    @property
    def log_return(self) -> float:
        if self.start_nav <= 0 or self.end_nav <= 0:
            return 0.0
        return log(self.end_nav / self.start_nav)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runner_version": RUNNER_VERSION,
            "arm": self.arm,
            "track": self.track,
            "start_nav": self.start_nav,
            "end_nav": self.end_nav,
            "log_return": self.log_return,
            "terminated": self.terminated,
            "failure": self.failure,
            "wall_seconds": self.wall_seconds,
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "steps": e.steps,
                    "decisions": e.decisions,
                    "start_nav": e.start_nav,
                    "end_nav": e.end_nav,
                    "log_return": e.log_return,
                    "terminated": e.terminated,
                }
                for e in self.episodes
            ],
        }


def run_arm(
    env: OptionsEnv,
    policy: Policy,
    episodes: Sequence[Episode],
    *,
    arm: str,
    track: str = "full",
    ledger: LedgerWriter | None = None,
    state_dir: Path | str | None = None,
    carried_in: BookState | None = None,
    on_step: Callable[[StepView], None] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
) -> RunResult:
    """Walk ``episodes`` with ``policy`` and return the run's summary.

    ``carried_in`` resumes a preempted run: the book is restored and the first
    episode is told it inherited positions, so the header describes a book the
    policy did not open.  Passing a state from a *different* config is caught by
    ``BookState.restore``'s NAV assertion only by luck, so the fingerprint is
    checked here first.
    """
    if not episodes:
        raise ValueError("run_arm needs at least one episode")

    # The ledger stamps every NAV row with *its* labels, and the manifest and
    # the decision rows get these.  A disagreement would produce a panel filed
    # under one arm and a manifest describing another, which is the kind of
    # mislabel that survives all the way into a published table.
    if ledger is not None and (ledger.arm, ledger.track) != (arm, track):
        raise ValueError(
            f"ledger is labelled ({ledger.arm}, {ledger.track}) but the run is "
            f"({arm}, {track}); the panel and the manifest would disagree"
        )

    if carried_in is not None:
        fingerprint = env.config.fingerprint()
        if carried_in.env_fingerprint not in (None, fingerprint):
            raise ValueError(
                "carried-in state was captured under a different env config "
                f"({carried_in.env_fingerprint[:12]}... vs {fingerprint[:12]}...); "
                "resuming across a config change would make the run's log return "
                "the sum of two different experiments"
            )
        env.load(carried_in)

    result = RunResult(arm=arm, track=track, start_nav=env.book.nav)
    states = Path(state_dir) if state_dir is not None else None
    if states is not None:
        states.mkdir(parents=True, exist_ok=True)

    system = env.state_space.system_block(env.config)
    grammar = env.state_space.action_grammar(env.config)
    # Built once, beside the grammar, because it is the other half of the same
    # declaration: under ``quote_channel="tool"`` the state space takes the
    # ``Q`` verb out of ``grammar`` and returns it here instead.  Empty on every
    # other arm, and a policy that cannot carry it says so at ``reset`` rather
    # than running a month without it.
    schema = env.state_space.quote_tool_schema(env.config)
    tools: tuple[Mapping[str, Any], ...] = (schema,) if schema is not None else ()
    started = time.monotonic()
    carried = carried_in is not None
    crash: str | None = None

    try:
        for episode in episodes:
            outcome = _run_episode(
                env,
                policy,
                episode,
                arm=arm,
                track=track,
                ledger=ledger,
                system=system,
                grammar=grammar,
                tools=tools,
                carried=carried,
                states=states,
                on_step=on_step,
            )
            result.episodes.append(outcome)
            carried = True
            if outcome.terminated is not None:
                result.terminated = outcome.terminated
                break
    except DataGap as exc:
        # The one failure that is not the policy's and not recoverable: the
        # environment was asked for a price the data does not have.  Section
        # 11.2 forbids interpolating one, so the run stops here with everything
        # before it intact.
        result.failure = f"data gap: {exc}"
    except Exception as exc:  # noqa: BLE001 - see below
        # Anything else is a bug or a failure mode nobody anticipated, and the
        # docstring's claim above is only true if it lands here too.  Two runs
        # were lost proving it: an ``http.client.IncompleteRead`` escaped the
        # policy's retry clauses, propagated through ``act``, and ended the runs
        # with five good decisions and zero on disk that could be scored -- the
        # ledger rows were written, the manifest was not, and a run without a
        # manifest is not a result.
        #
        # This is not a swallow.  The traceback goes into the manifest, and
        # ``result.failure`` makes ``run_policy_episodes`` exit non-zero, so a
        # crash is still a crash; it is a crash whose completed steps survive.
        result.failure = f"{type(exc).__name__}: {exc}"
        crash = traceback.format_exc()

    result.end_nav = env.book.nav
    result.wall_seconds = time.monotonic() - started

    if ledger is not None:
        ledger.write_manifest(
            {
                "runner_version": RUNNER_VERSION,
                "arm": arm,
                "track": track,
                "policy": policy.name,
                "policy_model": getattr(policy, "model", ""),
                # The output budget and the effort that spends it.  A run that
                # truncated 2 of 10 steps recorded no trace of what cap it was
                # truncating against, so the manifest could not distinguish a
                # budget that was too small from a model that had nothing to
                # say -- and the operator's memory of the flag is not evidence.
                "policy_reasoning_effort": getattr(policy, "reasoning_effort", ""),
                "policy_max_completion_tokens": getattr(
                    policy, "max_completion_tokens", None
                ),
                "policy_max_context_tokens": getattr(policy, "max_context_tokens", None),
                # Which ruler that budget was measured with.  Recorded because
                # the default ruler is ``len(text) // 4``, which under-reads the
                # Qwen3 tokenizer on this state space by 1.85x: over the 378
                # episodes of the astra corpus it saw a peak of 16,594 against a
                # true 32,763 in a 32,768 budget.  Both runs report "within
                # budget"; only one of them checked. Without this key a
                # trajectory that overflows the student's context window is
                # indistinguishable from one that fits, and the difference does
                # not surface until training truncates it.
                "policy_token_counter": getattr(
                    getattr(policy, "token_counter", None), "identity", None
                ),
                # How many recent assistant turns carried their reasoning back
                # into the prompt.  Recorded beside the effort because the two
                # together decide whether a trace exists at all: at effort=high
                # and a window of 0, ``moonshotai/kimi-k3`` measured zero
                # reasoning on 18 of 20 steps while the run reported success on
                # every one of them.  A ledger that does not say which window it
                # ran under cannot tell that failure from a quiet model.
                "policy_reasoning_replay_turns": getattr(
                    policy, "reasoning_replay_turns", 0
                ),
                # Together these say whether the arm is one draw or a member of
                # a seed sweep.  Recorded as a pair because neither is
                # interpretable alone: a seed at ``temperature 0`` is inert, so
                # four such arms are one arm plus host noise, and a manifest
                # carrying only the seed would make them look like four draws.
                "policy_temperature": getattr(policy, "temperature", None),
                "policy_seed": getattr(policy, "seed", None),
                # Which hosts the model id was allowed to resolve to. Empty
                # means unpinned, which is a different arm, not a missing field.
                "policy_providers": list(getattr(policy, "providers", ()) or ()),
                "policy_provider_fallbacks": getattr(
                    policy, "allow_provider_fallbacks", None
                ),
                "state_space_id": env.state_space.state_space_id,
                "env_config_version": env.config.version,
                "env_fingerprint": env.config.fingerprint(),
                # The config itself, and not only its digest.  A digest proves
                # two runs are the same experiment; it cannot say *which*
                # experiment, and it stops being reproducible the moment a
                # field is added to ``EnvConfig`` -- adding
                # ``flags.anonymize`` on 2026-09-23 changed the digest of every
                # possible config, so no run recorded before that date can have
                # its fingerprint rebuilt by later code.  That is a property of
                # hashing a schema, not a bug, and the remedy is to store the
                # preimage: this dict is exactly what ``fingerprint()`` hashes,
                # so any later reader can verify the digest and then read the
                # flags off the object the runner actually held.
                #
                # ~2 kB per manifest.  The alternative was a per-flag mirror
                # field, which is what the four below are, and which fails in
                # precisely the way a flag added next month would repeat.
                "env_config": env.config.as_dict(),
                "resolvers": resolver_versions(),
                "universe": ",".join(env.config.universe.tradeable),
                "episodes": [e.episode_id for e in result.episodes],
                "risk_free_rate": env.config.risk_free_rate,
                # Which strategies were hedged changes what the arm *is*, not
                # just how it performed, so it belongs in provenance rather
                # than in the operator's memory of the flag they passed.
                "hedge": {
                    "enabled": env.config.hedge.enabled,
                    "hedged_families": list(env.config.hedge.hedged_families),
                    "delta_band": env.config.hedge.delta_band,
                    "portfolio_level": env.config.hedge.portfolio_level,
                    # What the resolver actually did, not just how it was set.
                    # A run with ``enabled: true`` and no share fills is either
                    # a window that never left the band or a hedge that never
                    # ran, and the config block alone cannot tell them apart.
                    **env.hedge_stats,
                },
                "mark_quotes": _mark_quote_stats(env),
                # ``unbounded_marks`` must be 0.  A package's value provably
                # cannot lie outside ``payoff.value_bounds``, so a non-zero
                # count means the valuation is wrong and every return in this
                # manifest was computed from it.  It rides here because the
                # clamp is otherwise invisible: a clamped position is recorded
                # as ``mark_quality: stale``, which an old-but-correct mark
                # carries too.
                "marks": env.mark_stats,
                # Refuse-and-count, counted.  ``None`` on a non-anonymized arm;
                # on an anonymized one a zero is the positive claim that the
                # guard ran over every step and saw nothing, which is a
                # different statement from the field being absent.
                "anonymization": env.anonymization_stats,
                # Read here, at manifest-write time, rather than handed in by
                # the caller.  ``run_policy_episodes`` passed these through
                # ``manifest_extra``, which is evaluated at the *call* -- before
                # a single decision -- so a replay that consumed its whole
                # recording filed ``replayed_decisions: 0, consumed_all: false``.
                # That field exists to tell a complete replay from a truncated
                # one, and it reported the truncated value for both.
                **(
                    {"replay": policy.replay_stats}
                    if hasattr(policy, "replay_stats")
                    else {}
                ),
                "traceback": crash,
                **result.as_dict(),
                **(manifest_extra or {}),
            }
        )
    return result


def _mark_quote_stats(env: OptionsEnv) -> dict[str, Any] | None:
    """How much of the book was marked out of chain, read after the run.

    Taken from the environment here rather than passed in by the caller because
    these are counters, not configuration: a snapshot handed over before the
    first step would faithfully record zeros.  ``None`` distinguishes a run with
    no mark source at all from one that had a source and never needed it.
    """
    stats = getattr(env.marks, "stats", None)
    if stats is None:
        return None
    # Read off the source object, not a literal here: the same string is
    # hashed into ``EnvConfig.mark_quote_source`` and checked against this
    # object at construction, and a second literal could drift from both.
    return {"source_version": env.marks.source_version, **dict(stats)}


def _run_episode(
    env: OptionsEnv,
    policy: Policy,
    episode: Episode,
    *,
    arm: str,
    track: str,
    ledger: LedgerWriter | None,
    system: str,
    grammar: str,
    tools: Sequence[Mapping[str, Any]],
    carried: bool,
    states: Path | None,
    on_step: Callable[[StepView], None] | None,
) -> EpisodeResult:
    # The header is built before ``env.reset`` because it describes the book as
    # it arrives, and ``reset`` advances to the first grid point — which settles
    # expiries and accrues interest, so the book it leaves behind is already
    # one step into the month.
    header = env.episode_header(episode, carried_in=carried)
    policy.reset(system=system, grammar=grammar, episode_header=header, tools=tools)

    start_nav = env.book.nav
    view = env.reset(episode)
    steps = 0
    decisions = 0

    while not view.done:
        action: str | None = None
        pending = view.observation
        response = None
        # The proposal turn, when there was one: (observation, response) for the
        # completion that asked for prices.  Kept separate rather than appended
        # to a list because the two turns are not interchangeable -- only the
        # second one's ``results`` describe what happened to the book.
        proposal: tuple[Observation, PolicyResponse] | None = None
        quoted: tuple[Mapping[str, Any], ...] = ()
        applied = view.point
        if pending is not None:
            response = policy.act(pending)
            action = response.text
            decisions += 1

            # A completion carrying ``Q`` buys prices instead of a trade.  It
            # is answered from the *unstepped* point, so every quote reads the
            # chain slice the fill will read one turn later; stepping first and
            # quoting after would price against tomorrow.
            quoted = env.quote(action, applied) if applied is not None else ()
            if quoted:
                proposal = (pending, response)
                # Through the environment, not through ``env.state_space``.  The
                # state space renders; the environment owns the view the policy
                # is allowed to see, and de-identification lives there.
                pending = env.quote_observation(quoted, applied)
                response = policy.act(pending)
                action = response.text
                decisions += 1

        step_index = steps
        view = env.step(action)
        steps += 1

        if ledger is not None and env.proposals:
            # Drained every step rather than only on a decision point, because
            # ``OptionsEnv.step`` clears the list and a proposal that is not
            # taken here is gone.
            ledger.write_strategies(
                env.proposals, episode_id=episode.episode_id, step_index=step_index
            )

        if response is not None and ledger is not None and applied is not None:
            # The proposal turn is written first so the file reads in the order
            # the conversation happened.  It carries ``quoted`` as its results
            # -- the prices it was answered with -- and not ``view.results``,
            # which belong to the action that followed it.  Giving both rows
            # the same results would make a quote look like it traded.
            if proposal is not None:
                proposal_obs, proposal_response = proposal
                ledger.write_decision(
                    DecisionRow(
                        arm=arm,
                        track=track,
                        step_ts=applied.timestamp.isoformat(),
                        episode_id=episode.episode_id,
                        step_index=step_index,
                        state_space_id=env.state_space.state_space_id,
                        observation=proposal_obs.text,
                        completion=proposal_response.text,
                        reasoning=proposal_response.reasoning,
                        results=tuple(quoted),
                        turn="quote",
                        prompt_tokens=proposal_response.prompt_tokens,
                        completion_tokens=proposal_response.completion_tokens,
                        latency_seconds=proposal_response.latency_seconds,
                        model=proposal_response.model,
                        finish_reason=proposal_response.finish_reason,
                        error=proposal_response.error or "",
                        extra=dict(proposal_response.extra),
                    )
                )
            ledger.write_decision(
                DecisionRow(
                    arm=arm,
                    track=track,
                    step_ts=applied.timestamp.isoformat(),
                    episode_id=episode.episode_id,
                    step_index=step_index,
                    state_space_id=env.state_space.state_space_id,
                    observation=pending.text if pending is not None else "",
                    completion=response.text,
                    reasoning=response.reasoning,
                    results=view.results,
                    turn="act",
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_seconds=response.latency_seconds,
                    model=response.model,
                    finish_reason=response.finish_reason,
                    error=response.error or "",
                    extra=dict(response.extra),
                )
            )
        if on_step is not None:
            on_step(view)

    state_path = None
    if states is not None:
        state_path = states / f"{episode.episode_id}.json"
        state_path.write_text(
            json.dumps(env.state().as_dict(), indent=2), encoding="utf-8"
        )

    return EpisodeResult(
        episode_id=episode.episode_id,
        steps=steps,
        decisions=decisions,
        start_nav=start_nav,
        end_nav=env.book.nav,
        terminated=env.terminated,
        state_path=state_path,
    )


def load_state(path: Path | str) -> BookState:
    """Read a handoff written by :func:`run_arm`."""
    return BookState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
