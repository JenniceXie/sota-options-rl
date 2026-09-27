"""Drive one ``RolloutSession`` with any token generator.

Framework-neutral on purpose: RL (a veRL agent loop), evaluation (an offline
vLLM engine) and the byte-for-byte corpus test all call :func:`drive` with a
different ``generate``.  So the conversation the student is trained on in RL is
assembled by the same code as the one it is evaluated on.

What this adds on top of ``RolloutSession``, and why:

* **Token assembly.**  ``prompt_ids`` is the first rendered turn (system block +
  grammar, header + first observation, generation prompt).  Everything after
  it -- generated tokens (mask 1) and environment observations (mask 0) -- is
  the response, which is the shape veRL's GRPO trainer consumes.
* **The context budget.**  ``QueuePolicy`` does not enforce
  ``max_context_tokens``; the API policies do, fatally, counting content tokens
  (``BedrockPolicy.context_estimate``).  The same rule is applied here so an RL
  rollout cannot run in a looser environment than the teacher did.
* **Its own ledger.**  ``RolloutSession`` with no ledger returns *zero* NAV marks
  and ``compute_score`` then returns 0.0 for every trajectory, silently
  (``RunResult`` has no ``nav_rows``).  Every rollout here writes a ledger, the
  reward is read from its ``nav_panel.jsonl``, and the undiscounted return is
  asserted equal to ``run_arm``'s own log return.
* **The time split** (deliverable 1): seconds awaiting the generator vs seconds
  awaiting the environment, per rollout.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from portfolio_monkey.env.ledger import LedgerWriter
from portfolio_monkey.training.agent import QueuePolicy, RolloutSession
from portfolio_monkey.training.reward import (
    discounted_return,
    load_trajectory,
    step_rewards,
    undiscounted_return,
)

from .chat import GENERATION_PROMPT, IM_END, render_message


@dataclass
class GenOut:
    token_ids: list[int]
    log_probs: list[float] | None = None


Generate = Callable[[list[int], int], Awaitable[GenOut]]


class BudgetExceeded(RuntimeError):
    pass


class EpisodicQueuePolicy(QueuePolicy):
    """``QueuePolicy`` that numbers turns per episode, not per run.

    The stock policy sends the system block and header only on the run's very
    first turn (``_index == 0``) and never resets the counter, so in a
    multi-episode run -- evaluation carries the book across six months -- the
    driver could not see that ``run_arm`` had reset the context.  ``reset`` is
    called once per episode, so restarting the count there makes every
    episode's first turn carry its own system block and header.
    """

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self._index = 0


@dataclass
class EpisodeOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    messages: list[dict[str, str]]
    run_dir: str
    log_return: float = float("nan")
    discounted: float | None = None
    context_tokens: int = 0
    assistant_turns: int = 0
    truncated_turns: int = 0
    gen_seconds: float = 0.0
    env_seconds: float = 0.0
    wall_seconds: float = 0.0
    budget_exceeded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


async def drive(
    *,
    arm: str,
    episodes: Sequence[Any],
    env_factory_for: Callable[[LedgerWriter], Callable[[], Any]],
    run_dir: str | Path,
    tokenizer,
    counter,
    generate: Generate,
    max_turn_tokens: int,
    context_budget: int,
    gamma: float | None,
    carried_in=None,
    manifest_extra: dict | None = None,
    turn_timeout: float = 1800.0,
    multi_episode: bool = False,
    on_episode_end: Callable[[list[dict[str, str]], int], None] | None = None,
    session_factory: Callable[[], Any] | None = None,
) -> EpisodeOutput:
    """``multi_episode`` is for evaluation only: the context restarts at each
    episode and only the final episode's tokens are returned (they are not
    trained on).  ``on_episode_end(messages, context_tokens)`` sees each
    finished episode's conversation."""
    loop = asyncio.get_running_loop()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if session_factory is not None:
        # The environment, its ledger and its RolloutSession live in a child
        # process (procsession.ProcessRolloutSession); only turns cross over.
        ledger = None
        session = session_factory()
    else:
        ledger = LedgerWriter(run_dir, arm=arm, track="full")
        run_kwargs: dict[str, Any] = {}
        if carried_in is not None:
            run_kwargs["carried_in"] = carried_in
        if manifest_extra:
            run_kwargs["manifest_extra"] = manifest_extra
        session = RolloutSession(
            env_factory_for(ledger),
            episodes,
            arm=arm,
            ledger_factory=lambda: ledger,
            state_dir=run_dir / "state",
            timeout=turn_timeout,
            run_kwargs=run_kwargs,
        )
        session._policy = EpisodicQueuePolicy(session._out, session._in, turn_timeout)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    t0 = time.perf_counter()
    env_s = 0.0
    gen_s = 0.0

    async def env_call(fn, *args):
        nonlocal env_s
        s = time.perf_counter()
        try:
            return await loop.run_in_executor(None, fn, *args)
        finally:
            env_s += time.perf_counter() - s

    prompt_ids: list[int] = []
    response_ids: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    have_logprobs = True
    messages: list[dict[str, str]] = []
    used = 0
    turns = truncated = 0
    exceeded = False
    try:
        turn = await env_call(session.start)
        while turn is not None:
            if turn.is_first:
                # A new episode.  run_arm resets the policy per episode, so the
                # context restarts; training rollouts are one episode each.
                if messages:
                    if not multi_episode:
                        raise RuntimeError("a second episode in a training rollout")
                    if on_episode_end is not None:
                        on_episode_end(messages, used)
                    messages, prompt_ids, response_ids, mask, logprobs = [], [], [], [], []
                system = f"{turn.system}\n\n{turn.grammar}".strip()
                first = f"{turn.header}\n{turn.text}" if turn.header else turn.text
                messages += [{"role": "system", "content": system},
                             {"role": "user", "content": first}]
                used = counter.count(system) + counter.count(turn.header) + counter.count(turn.text)
                prompt_ids = enc(render_message("system", system)
                                 + render_message("user", first) + GENERATION_PROMPT)
            else:
                messages.append({"role": "user", "content": turn.text})
                used += counter.count(turn.text)
                piece = enc(render_message("user", turn.text) + GENERATION_PROMPT)
                response_ids += piece
                mask += [0] * len(piece)
                logprobs += [0.0] * len(piece)
            if used > context_budget:
                exceeded = True
                raise BudgetExceeded(
                    f"{arm}: context {used} > {context_budget} content tokens at turn {turns}"
                )
            s = time.perf_counter()
            out = await generate(prompt_ids + response_ids, max_turn_tokens)
            gen_s += time.perf_counter() - s
            toks = list(out.token_ids)
            if out.log_probs is None:
                have_logprobs = False
            lps = list(out.log_probs or [0.0] * len(toks))
            ended = bool(toks) and toks[-1] == im_end_id
            if not ended:
                truncated += 1
            text = tokenizer.decode(toks, skip_special_tokens=True)
            response_ids += toks
            mask += [1] * len(toks)
            logprobs += lps
            # Template framing the model did not produce: the closing tag of a
            # truncated turn, and the newline after every <|im_end|>.  Mask 0.
            tail = ([] if ended else [im_end_id]) + newline_ids
            response_ids += tail
            mask += [0] * len(tail)
            logprobs += [0.0] * len(tail)
            messages.append({"role": "assistant", "content": text})
            used += counter.count(text)
            turns += 1
            turn = await env_call(session.respond, text)
        if getattr(session, "remote", False):
            from types import SimpleNamespace
            result = SimpleNamespace(**session.result())
        else:
            result = _run_result(session)
        if on_episode_end is not None and messages:
            on_episode_end(messages, used)
    except BaseException:
        session.close()
        if ledger is not None:
            ledger.close()
        if exceeded:
            out = EpisodeOutput(prompt_ids, response_ids, mask, logprobs if have_logprobs else None,
                                messages, str(run_dir), context_tokens=used, assistant_turns=turns,
                                truncated_turns=truncated, gen_seconds=gen_s, env_seconds=env_s,
                                wall_seconds=time.perf_counter() - t0, budget_exceeded=True,
                                extra={"episode_turns": sum(m["role"] == "assistant" for m in messages)})
            return out
        raise
    if ledger is not None:
        ledger.close()

    traj = load_trajectory(run_dir, track="full")
    rewards = step_rewards(traj)
    undiscounted = undiscounted_return(rewards)
    if not (math.isfinite(undiscounted) and math.isfinite(result.log_return)
            and abs(undiscounted - result.log_return) <= 1e-9):
        raise RuntimeError(
            f"{arm}: reward stream says {undiscounted:+.12f} but run_arm says "
            f"{result.log_return:+.12f}; training and evaluation disagree on the return"
        )
    return EpisodeOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=mask,
        response_logprobs=logprobs if have_logprobs else None,
        messages=messages,
        run_dir=str(run_dir),
        log_return=float(result.log_return),
        discounted=None if gamma is None else discounted_return(rewards, gamma),
        context_tokens=used,
        assistant_turns=turns,
        truncated_turns=truncated,
        gen_seconds=gen_s,
        env_seconds=env_s,
        wall_seconds=time.perf_counter() - t0,
    )


def _run_result(session: RolloutSession):
    """``session.result()``'s refusals, without its ``_marks``.

    ``_marks`` reads ``<state_dir>/nav_panel.jsonl`` while the ledger writes one
    level up, so with a state dir it raises ``FileNotFoundError`` and without one
    it returns no marks (reward 0.0).  The reward here is read from the ledger
    directly, so only the run result and its failure checks are needed.
    """
    from portfolio_monkey.training.agent import RolloutError

    if not session._finished:
        raise RolloutError("episode has not ended")
    if session._error is not None:
        raise RolloutError(f"rollout crashed: {session._error}") from session._error
    run = session._run_result
    if run is None:
        raise RolloutError("worker produced no RunResult (a crashed environment)")
    if getattr(run, "failure", None):
        raise RolloutError(f"environment failure: {run.failure}")
    return run


def write_messages(out: EpisodeOutput, path: str | Path) -> None:
    Path(path).write_text(json.dumps(out.messages), encoding="utf-8")
