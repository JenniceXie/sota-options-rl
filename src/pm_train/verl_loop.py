"""veRL 0.7.1 agent loop for the GRPO stage.

Why not ``tool_agent`` + ``PortfolioMonkeyInteraction`` (the path the repo's
veRL backend was written for).  Read against the installed 0.7.1 source:

* ``ToolAgentLoop.run`` generates from the dataset's ``raw_prompt`` *before*
  the interaction is first consulted, but this environment's first observation
  exists only after ``env.reset`` -- it cannot be in the dataset;
* ``_handle_interacting_state`` unpacks ``generate_response``'s fourth value
  (where the adapter puts ``nav_marks``/``start_nav``) and discards it, so the
  reward function would receive neither and raise.

So this loop drives :func:`pm_train.episode.drive` directly and sets
``AgentLoopOutput.reward_score``; veRL then writes it into ``rm_scores`` at the
last response token (``agent_loop.py`` ``_postprocess``) and the trainer uses
that instead of a reward manager.

The dataset row carries only which arm and which month; ``raw_prompt`` is a
placeholder the model never sees.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from portfolio_monkey.env.tokens import build_token_counter
from portfolio_monkey.training.reward import RUIN_REWARD

from .envfactory import arm_config, env_factory, episodes_for, read_dates
from .episode import GenOut, drive

_CACHE: dict[tuple, Any] = {}
_LOCK = threading.Lock()


def _cached(key, make):
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = make()
        return _CACHE[key]


class PMAgentLoop(AgentLoopBase):
    def __init__(self, *args, data_root: str, mark_cache_dir: str, rollout_dir: str,
                 gamma: float, max_turn_tokens: int, context_budget: int,
                 tokenizer_json: str, overflow: str = "error", env_in_subprocess: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if overflow not in ("error", "ruin"):
            raise ValueError(f"overflow must be 'error' or 'ruin', got {overflow!r}")
        self.data_root = data_root
        self.mark_cache_dir = mark_cache_dir
        self.rollout_dir = Path(rollout_dir)
        self.gamma = float(gamma)
        self.max_turn_tokens = int(max_turn_tokens)
        self.context_budget = int(context_budget)
        self.overflow = overflow
        # veRL 0.9.1 runs all G rollouts of a prompt in one worker process; with
        # the environment on threads they serialise on the GIL (6.8x slower,
        # tests/bench_concurrency.py).  One process per rollout removes that.
        self.env_in_subprocess = bool(env_in_subprocess)
        self.counter = _cached(("counter", tokenizer_json), lambda: build_token_counter(tokenizer_json))
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        info = kwargs["extra_info"]
        arm, episode_id, dates_file = info["arm"], info["episode_id"], info["dates_file"]
        config = _cached(("config", arm), lambda: arm_config(arm))
        episodes = _cached(("episodes", arm, dates_file),
                           lambda: {e.episode_id: e for e in episodes_for(config, read_dates(dates_file))})
        episode = episodes[episode_id]
        request_id = uuid.uuid4().hex
        run_dir = self.rollout_dir / arm / episode_id / f"{time.strftime('%Y%m%dT%H%M%S')}_{request_id[:8]}"

        # Which policy version produced this trajectory.  veRL 0.9.1's metrics
        # require min/max_global_steps on every sample; the server puts them in
        # TokenOutput.extra_fields on each call (first call = min, last = max).
        tags: dict[str, Any] = {}

        async def generate(prompt_ids, max_tokens):
            params = dict(sampling_params)
            params["max_tokens"] = max_tokens
            out = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=params)
            ef = dict(getattr(out, "extra_fields", None) or {})
            if "min_global_steps" in ef and "min_global_steps" not in tags:
                tags["min_global_steps"] = ef["min_global_steps"]
            if "max_global_steps" in ef:
                tags["max_global_steps"] = ef["max_global_steps"]
            return GenOut(list(out.token_ids), list(out.log_probs) if out.log_probs else None)

        label = f"{arm}__rl_{request_id[:8]}"
        factory = None
        if self.env_in_subprocess:
            from .procsession import ProcessRolloutSession
            factory = lambda: ProcessRolloutSession(
                arm_id=arm, label=label, episode_ids=[episode_id], dates_file=dates_file,
                run_dir=run_dir, data_root=self.data_root, mark_cache_dir=self.mark_cache_dir)
        out = await drive(
            session_factory=factory,
            arm=label,
            episodes=[episode],
            env_factory_for=lambda ledger: env_factory(
                config, data_root=self.data_root, mark_cache_dir=self.mark_cache_dir, ledger=ledger),
            run_dir=run_dir,
            tokenizer=self.tokenizer,
            counter=self.counter,
            generate=generate,
            max_turn_tokens=self.max_turn_tokens,
            context_budget=self.context_budget,
            gamma=self.gamma,
        )
        if out.budget_exceeded:
            if self.overflow == "error":
                raise RuntimeError(
                    f"{arm} {episode_id}: rollout exceeded the {self.context_budget}-token "
                    f"context budget ({out.context_tokens}) at turn {out.assistant_turns}")
            reward = RUIN_REWARD
        else:
            reward = float(out.discounted)
        if len(out.prompt_ids) > self.prompt_length:
            raise RuntimeError(f"first turn is {len(out.prompt_ids)} tokens > prompt_length {self.prompt_length}")
        if len(out.response_ids) > self.response_length:
            raise RuntimeError(
                f"response is {len(out.response_ids)} tokens > response_length {self.response_length}; "
                "veRL would truncate it and the reward would score tokens it never trained on")

        record = {
            "arm": arm, "episode_id": episode_id, "run_dir": out.run_dir, "pid": os.getpid(),
            "reward": reward, "log_return": out.log_return, "discounted": out.discounted,
            "budget_exceeded": out.budget_exceeded, "context_tokens": out.context_tokens,
            "prompt_tokens": len(out.prompt_ids), "response_tokens": len(out.response_ids),
            "generated_tokens": int(sum(out.response_mask)),
            "assistant_turns": out.assistant_turns, "truncated_turns": out.truncated_turns,
            "gen_seconds": out.gen_seconds, "env_seconds": out.env_seconds,
            "wall_seconds": out.wall_seconds, "finished_at": time.time(),
        }
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        with _LOCK, (self.rollout_dir / "rollouts.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        extra = {**tags, "reward_extra_info": {
            "pm_log_return": out.log_return,
            "pm_gen_seconds": out.gen_seconds,
            "pm_env_seconds": out.env_seconds,
            "pm_context_tokens": out.context_tokens,
            "pm_truncated_turns": out.truncated_turns,
            "pm_budget_exceeded": float(out.budget_exceeded),
        }}
        return AgentLoopOutput(
            prompt_ids=out.prompt_ids,
            response_ids=out.response_ids,
            response_mask=out.response_mask,
            response_logprobs=out.response_logprobs,
            reward_score=reward,
            num_turns=2 * out.assistant_turns + 1,
            metrics=AgentLoopMetrics(generate_sequences=out.gen_seconds, tool_calls=out.env_seconds),
            extra_fields=extra,
        )
