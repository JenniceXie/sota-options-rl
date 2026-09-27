"""Smoke gate: 1 card, 2 optimizer steps per stage, RL on 1 episode at G=2.

Uses the repo's SINGLE_GPU_SMOKE with exactly the changes a 26k-token episode
forces, each named:

* sft ``data.max_token_len_per_gpu`` 8192 -> max_seq_len: a dynamic batch cannot
  hold one example longer than its token cap;
* rl ``data.train_batch_size`` / ``ppo_mini_batch_size`` 2 -> 1: one episode;
* rl ``gpu_memory_utilization`` 0.4 of 183 GB is ample and is kept.

Writes the three invocations (smoke settings) to <work>/plan.json and prints
the two it runs.  gamma here is a PLACEHOLDER for the smoke, not a ruling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from portfolio_monkey.training.backends.base import Launcher
from portfolio_monkey.training.backends.verl import SINGLE_GPU_SMOKE
from pm_train.driver import plan, write_agent_loop_yaml, write_rl_dataset

HOME = Path.home()
PKG = HOME / "rl_package_v1"
WORK = HOME / "work" / "smoke"
MODEL = str(HOME / "models" / "Qwen3-4B-pm")
MAX_SEQ = 33_280
GAMMA_PLACEHOLDER = 0.99


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(PKG / "sft_corpus" / "train.parquet")
    sft_data = WORK / "sft_2ep.parquet"
    corpus.iloc[:2].to_parquet(sft_data)

    rl_data, yaml = {}, {}
    for arm in ("sys_sft_rl", "s4_none"):
        p = WORK / f"rl_{arm}.parquet"
        write_rl_dataset(p, arm=arm, dates_file=str(PKG / "code/configs/dates_rl.txt"),
                         months=["ep_2024_12"])
        rl_data[arm] = str(p)
        yaml[arm] = write_agent_loop_yaml(
            WORK / f"agent_loop_{arm}.yaml",
            data_root=str(PKG / "data"), mark_cache_dir=str(HOME / "work" / "markquotes"),
            rollout_dir=str(WORK / arm / "rollouts"), gamma=GAMMA_PLACEHOLDER,
            max_turn_tokens=256, context_budget=32_768,
            tokenizer_json=str(Path(MODEL) / "tokenizer.json"), overflow="error")

    sft_hp = {**SINGLE_GPU_SMOKE["sft"], "data.max_token_len_per_gpu": MAX_SEQ,
              "trainer.total_epochs": 2, "optim.lr": 1e-5}
    rl_hp = {
        **SINGLE_GPU_SMOKE["rl"],
        "data.train_batch_size": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 1,
        "data.max_prompt_length": 8192,
        "data.max_response_length": 28672,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 8192 + 28672,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 8192 + 28672,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 8192 + 28672,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.agent.num_workers": 2,
        "reward.custom_reward_function.reward_kwargs.gamma": GAMMA_PLACEHOLDER,
        "trainer.val_before_train": False,
        "trainer.total_epochs": 2,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "trainer.test_freq": -1,
    }
    launcher = Launcher(python=str(HOME / "verl-env/bin/python"), nodes=1, gpus_per_node=1,
                        work_dir=str(WORK))
    invocations = plan(work=WORK, model=MODEL, sft_data=str(sft_data), rl_data=rl_data,
                       max_seq_len=MAX_SEQ, group_size=2, sft_hp=sft_hp, rl_hp=rl_hp,
                       launcher=launcher, agent_loop_yaml=yaml)
    (WORK / "plan.json").write_text(json.dumps({k: v.as_dict() for k, v in invocations.items()}, indent=2))
    for arm, inv in invocations.items():
        (WORK / f"cmd_{arm}.sh").write_text(inv.command + "\n")
        print(arm, "->", inv.expected_checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
