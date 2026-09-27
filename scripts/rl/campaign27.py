"""27B campaign: 3 jobs, one seed, Qwen3.8-27B, 8 GPUs, veRL 0.9.1.  Generated from
campaign.py; differences are marked 27B.

Writes <work>/plan.json and cmd_<arm>.sh for all three jobs.  RL jobs read
init_checkpoint from the SFT invocation's expected_checkpoint (driver.plan).

Hyperparameters as agreed 2026-09-25 (SFT lr 1e-5 per user, revised from 1e-6; the rest as
proposed and not changed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from portfolio_monkey.training.backends.base import Launcher
from pm_train.driver import plan, write_agent_loop_yaml, write_rl_dataset

HOME = Path.home()
PKG = HOME / "rl_package_v1"
import os
WORK = Path(os.environ.get("CAMPAIGN_WORK", str(HOME / "work" / "campaign27")))  # 27B
MODEL = str(HOME / "models" / "Qwen3.8-27B-pm")  # 27B
MAX_SEQ = 33_280
GAMMA = 0.99
G = 8
GPUS = 8
MAX_TURN_TOKENS = 256
RL_STEPS = int(os.environ.get("RL_STEPS", "40"))
RL_SAVE_FREQ = int(os.environ.get("RL_SAVE_FREQ", "10"))
# Round 2: training-time overflow threshold with a 2,048-token margin under the
# environment's 32,768 (eval keeps 32,768, fatal).  See REPORT.md.
TRAIN_CONTEXT_BUDGET = 30_720
MONTHS = 3  # ep_2024_12, ep_2025_01, ep_2025_02

SFT_HP = {
    "optim.lr": 1e-5,
    "optim.lr_scheduler_type": "cosine",
    "optim.lr_warmup_steps_ratio": 0.03,
    "optim.weight_decay": 0.01,
    "optim.clip_grad": 1.0,
    "trainer.total_epochs": 3,
    "data.train_batch_size": 16,
    "data.micro_batch_size_per_gpu": 1,
    "data.use_dynamic_bsz": True,
    "engine.strategy": "fsdp2",
    "model.enable_gradient_checkpointing": True,
    "trainer.save_freq": 75,  # 27B: 414/16 -> 25 steps/epoch x 3; save the final step only
    "checkpoint.save_contents": ["model", "extra", "hf_model"],  # 27B: no optimizer state (~430 GB), no resume needed
    "trainer.logger": ["console"],
}

RL_HP = {
    "data.train_batch_size": MONTHS,
    "data.max_prompt_length": 8192,
    "data.max_response_length": 28672,
    "data.shuffle": False,
    "actor_rollout_ref.actor.ppo_mini_batch_size": MONTHS,
    "actor_rollout_ref.actor.ppo_epochs": 1,
    # NOT dynamic batching.  veRL 0.7.1 dp_actor.compute_log_prob calls
    # prepare_dynamic_batch without dp_group, so same_micro_num_in_dp never
    # applies; when two short sequences pack into one micro-batch on some
    # ranks, ranks disagree on the micro-batch count and FSDP deadlocks
    # (observed: step 2, rank 0 alone at 100% for 8+ minutes).  One sequence
    # per micro-batch makes the count 24/8 = 3 on every rank by construction.
    "actor_rollout_ref.actor.use_dynamic_bsz": False,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": False,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.optim.lr": float(os.environ.get("RL_LR", "5e-7")),  # round-3 recipe
    "actor_rollout_ref.actor.clip_ratio": 0.2,
    "actor_rollout_ref.actor.use_kl_loss": True,
    "actor_rollout_ref.actor.kl_loss_coef": float(os.environ.get("RL_KL", "0.01")),  # round-3 recipe
    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff": 0.0,
    "actor_rollout_ref.model.enable_gradient_checkpointing": True,
    "actor_rollout_ref.model.use_remove_padding": True,
    "actor_rollout_ref.actor.strategy": "fsdp2",
    "actor_rollout_ref.ref.strategy": "fsdp2",
    "actor_rollout_ref.ref.fsdp_config.param_offload": False,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
        # 27B: text-only view of the VLM (see pm_train.chat.text_only_view)
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only": True,  # 27B
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.5,
    "actor_rollout_ref.rollout.temperature": 1.0,
    "actor_rollout_ref.rollout.top_p": 1.0,
    "actor_rollout_ref.rollout.agent.num_workers": MONTHS * G,
    "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "extra", "hf_model"],  # 27B: no optimizer state
    "algorithm.use_kl_in_reward": False,
    "reward.custom_reward_function.reward_kwargs.gamma": GAMMA,
    "trainer.total_training_steps": RL_STEPS,
    "trainer.total_epochs": RL_STEPS,
    "trainer.save_freq": RL_SAVE_FREQ,
    "trainer.test_freq": -1,
    "trainer.val_before_train": False,
    "trainer.logger": ["console"],
}


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rl_data, yaml = {}, {}
    for arm in ("sys_sft_rl", "s4_none"):
        p = WORK / f"rl_{arm}.parquet"
        months = write_rl_dataset(p, arm=arm, dates_file=str(PKG / "code/configs/dates_rl.txt"))
        assert len(months) == MONTHS, months
        rl_data[arm] = str(p)
        yaml[arm] = write_agent_loop_yaml(
            WORK / f"agent_loop_{arm}.yaml",
            data_root=str(PKG / "data"), mark_cache_dir=str(HOME / "work" / "markquotes"),
            rollout_dir=str(WORK / arm / "rollouts"), gamma=GAMMA,
            max_turn_tokens=MAX_TURN_TOKENS, context_budget=TRAIN_CONTEXT_BUDGET,
            tokenizer_json=str(Path(MODEL) / "tokenizer.json"), overflow="ruin",
            env_in_subprocess=True)  # 27B/veRL 0.9.1: one env process per rollout (GIL)
    launcher = Launcher(python=str(HOME / "verl91-env/bin/python"), nodes=1, gpus_per_node=GPUS,  # 27B
                        work_dir=str(WORK))
    invocations = plan(work=WORK, model=MODEL, sft_data=str(HOME / "work" / "corpus_qwen38" / "train.parquet"),  # 27B: 414 episodes
                       rl_data=rl_data, max_seq_len=MAX_SEQ, group_size=G, sft_hp=SFT_HP,
                       rl_hp=RL_HP, launcher=launcher, agent_loop_yaml=yaml)
    (WORK / "plan.json").write_text(json.dumps({k: v.as_dict() for k, v in invocations.items()}, indent=2))
    for arm, inv in invocations.items():
        (WORK / f"cmd_{arm}.sh").write_text(inv.command + "\n")
        print(f"{arm:11s} expected_checkpoint {inv.expected_checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
