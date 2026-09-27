"""Matrix -> jobs, for the three in-scope arms only.

The missing ``scripts/training/`` driver, reduced to this campaign's scope:
``paper_matrix(seeds=(0,))`` is queried (never edited), the arms are intersected
with ``IN_SCOPE`` and any of the seven un-ablatable ids is refused, one
``TrainingJob`` is emitted per stage, and every RL job's ``init_checkpoint`` is
the SFT job's ``Invocation.expected_checkpoint`` -- read, not re-derived.

The backend subclasses the repo's ``VerlBackend`` so its refusals (no
``max_seq_len``, RL without ``init_checkpoint``, ``group_size`` unset, unknown
Hydra roots, non-parquet SFT data) still run.  Two things differ:

* SFT saves an HF copy and ``expected_checkpoint`` is that directory, which is
  what the RL stage and vLLM can actually load (the stock backend returns the
  FSDP checkpoint root);
* RL uses ``pm_agent`` (``verl_loop.PMAgentLoop``) instead of ``tool_agent``,
  because the interaction path cannot deliver this environment's first
  observation or its reward in 0.7.1 (see ``verl_loop``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from portfolio_monkey.training.backends.base import Invocation, Launcher, TrainingJob, _wrap
from portfolio_monkey.training.backends.verl import VerlBackend
from portfolio_monkey.training.spec import paper_matrix

from .envfactory import IN_SCOPE, UNABLATABLE, ScopeError

REWARD_PY = str(Path(__file__).with_name("reward_shim.py"))


@dataclass(frozen=True)
class PMVerlBackend(VerlBackend):
    agent_loop_yaml: str = ""

    def _sft(self, job, launcher):
        argv, files, local_dir = super()._sft(job, launcher)
        # verl.trainer.sft_trainer calls initialize_global_process_group(), which
        # reads RANK/WORLD_SIZE/MASTER_ADDR: it must be launched by torchrun.
        # The stock backend's `python -m ...` cannot start it.
        assert argv[:3] == [launcher.python, "-m", self.sft_module], argv[:3]
        torchrun = str(Path(launcher.python).with_name("torchrun"))
        argv = [torchrun, "--standalone", "--nnodes=1",
                f"--nproc_per_node={launcher.gpus_per_node}", "-m", self.sft_module] + argv[3:]
        argv += [
            "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
            f"data.max_token_len_per_gpu={job.max_seq_len}",
        ]
        return argv, files, f"{job.output_dir}/hf"

    def _rl(self, job, launcher):
        local_dir = f"{job.output_dir}/checkpoints"
        hp = dict(job.hyperparameters)
        argv = [
            launcher.python, "-m", self.rl_module,
            "algorithm.adv_estimator=grpo",
            f"data.train_files={job.data}",
            f"data.val_files={job.data}",
            "data.return_raw_chat=True",
            f"actor_rollout_ref.model.path={job.init_checkpoint}",
            "actor_rollout_ref.model.trust_remote_code=True",
            f"actor_rollout_ref.rollout.name={self.rollout_engine}",
            "actor_rollout_ref.rollout.mode=async",
            f"actor_rollout_ref.rollout.n={job.group_size}",
            "actor_rollout_ref.rollout.agent.default_agent_loop=pm_agent",
            f"actor_rollout_ref.rollout.agent.agent_loop_config_path={self.agent_loop_yaml}",
            # Belt and braces: PMAgentLoop sets rm_scores and every reward manager
            # short-circuits on them.  If they were ever missing, this function
            # raises ("neither run_dir nor nav_marks") instead of the default
            # math scorer returning a plausible 0.
            f"reward.custom_reward_function.path={REWARD_PY}",
            "reward.custom_reward_function.name=compute_score",
            f"trainer.default_local_dir={local_dir}",
            "trainer.project_name=portfolio_monkey",
            f"trainer.experiment_name={job.arm_id}",
            # 0.7.1 has no trainer.seed for PPO; the seed lives in three places.
            f"data.seed={job.seed}",
            f"actor_rollout_ref.actor.data_loader_seed={job.seed}",
            f"actor_rollout_ref.actor.fsdp_config.seed={job.seed}",
            f"trainer.nnodes={launcher.nodes}",
            f"trainer.n_gpus_per_node={launcher.gpus_per_node}",
        ]
        return argv, {}, f"{job.output_dir}/hf"

    def build(self, job: TrainingJob, launcher: Launcher) -> Invocation:
        if job.arm_id in UNABLATABLE:
            raise ScopeError(f"{job.arm_id}: permanently out of scope -- refused")
        if job.arm_id not in IN_SCOPE:
            raise ScopeError(f"{job.arm_id}: not an in-scope arm")
        # Keys starting with "+" are Hydra appends (not in the struct).  The base
        # check() would refuse "+x" as an unknown root, so they are taken out,
        # checked against the same roots without the "+", and appended verbatim.
        hp = dict(job.hyperparameters)
        appends = {k: hp.pop(k) for k in list(hp) if k.startswith("+")}
        from portfolio_monkey.training.backends.verl import _ROOTS, _hydra
        bad = [k for k in appends if k[1:].split(".", 1)[0] not in _ROOTS[job.stage]]
        if bad:
            raise ValueError(f"{job.arm_id}: appended key(s) with unknown root: {bad}")
        extra = tuple(f"{k}={_hydra(v)}" for k, v in sorted(appends.items()))
        job = replace(job, hyperparameters=hp)
        if job.stage != "rl":
            inv = super().build(job, launcher)
            return replace(inv, argv=tuple(inv.argv) + extra)
        # reward_kwargs is not in 0.7.1's struct, so it needs Hydra's "+"; the
        # base check() refuses "+reward" as an unknown root.  So gamma is taken
        # out of the hyperparameters, and appended here.  It is still required:
        # compute_score refuses a default, and so does this.
        hp = dict(job.hyperparameters)
        gamma = hp.pop(GAMMA_KEY, None)
        if gamma is None:
            raise ValueError(f"{job.arm_id}: RL job without a gamma ({GAMMA_KEY})")
        inv = super().build(replace(job, hyperparameters=hp), launcher)
        return replace(inv, argv=tuple(inv.argv) + (f"+{GAMMA_KEY}={gamma}",) + extra)


GAMMA_KEY = "reward.custom_reward_function.reward_kwargs.gamma"


def in_scope_arms():
    """The three arms, as queries against the matrix at one seed."""
    m = paper_matrix(seeds=(0,))
    arms = {a.arm_id: a for a in m.arms}
    for arm_id in IN_SCOPE:
        a = arms[arm_id]
        assert tuple(a.seeds) == (0,), (arm_id, a.seeds)
        assert a.policy.kind == "llm" and not a.reuses
    return {k: arms[k] for k in IN_SCOPE}


def plan(*, work: str | Path, model: str, sft_data: str, rl_data: Mapping[str, str],
         max_seq_len: int, group_size: int, sft_hp: Mapping[str, Any],
         rl_hp: Mapping[str, Any], launcher: Launcher, agent_loop_yaml: Mapping[str, str],
         seed: int = 0) -> dict[str, Invocation]:
    arms = in_scope_arms()
    work = Path(work)
    sft_job = TrainingJob(stage="sft", arm_id="sys_sft", model=model, data=sft_data,
                          output_dir=str(work / "sys_sft"), max_seq_len=max_seq_len,
                          hyperparameters=dict(sft_hp), seed=seed)
    sft = PMVerlBackend().build(sft_job, launcher)
    out = {"sys_sft": sft}
    for arm_id in ("sys_sft_rl", "s4_none"):
        assert arms[arm_id].training == "sft+rl"
        job = TrainingJob(stage="rl", arm_id=arm_id, model=model, data=rl_data[arm_id],
                          output_dir=str(work / arm_id), max_seq_len=max_seq_len,
                          hyperparameters=dict(rl_hp), group_size=group_size, seed=seed,
                          init_checkpoint=sft.expected_checkpoint)
        out[arm_id] = PMVerlBackend(agent_loop_yaml=agent_loop_yaml[arm_id]).build(job, launcher)
    return out


def write_rl_dataset(path: str | Path, *, arm: str, dates_file: str, repeat: int = 1,
                     months: list[str] | None = None) -> list[str]:
    """One row per month of the RL window: which arm, which episode.  The
    ``prompt`` is a placeholder -- PMAgentLoop builds the real first turn from
    the environment, which is the only place it exists."""
    import pandas as pd

    from .envfactory import arm_config, episodes_for, read_dates

    config = arm_config(arm)
    episodes = [e.episode_id for e in episodes_for(config, read_dates(dates_file))]
    if months:
        episodes = [e for e in episodes if e in months]
    rows = []
    for r in range(repeat):
        for e in episodes:
            rows.append({
                "data_source": "portfolio_monkey",
                "agent_name": "pm_agent",
                "prompt": [{"role": "user", "content": f"{arm} {e}"}],
                "reward_model": {"style": "env", "ground_truth": ""},
                "extra_info": {"index": len(rows), "arm": arm, "episode_id": e,
                               "dates_file": str(dates_file)},
            })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return episodes


def write_agent_loop_yaml(path: str | Path, **params) -> str:
    lines = ["- name: pm_agent", "  _target_: pm_train.verl_loop.PMAgentLoop"]
    for k, v in params.items():
        lines.append(f"  {k}: {json.dumps(v)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)
