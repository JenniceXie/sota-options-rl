"""veRL v0.7.1, written from the source tree on disk rather than from memory.

WHY THIS BACKEND EXISTS AT ALL WHEN MILES IS THE RULED FRAMEWORK.  Miles' CLI
could not be read anywhere on this machine on 2026-09-24; veRL v0.7.1 could, as
a 12 GB ``.sif`` with its source beside it.  So veRL is the backend that can be
written *truthfully*, and it earns its place twice: it is the only thing here
that can smoke-test the pipeline on one shared GPU before an H100 node is
requested, and it is the worked example of what a Miles profile must fill in.

EVERY KEY BELOW WAS READ OUT OF THE INSTALLED SOURCE.  Paths are relative to
``envs/verl-v0.7.1/src/verl/``, and the facts that shaped this file are:

*   SFT entry point is ``verl.trainer.sft_trainer`` with Hydra
    ``config_name="sft_trainer_engine"`` (``verl/trainer/sft_trainer.py:451``).
    It is **not** ``fsdp_sft_trainer``, which is what 0.4-era examples use.
*   The SFT dataset defaults to ``MultiTurnSFTDataset`` when
    ``data.custom_cls.path`` is unset (``sft_trainer.py:461-469``), and that
    class reads **parquet only**, through
    ``pd.read_parquet(parquet_file, dtype_backend="pyarrow")``
    (``verl/utils/dataset/multiturn_sft_dataset.py:146``).  A chat ``.jsonl``
    handed to it fails *after* Ray has started.  Hence :attr:`data_format` and
    the refusal in :meth:`VerlBackend.check`.
*   Chat data is keyed by ``data.messages_key`` (default ``messages``) and
    ``data.tools_key`` (``config/sft_trainer_engine.yaml:26-27``), so the
    OpenAI-shaped corpus needs no reshaping beyond the file format.
*   RL entry point is ``verl.trainer.main_ppo``, ``config_name="ppo_trainer"``
    (``verl/trainer/main_ppo.py:35``).  GRPO is selected by
    ``algorithm.adv_estimator=grpo``; the default is ``gae``
    (``config/_generated_ppo_trainer.yaml:646``).
*   ``actor_rollout_ref.rollout.name`` is ``???`` -- mandatory, no default
    (``_generated_ppo_trainer.yaml:227``).  Omitting it is a Hydra error before
    anything loads, which is the good kind of failure.
*   The multi-turn *interaction* path -- the one that lets this repo's
    environment answer the model turn by turn -- is consumed only by
    ``verl/experimental/agent_loop/tool_agent_loop.py:117``, registered as
    ``tool_agent`` (``tool_agent_loop.py:95``).  So it needs
    ``rollout.mode=async`` plus ``rollout.agent.default_agent_loop=tool_agent``
    plus ``rollout.multi_turn.enable=true`` plus
    ``rollout.multi_turn.interaction_config_path``.  No sglang gate appears in
    that file or in ``agent_loop.py`` in 0.7.1; earlier versions did restrict
    the tool path to sglang, so this is recorded as *checked*, not assumed.
*   Interactions are constructed from a yaml of
    ``interaction: [{name, class_name, config}]``
    (``verl/interactions/utils/interaction_registry.py:42-82``), and each
    prompt row selects one by ``extra_info.interaction_kwargs.name``, which is
    **required** (``tool_agent_loop.py:138-143``;
    ``verl/utils/dataset/rl_dataset.py:362``).
*   The custom reward is ``reward.custom_reward_function.{path,name}`` with
    optional ``reward_kwargs`` (``verl/trainer/ppo/reward.py:70-86``), and it is
    called as ``fn(data_source=, solution_str=, ground_truth=, extra_info=)``
    (``verl/workers/reward_manager/naive.py:86-90``).

NOTHING HERE NAMES A MACHINE.  Partition, account, container path, node count
and bind mounts all arrive through :class:`~.base.Launcher`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .base import (
    BackendError,
    Invocation,
    Launcher,
    TrainerBackend,
    TrainingJob,
    _wrap,
)

__all__ = ["VerlBackend", "SINGLE_GPU_SMOKE", "interaction_config"]


#: Top-level Hydra groups that exist in v0.7.1, per stage.  Overrides are
#: checked against these before launch: Hydra rejects an unknown root too, but
#: it does so after Ray has been initialised, and on a shared queue that is a
#: twenty-minute round trip to learn that ``trainer`` was typed ``trianer``.
_ROOTS: Mapping[str, frozenset[str]] = {
    "sft": frozenset({
        "data", "model", "engine", "optim", "checkpoint", "trainer",
        "global_profiler", "profiler",
    }),
    "rl": frozenset({
        "data", "actor_rollout_ref", "critic", "reward_model", "reward",
        "custom_reward_function", "algorithm", "trainer", "ray_init",
        "global_profiler",
    }),
}


#: A configuration small enough for one GPU, and the reason each number is
#: what it is.  These satisfy ``ActorConfig.validate``
#: (``verl/workers/config/actor.py:212-232``), which refuses
#: ``train_batch_size < ppo_mini_batch_size`` and requires
#: ``ppo_micro_batch_size`` to divide the mini batch and to be at least
#: ``n_gpus`` after sequence parallelism.  The defaults (256 / 1024) violate the
#: first of those the moment the batch is shrunk, which is the failure everybody
#: hits when they try to run the tutorial on one card.
SINGLE_GPU_SMOKE: Mapping[str, Mapping[str, Any]] = {
    "sft": {
        "data.train_batch_size": 2,
        "data.micro_batch_size_per_gpu": 1,
        "data.use_dynamic_bsz": True,
        "data.max_token_len_per_gpu": 8192,
        "engine.strategy": "fsdp2",
        "model.enable_gradient_checkpointing": True,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 2,
        "trainer.save_freq": 2,
        "trainer.logger": "[console]",
    },
    "rl": {
        "data.train_batch_size": 2,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        # 0.5 of an 80 GB card leaves room for the FSDP shard and the
        # optimizer state on the same device; the default is also 0.5 but it
        # assumes the actor lives elsewhere.
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 2,
        "trainer.save_freq": 2,
        "trainer.logger": "[console]",
    },
}


def interaction_config(class_name: str, name: str, config: Mapping[str, Any] | None = None) -> str:
    """The yaml ``initialize_interactions_from_config`` expects, as text.

    One list entry, because one environment answers every prompt: the *episode*
    is chosen per row through ``extra_info.interaction_kwargs``, not by having
    an interaction class per episode.  ``name`` is written explicitly rather
    than left to the registry's derive-from-class-name rule
    (``interaction_registry.py:62-70``), since that rule strips an
    ``Interaction`` suffix and lowercases, and a silently different name is a
    ``KeyError`` inside the rollout worker.
    """
    lines = ["interaction:", f"  - name: {name}", f"    class_name: {class_name}", "    config:"]
    for key, value in sorted((config or {}).items()):
        lines.append(f"      {key}: {_yaml_scalar(value)}")
    if not config:
        lines[-1] = "    config: {}"
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    return f'"{value}"'


@dataclass(frozen=True)
class VerlBackend(TrainerBackend):
    """veRL as a command line.

    The hyperparameters mapping is passed through as dotted Hydra overrides
    *verbatim*: there is no translation table from friendly names onto veRL
    names, because such a table is a second place to be wrong and it goes stale
    the moment veRL renames a key.  What is checked is the root -- see
    :data:`_ROOTS`.
    """

    name: str = "verl"
    stages: tuple[str, ...] = ("sft", "rl")
    #: What ``MultiTurnSFTDataset`` can actually open.
    data_format: str = "parquet"
    #: Overridable so a fork or a newer tag can be pointed at without editing
    #: this file.
    sft_module: str = "verl.trainer.sft_trainer"
    rl_module: str = "verl.trainer.main_ppo"
    #: Rollout engine. ``actor_rollout_ref.rollout.name`` has no default in
    #: 0.7.1, so something must be supplied and it may as well be visible here.
    rollout_engine: str = "vllm"
    #: Where the environment-as-interaction class lives, for the RL stage.
    interaction_class: str = "portfolio_monkey.training.agent.PortfolioMonkeyInteraction"
    interaction_name: str = "portfolio_monkey"
    #: Module and function veRL loads for the trajectory reward.
    reward_path: str = "portfolio_monkey/training/reward.py"
    reward_name: str = "compute_score"

    def check(self, job: TrainingJob) -> None:
        super().check(job)
        if job.stage == "sft" and not job.data.endswith(".parquet"):
            raise BackendError(
                f"verl: SFT data must be parquet, got {job.data!r}. "
                "MultiTurnSFTDataset reads it with pandas.read_parquet "
                "(multiturn_sft_dataset.py:146); a .jsonl fails after Ray has "
                "already started."
            )
        if job.stage == "rl" and job.group_size <= 0:
            raise BackendError(
                "verl: an RL job needs group_size (actor_rollout_ref.rollout.n). "
                "Left at the default of 1 there is no group, and GRPO's "
                "advantage is the deviation from a group of one -- identically "
                "zero, so nothing trains and nothing errors."
            )
        bad = sorted({
            key for key in job.hyperparameters
            if key.split(".", 1)[0] not in _ROOTS[job.stage]
        })
        if bad:
            raise BackendError(
                f"verl: override(s) {', '.join(bad)} name no top-level config "
                f"group for stage {job.stage!r} (have {sorted(_ROOTS[job.stage])}). "
                "Hydra would also refuse these, but only after the job is "
                "scheduled."
            )

    def build(self, job: TrainingJob, launcher: Launcher) -> Invocation:
        self.check(job)
        if job.stage == "sft":
            argv, files, checkpoint = self._sft(job, launcher)
        else:
            argv, files, checkpoint = self._rl(job, launcher)
        # The caller's overrides go last: Hydra takes the final occurrence, so
        # a hyperparameter may deliberately contradict anything derived above.
        argv += [f"{k}={_hydra(v)}" for k, v in sorted(job.hyperparameters.items())]
        return Invocation(
            argv=tuple(_wrap(argv, launcher)),
            files=files,
            env=dict(launcher.env),
            expected_checkpoint=checkpoint,
            description=f"verl {job.stage} for {job.arm_id}",
        )

    # -- stages -----------------------------------------------------------

    def _sft(self, job, launcher):
        local_dir = f"{job.output_dir}/checkpoints"
        argv = [
            launcher.python, "-m", self.sft_module,
            f"data.train_files={job.data}",
            "data.messages_key=messages",
            f"data.max_length={job.max_seq_len}",
            # `error`, not `right`: a truncated training example teaches the
            # model to stop mid-DSL, and the long examples are the ones with a
            # full book. Better to fail and re-measure the corpus.
            "data.truncation=error",
            f"model.path={job.model}",
            "model.trust_remote_code=True",
            f"trainer.default_local_dir={local_dir}",
            "trainer.project_name=portfolio_monkey",
            f"trainer.experiment_name={job.arm_id}",
            f"trainer.seed={job.seed}",
            f"trainer.nnodes={launcher.nodes}",
            f"trainer.n_gpus_per_node={launcher.gpus_per_node}",
        ]
        return argv, {}, local_dir

    def _rl(self, job, launcher):
        local_dir = f"{job.output_dir}/checkpoints"
        interaction_path = f"{job.output_dir}/interaction.yaml"
        argv = [
            launcher.python, "-m", self.rl_module,
            "algorithm.adv_estimator=grpo",
            f"data.train_files={job.data}",
            f"data.max_prompt_length={job.max_seq_len}",
            # The response is generated *inside* the prompt budget the
            # environment enforces, so it is not a second free allowance.
            f"data.max_response_length={job.hyperparameters.get('data.max_response_length', 2048)}",
            "data.return_raw_chat=True",
            # The SFT checkpoint enters as the actor's weights. veRL has no
            # "start RL from that SFT run" switch -- `trainer.resume_from_path`
            # resumes *this* job's own checkpoints -- so the wiring is a model
            # path, and `check()` refuses an empty one.
            f"actor_rollout_ref.model.path={job.init_checkpoint}",
            "actor_rollout_ref.model.trust_remote_code=True",
            f"actor_rollout_ref.rollout.name={self.rollout_engine}",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
            f"actor_rollout_ref.rollout.n={job.group_size}",
            "actor_rollout_ref.rollout.multi_turn.enable=True",
            f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_path}",
            f"reward.custom_reward_function.path={self.reward_path}",
            f"reward.custom_reward_function.name={self.reward_name}",
            f"trainer.default_local_dir={local_dir}",
            "trainer.project_name=portfolio_monkey",
            f"trainer.experiment_name={job.arm_id}",
            f"trainer.seed={job.seed}",
            f"trainer.nnodes={launcher.nodes}",
            f"trainer.n_gpus_per_node={launcher.gpus_per_node}",
        ]
        files = {
            interaction_path: interaction_config(
                self.interaction_class, self.interaction_name,
                {"arm_id": job.arm_id},
            )
        }
        return argv, files, local_dir


def _hydra(value: Any) -> str:
    """A Python value as Hydra spells it on a command line."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra(v) for v in value) + "]"
    return str(value)
