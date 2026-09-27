"""The backend interface: a training job in, files and a command line out.

WHY A JOB IS A VALUE AND NOT A FUNCTION CALL.  ``TrainingJob`` holds everything
that distinguishes one training run from another -- stage, model, data,
hyperparameters, sequence length, the arm it serves -- and nothing about the
machine.  ``Launcher`` holds everything about the machine -- paths, container,
partition, account, node and GPU counts -- and nothing about the experiment.
A backend is the function that crosses them.  Keeping the two apart is what
makes "run this on another cluster" a matter of supplying a different
``Launcher``, rather than of grepping the repo for hard-coded paths.

THE TEMPLATE IS DATA BECAUSE ONE OF THE TWO FRAMEWORKS IS NOT HERE.  Miles is
the ruled framework and its command line could not be read anywhere on disk on
2026-09-24.  Inventing one and embedding it in Python would produce code that
looks finished and fails on contact.  Instead a backend may be defined entirely
by a *profile*: a JSON document naming the executable, the argument template,
and the mapping from :class:`TrainingJob` fields onto that framework's
parameter names.  Supplying Miles is then a file, written by whoever has the
docs, and it is validated the moment it is loaded -- an unresolved placeholder
is an error, not a literal ``{model}`` in a shell command.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BackendError",
    "Invocation",
    "Launcher",
    "ProfileBackend",
    "TrainerBackend",
    "TrainingJob",
    "load_profile",
    "render",
]


class BackendError(ValueError):
    """A training job cannot be turned into something a framework would accept."""


_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")


def render(template: str, values: Mapping[str, Any]) -> str:
    """``template`` with ``{name}`` replaced, refusing any name it cannot fill.

    ``str.format`` is not used, and the difference is the entire reason this
    function exists: ``format`` raises on a missing key but happily passes
    through a stray brace, and a shell command containing a literal ``{model}``
    does not fail -- it runs, with a nonsense argument, and the failure surfaces
    forty minutes later as a confusing framework error. Here an unfillable
    placeholder is refused at render time, by name.
    """
    missing = sorted({m.group(1) for m in _PLACEHOLDER.finditer(template)} - set(values))
    if missing:
        raise BackendError(
            f"template placeholder(s) {', '.join(missing)} have no value. "
            "Rendering anyway would put the literal text into a command line, "
            "which runs and fails elsewhere."
        )
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


@dataclass(frozen=True)
class Launcher:
    """Everything about the machine, and nothing about the experiment.

    Every field is empty or zero by default and every one of them is supplied by
    the caller.  There is no PSC partition here, no ``/ocean`` path and no
    module name, because the repo must not know which cluster it is on: the
    hand-off asks another agent to run this somewhere else, and a default that
    happens to work here is a default that silently misconfigures there.
    """

    #: How a command is wrapped to reach the framework: ``""`` (run directly),
    #: ``"singularity"``, ``"apptainer"``, ``"docker"``, ``"srun"``...
    runtime: str = ""
    #: Container image or ``.sif`` path, when ``runtime`` needs one.
    image: str = ""
    #: Interpreter or entry command inside that runtime.
    python: str = "python"
    #: Bind mounts / volume maps, as the runtime spells them.
    binds: tuple[str, ...] = ()
    #: Environment variables set for the job.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Scheduler parameters, kept as an opaque mapping: Slurm, PBS and a bare
    #: shell disagree about every one of them, and a typed field per Slurm flag
    #: would bake Slurm into the interface.
    scheduler: Mapping[str, Any] = field(default_factory=dict)
    nodes: int = 1
    gpus_per_node: int = 0
    #: Where run outputs go on that machine.
    work_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["env"] = dict(self.env)
        out["scheduler"] = dict(self.scheduler)
        out["binds"] = list(self.binds)
        return out


@dataclass(frozen=True)
class TrainingJob:
    """Everything about the experiment, and nothing about the machine."""

    #: ``"sft"`` or ``"rl"``.  The two consume different data and different
    #: hyperparameters, and a backend is allowed to refuse a stage it cannot run.
    stage: str
    #: The arm from :mod:`..spec` this job produces a checkpoint for.
    arm_id: str
    model: str
    #: SFT: the chat-jsonl written by :func:`..corpus.write_chat_jsonl`.
    #: RL: the file of episode specifications the rollout agent draws from.
    data: str
    output_dir: str
    #: Must hold the longest example in ``data``.  Not defaulted: the measured
    #: ceiling for Qwen3 on this state space is 40,960, the prompt budget the
    #: environment enforces is 32,768, and the two are different numbers for
    #: different reasons. Guessing either one truncates training examples
    #: silently.
    max_seq_len: int = 0
    #: Free-form and framework-specific: learning rate, LoRA rank, group size,
    #: KL coefficient.  Untyped because the two frameworks do not agree on names
    #: and a typed union would have to be edited to add a hyperparameter.
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    #: RL only: how many rollouts per prompt.  ``G=8`` is the ruled group size.
    group_size: int = 0
    seed: int = 0
    #: Checkpoint this job starts from.  RL starts from the SFT checkpoint, so
    #: leaving it empty on an RL job is almost always a mistake and the backends
    #: say so rather than starting from the base weights.
    init_checkpoint: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hyperparameters"] = dict(self.hyperparameters)
        return out

    def template_values(self) -> dict[str, Any]:
        """Flattened for :func:`render`, with hyperparameters under ``hp.*``."""
        values: dict[str, Any] = {k: v for k, v in self.as_dict().items()
                                  if k != "hyperparameters"}
        for key, value in self.hyperparameters.items():
            values[f"hp.{key}"] = value
        return values


@dataclass(frozen=True)
class Invocation:
    """What to run, what to write first, and what it will produce.

    ``files`` is part of the invocation rather than a side effect of building
    it, because a backend that writes to disk while you are still deciding
    whether to launch is a backend you cannot dry-run.  Nothing is written until
    :meth:`materialize` is called.
    """

    argv: tuple[str, ...]
    #: path -> contents, written relative to nothing: the paths are absolute or
    #: they are the caller's problem.
    files: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    #: Where the backend expects the checkpoint to land, so the next stage can
    #: be wired to it without re-deriving the framework's layout convention.
    expected_checkpoint: str = ""
    description: str = ""

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def materialize(self, root: Path | str | None = None) -> list[Path]:
        written: list[Path] = []
        for name, contents in self.files.items():
            path = Path(root) / name if root is not None else Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            written.append(path)
        return written

    def as_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "command": self.command,
            "env": dict(self.env),
            "files": sorted(self.files),
            "expected_checkpoint": self.expected_checkpoint,
            "description": self.description,
        }


class TrainerBackend:
    """One framework.

    Subclasses implement :meth:`build`.  They may also declare which stages they
    support: a backend asked for a stage it cannot run must say so, rather than
    emit a command that fails after the allocation is granted.
    """

    name: str = "backend"
    stages: tuple[str, ...] = ("sft", "rl")

    def check(self, job: TrainingJob) -> None:
        """Refusals that are the same for every framework.

        Three of them, and each is a failure mode that costs an allocation
        rather than a traceback:

        *   an unsupported stage;
        *   ``max_seq_len`` unset, which every framework will happily default
            and then truncate against, silently dropping the tail of a long
            episode -- and the long episodes are the ones with a full book;
        *   an RL job with no ``init_checkpoint``, which starts GRPO from the
            base weights.  That trains, converges to something, and is not the
            experiment: the paper's RL arm is *SFT followed by* RL.
        """
        if job.stage not in self.stages:
            raise BackendError(
                f"{self.name}: stage {job.stage!r} not supported (have {self.stages})"
            )
        if job.max_seq_len <= 0:
            raise BackendError(
                f"{self.name}: {job.arm_id} has no max_seq_len. Every framework "
                "defaults this and then truncates against the default, which drops "
                "the tail of exactly the episodes that carry a full book."
            )
        if job.stage == "rl" and not job.init_checkpoint:
            raise BackendError(
                f"{self.name}: {job.arm_id} is an RL job with no init_checkpoint, so "
                "it would start from the base weights. The arm is SFT *followed by* "
                "RL; starting cold trains a different system under its name."
            )

    def build(self, job: TrainingJob, launcher: Launcher) -> Invocation:
        raise NotImplementedError


# --------------------------------------------------------------------------
# profile-driven backends
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ProfileBackend(TrainerBackend):
    """A framework described by a file rather than by Python.

    Keyword-only, and not by taste: :class:`TrainerBackend` carries ``name`` and
    ``stages`` as plain class attributes, which ``dataclass`` reads as defaults
    through the MRO, so a positional signature would put the *required* ``argv``
    after two defaulted fields and fail to build at import time.

    This is how Miles gets supported without its documentation.  A profile names
    the executable and an argument template per stage, plus the config file the
    framework wants written, and every ``{placeholder}`` in it is filled from
    :meth:`TrainingJob.template_values` and the launcher.  Anything unfillable
    is refused at build time.
    """

    name: str
    stages: tuple[str, ...]
    #: stage -> argv template, each element rendered separately so that an
    #: argument containing a space stays one argument.
    argv: Mapping[str, tuple[str, ...]]
    #: stage -> {relative path template: contents template}
    files: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: Where this framework writes its final checkpoint, as a template.
    checkpoint: str = "{output_dir}"
    description: str = ""

    def build(self, job: TrainingJob, launcher: Launcher) -> Invocation:
        self.check(job)
        values = job.template_values()
        values.update({f"launcher.{k}": v for k, v in launcher.as_dict().items()})
        values["launcher.env"] = json.dumps(dict(launcher.env), sort_keys=True)
        values["launcher.binds"] = ",".join(launcher.binds)
        values["launcher.scheduler"] = json.dumps(dict(launcher.scheduler), sort_keys=True)

        argv = [render(part, values) for part in self.argv[job.stage]]
        files = {
            render(path, values): render(body, values)
            for path, body in self.files.get(job.stage, {}).items()
        }
        return Invocation(
            argv=tuple(_wrap(argv, launcher)),
            files=files,
            env=dict(launcher.env),
            expected_checkpoint=render(self.checkpoint, values),
            description=self.description or f"{self.name} {job.stage} for {job.arm_id}",
        )


def _wrap(argv: Sequence[str], launcher: Launcher) -> list[str]:
    """Put ``argv`` inside whatever runtime the machine uses to reach the GPUs.

    Deliberately small and deliberately not exhaustive.  Three runtimes are
    spelled out because they are the three in play; anything else is refused
    rather than guessed, since a wrong container invocation fails in a way that
    looks like a framework bug.
    """
    if not launcher.runtime:
        return list(argv)
    if launcher.runtime in ("singularity", "apptainer"):
        if not launcher.image:
            raise BackendError(f"runtime {launcher.runtime!r} needs an image")
        binds: list[str] = []
        for bind in launcher.binds:
            binds += ["--bind", bind]
        nv = ["--nv"] if launcher.gpus_per_node else []
        return [launcher.runtime, "exec", *nv, *binds, launcher.image, *argv]
    if launcher.runtime == "docker":
        if not launcher.image:
            raise BackendError("runtime 'docker' needs an image")
        binds = []
        for bind in launcher.binds:
            binds += ["-v", bind]
        gpus = ["--gpus", "all"] if launcher.gpus_per_node else []
        return ["docker", "run", "--rm", *gpus, *binds, launcher.image, *argv]
    raise BackendError(
        f"unknown runtime {launcher.runtime!r}. Guessing a container invocation "
        "produces a failure that reads like a framework bug."
    )


def load_profile(path: Path | str) -> ProfileBackend:
    """A backend from JSON.

    Validated on load rather than on first use: a profile with a stage missing
    from ``argv`` is discovered when somebody supplies it, not forty minutes
    into an allocation.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        name = str(payload["name"])
        stages = tuple(str(s) for s in payload["stages"])
        argv = {str(k): tuple(str(x) for x in v) for k, v in payload["argv"].items()}
    except KeyError as exc:
        raise BackendError(f"{path}: profile is missing {exc.args[0]!r}") from exc
    missing = sorted(set(stages) - set(argv))
    if missing:
        raise BackendError(
            f"{path}: declares stage(s) {', '.join(missing)} with no argv template"
        )
    return ProfileBackend(
        name=name,
        stages=stages,
        argv=argv,
        files={
            str(stage): {str(p): str(b) for p, b in spec.items()}
            for stage, spec in payload.get("files", {}).items()
        },
        checkpoint=str(payload.get("checkpoint", "{output_dir}")),
        description=str(payload.get("description", "")),
    )
