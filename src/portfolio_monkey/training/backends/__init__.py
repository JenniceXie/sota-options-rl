"""Where a training job runs, expressed without importing the thing that runs it.

THE CONSTRAINT THAT SHAPES THIS PACKAGE.  The machine that writes the training
config has no GPU, no Miles, no veRL, and no CUDA; the machine that runs it has
all four and does not have this repo's data mounted the same way.  So a backend
here never *calls* a framework -- it emits the artifacts a framework consumes:
a dataset file, a config file, and a command line.  That is testable on a
laptop, reviewable in a diff, and reproducible by hand if the launcher breaks.

MILES IS THE RULED FRAMEWORK AND ITS CLI IS NOT ON DISK.  Searched 2026-09-24:
nothing under the project root mentions Miles or RadixArk, there is no checkout,
no wheel, and no image -- only the Docker tag ``radixark/miles:v0.1.0``.  veRL
v0.7.1 *is* on disk, as a 12 GB ``.sif`` with its source tree beside it.  The
response to that asymmetry is not to invent a Miles command line and hard-code
it: it is to make the command line **data**.  A backend is a template plus a
parameter mapping, both loadable from a file, so supplying Miles' real interface
is an edit to a config, not to this package.  The veRL backend is written out in
full because its interface could be read, and it doubles as the worked example
of what a Miles profile has to fill in.

NOTHING HERE DECIDES WHERE IT RUNS.  No queue name, no account, no node count,
no container path, no module load.  Those come in through
:class:`~.base.Launcher` as values, because a repo that knows it lives on PSC is
a repo that cannot be handed to another machine -- which is the whole point of
this package.
"""

from __future__ import annotations

from .base import (
    BackendError,
    Invocation,
    Launcher,
    ProfileBackend,
    TrainerBackend,
    TrainingJob,
    load_profile,
    render,
)
from .verl import VerlBackend

__all__ = [
    "BackendError",
    "Invocation",
    "Launcher",
    "ProfileBackend",
    "TrainerBackend",
    "TrainingJob",
    "VerlBackend",
    "load_profile",
    "render",
]
