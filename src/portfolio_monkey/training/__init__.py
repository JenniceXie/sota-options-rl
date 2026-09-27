"""Turning recorded episodes into training runs, and training runs into tables.

Four modules, in the order a campaign uses them:

``selection``  Which recorded runs enter a sample.  A *filter object*, not a
               frozen list of run names, because the sample is expected to grow:
               the 126 that passed the V+ASR+MDD gate came out of the first
               thousand draws and a second thousand is already on the queue.
``corpus``     One recorded run -> the conversation that produced it, in a
               chat format any trainer can read.  This is the module that needs
               ``EnvConfig.from_dict``: the system block is not stored in a run
               directory, so without it a trajectory is answers with the
               rulebook missing.
``spec``       The experiment matrix as data.  Every arm in the paper's Section 5
               -- baselines, ablations, training stages -- is a row here, and
               nothing about an arm is expressed as a branch in Python.
``backends``   Where a training job actually runs.  ``TrainerBackend`` is an
               interface; Miles and veRL are two implementations of it.  The
               repo does not import either one: it *emits* their invocations,
               so this package stays runnable on a machine with no GPU and no
               framework installed, which is the machine it was written on.

The split exists because these four have very different lifetimes.  The matrix
changes whenever the paper changes; the corpus format changes whenever the state
space changes; the backend changes when the cluster changes; and the selection
filter changes every time a new batch of draws lands.  Folding any two together
would make the most volatile one set the release cadence for the rest.
"""

from __future__ import annotations

__all__ = ["corpus", "selection", "spec"]
