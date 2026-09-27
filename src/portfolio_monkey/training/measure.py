"""The missing half of the selection gate: who computes ``RunRecord.metrics``.

:mod:`portfolio_monkey.training.selection` documents ``metrics`` as "attached
later by whatever computes them" and then rejects any run that reaches a metric
gate without the key.  That is the right split -- ``scan_runs`` has to stay
cheap enough to point at ten thousand directories casually, and it would not be
if it built a nav panel per run -- but until this module existed nothing filled
the mapping, so ``main_sample_spec``'s ``asr``/``mdd_pm`` gates rejected every
run in every campaign.  Structurally the sample looked perfect and the funnel
reported zero kept.

This is a bridge and nothing more: it reads each run's ledger through the same
:func:`~portfolio_monkey.eval.builders.from_ledger.build_ledger_panel` the paper
tables read, scores it with the same :func:`portfolio_monkey.eval.metrics.compute`,
and renames two of its fields to the keys the gate spells.  No metric is defined
here.  If a number in the selection funnel disagrees with the same number in
Table 1, the two were not computed by the same code and this module is the bug.

WHY THE KEY CARRIES THE GRID.  ``mdd_pm`` is a drawdown on the close-only grid.
The AM+PM grid sees intraday marks the PM grid skips, so it can only report a
*deeper* drawdown for the same book; the two are not interchangeable and a gate
written for one must not silently receive the other.  Hence
:func:`metric_keys`: the grid is in the name, so a spec asking for ``mdd_pm``
cannot be satisfied by an AM+PM measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..eval import metrics as _metrics
from ..eval.builders.from_ledger import build_ledger_panel
from .selection import RunRecord

__all__ = ["MEASURE_VERSION", "attach_metrics", "measure_run", "metric_keys"]

#: Bumped when the mapping below changes, so a cached measurement can be told
#: apart from a fresh one.  The metric *definitions* are versioned by the eval
#: stack, not here.
MEASURE_VERSION = "measure.v1"

#: Suffix per grid for grid-dependent keys.  ``AM+PM`` is not a legal identifier
#: fragment, hence the spelling.
_GRID_SUFFIX = {"PM": "pm", "AM+PM": "ampm"}


def metric_keys(grid: str = "PM") -> Mapping[str, str]:
    """``{gate key: ArmMetrics field}`` for one grid.

    Deliberately small.  Every key a :class:`~.selection.SelectionSpec` can gate
    on has to appear here, and every entry has to name a field that
    :func:`portfolio_monkey.eval.metrics.compute` actually returns -- a typo
    would surface as a universally failing gate, which reads exactly like a
    campaign of bad runs.
    """
    suffix = _GRID_SUFFIX.get(grid)
    if suffix is None:
        raise ValueError(f"grid must be one of {sorted(_GRID_SUFFIX)}; got {grid!r}")
    return {
        # The gate's ``asr`` is the annualized Sharpe ratio of section 3, which
        # ``compute`` calls ``sharpe``.  It is grid-invariant in expectation
        # only; the name stays unsuffixed because the spec spells it that way.
        "asr": "sharpe",
        f"mdd_{suffix}": "max_drawdown_daily",
        # Not gated on today, carried because the funnel report is more useful
        # when a rejected run can be inspected without a second pass.
        "log_return": "cumulative_log_return",
        "ann_vol": "annualized_vol",
        f"cost_share_{suffix}": "cost_share",
        "sharpe_se": "sharpe_se",
    }


def measure_run(
    run_dir: Path | str,
    *,
    grid: str = "PM",
    track: str | None = None,
) -> dict[str, float | None]:
    """Metrics for one run directory, or ``{}`` if its ledger will not build.

    ``{}`` rather than a raise, and rather than zeros: a run whose
    ``nav_panel.jsonl`` is empty or holds two ``(arm, track)`` pairs is a run
    that cannot be scored, and the selection funnel already has the right
    behaviour for an unmeasured run -- it rejects it at the first metric gate
    and says so.  Filling in a zero Sharpe would instead put it in the reject
    column labelled as a *bad* run, which is a different claim.
    """
    keys = metric_keys(grid)
    try:
        panel = build_ledger_panel(Path(run_dir), grid=grid, track=track)
        scored = _metrics.compute(panel).as_dict()
    except Exception:  # noqa: BLE001 - see docstring: unmeasurable, not bad
        return {}
    return {key: scored[field] for key, field in keys.items()}


def attach_metrics(
    records: Sequence[RunRecord],
    *,
    grid: str = "PM",
    workers: int = 8,
) -> list[RunRecord]:
    """``records`` with ``metrics`` filled, order preserved.

    Threaded because the work is one small file read per run and the cost is
    ``/ocean`` latency rather than arithmetic.  Order is preserved so a caller
    can zip the result against its own scan.
    """
    metric_keys(grid)  # fail fast on a bad grid, before touching the disk
    if not records:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        measured = list(pool.map(lambda r: measure_run(r.run_dir, grid=grid), records))
    return [r.with_metrics(m) for r, m in zip(records, measured)]
