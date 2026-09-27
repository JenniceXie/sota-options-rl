"""Portfolio metrics computed from a NAV panel and nothing else.

Pure: no I/O, no configuration, no knowledge of features, prompts or policies.
Everything here is a function of ``schema.NavPanel``, so a change to the state
design cannot move a reported number (``docs/evaluation_protocol.md`` 0.1).

Conventions follow the protocol document:

* The criterion is the **cumulative log return**: the undiscounted sum of log
  NAV changes, which telescopes to ``log(V_T / V_0)`` and is therefore
  grid-invariant (section 1.1).
* Second-moment metrics use the **daily PM** grid. The AM+PM grid has a
  deterministic intraday/overnight 2-cycle in variance, and partitioning each
  daily return into two pieces adds no independent information about daily
  variance (section 3.2).
* Drawdown uses the **finest available** grid, because drawdown is a maximum
  over a path and coarse sampling is biased downward (section 3.1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from portfolio_monkey.eval.schema import NavPanel, NavRow

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class ArmMetrics:
    """One row of Table 1. ``None`` means not computable for this arm."""

    arm: str
    track: str
    n_obs: int
    grid: str
    universe: str

    cumulative_log_return: float
    cumulative_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float | None
    sharpe_se: float | None
    sortino: float | None
    max_drawdown_daily: float
    max_drawdown_native: float | None
    calmar: float | None
    skewness: float | None
    excess_kurtosis: float | None
    positive_day_share: float
    turnover: float | None
    frictions: float | None
    gross_pnl: float | None
    cost_share: float | None
    mean_stale_mark_share: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------
# series extraction
# --------------------------------------------------------------------------


def log_levels(rows: Sequence[NavRow]) -> list[float]:
    return [math.log(r.nlv) for r in rows]


def log_returns(rows: Sequence[NavRow]) -> list[float]:
    levels = log_levels(rows)
    return [b - a for a, b in zip(levels, levels[1:])]


def daily_log_returns(panel: NavPanel) -> list[float]:
    """Close-to-close log returns on the PM grid."""

    return log_returns(panel.pm_rows())


# --------------------------------------------------------------------------
# criterion and moments
# --------------------------------------------------------------------------


def cumulative_log_return(panel: NavPanel) -> float:
    """``sum_t r_t = log(V_T / V_0)``, the evaluation criterion.

    Computed from the endpoints rather than by summation, which is the identity
    the telescoping property guarantees; :func:`selftest_telescoping` asserts
    that the summed form agrees.
    """

    rows = panel.rows
    return math.log(rows[-1].nlv) - math.log(rows[0].nlv)


def annualized_log_return(cumulative: float, n_obs: int) -> float:
    return cumulative * TRADING_DAYS_PER_YEAR / n_obs


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def stdev(values: Sequence[float], ddof: int = 1) -> float:
    if len(values) - ddof <= 0:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - ddof))


def annualized_vol(daily: Sequence[float]) -> float:
    return stdev(daily) * math.sqrt(TRADING_DAYS_PER_YEAR)


def skewness(values: Sequence[float]) -> float | None:
    n = len(values)
    if n < 3:
        return None
    mu = mean(values)
    sd = stdev(values, ddof=0)
    if sd == 0.0:
        return None
    return sum((v - mu) ** 3 for v in values) / (n * sd**3)


def excess_kurtosis(values: Sequence[float]) -> float | None:
    n = len(values)
    if n < 4:
        return None
    mu = mean(values)
    sd = stdev(values, ddof=0)
    if sd == 0.0:
        return None
    return sum((v - mu) ** 4 for v in values) / (n * sd**4) - 3.0


# --------------------------------------------------------------------------
# risk-adjusted
# --------------------------------------------------------------------------


def sharpe(daily: Sequence[float], rf_daily: float = 0.0) -> float | None:
    """Annualized Sharpe on daily log excess returns."""

    excess = [d - rf_daily for d in daily]
    sd = stdev(excess)
    if sd == 0.0:
        return None
    return mean(excess) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_standard_error(sharpe_ann: float | None, n_obs: int) -> float | None:
    """Lo (2002) iid standard error of the annualized Sharpe ratio.

    ``SE = sqrt(252 / N) * sqrt(1 + SR_daily^2 / 2)``. At ``N = 126`` this is
    approximately 1.41 before the second factor, which is why every Sharpe in
    Table 1 must be printed with its standard error (protocol section 9.1).
    """

    if sharpe_ann is None or n_obs <= 1:
        return None
    sr_daily = sharpe_ann / math.sqrt(TRADING_DAYS_PER_YEAR)
    return math.sqrt(TRADING_DAYS_PER_YEAR / n_obs) * math.sqrt(1.0 + 0.5 * sr_daily**2)


def sortino(daily: Sequence[float], rf_daily: float = 0.0) -> float | None:
    excess = [d - rf_daily for d in daily]
    downside = [e for e in excess if e < 0.0]
    if not downside:
        return None
    semidev = math.sqrt(sum(e**2 for e in downside) / len(excess))
    if semidev == 0.0:
        return None
    return mean(excess) / semidev * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown_log(levels: Sequence[float]) -> float:
    """Maximum peak-to-trough decline in log space (non-negative)."""

    peak = levels[0]
    worst = 0.0
    for level in levels:
        peak = max(peak, level)
        worst = max(worst, peak - level)
    return worst


def max_drawdown_simple(levels: Sequence[float]) -> float:
    """Maximum drawdown as a positive fraction, ``1 - exp(-mdd_log)``."""

    return 1.0 - math.exp(-max_drawdown_log(levels))


# --------------------------------------------------------------------------
# ledger-only quantities
# --------------------------------------------------------------------------


def _sum_optional(rows: Sequence[NavRow], attribute: str) -> float | None:
    values = [getattr(r, attribute) for r in rows]
    if any(v is None for v in values):
        return None
    return sum(values)


def turnover(panel: NavPanel) -> float | None:
    traded = _sum_optional(panel.rows, "notional_traded")
    if traded is None:
        return None
    average_nlv = mean([r.nlv for r in panel.rows])
    if average_nlv == 0.0:
        return None
    n_days = len(panel.pm_rows())
    annualization = TRADING_DAYS_PER_YEAR / n_days if n_days else 1.0
    return traded / average_nlv * annualization


def frictions(panel: NavPanel) -> float | None:
    """Total half-spread plus fees, in dollars.

    Reported as a level rather than only inside a ratio.  A ratio hides which
    of its two terms moved, and on a short window it is the numerator that is
    stable and the denominator that is noise.
    """
    spread = _sum_optional(panel.rows, "cost_half_spread")
    fees = _sum_optional(panel.rows, "cost_fees")
    if spread is None or fees is None:
        return None
    return spread + fees


def gross_pnl(panel: NavPanel) -> float | None:
    """Absolute trading PnL per step, before frictions and before interest.

    Three corrections against the obvious version, each of which was wrong in
    the shipped metric:

    * ``realized_pnl`` is a catch-all — it already carries the risk-free accrual
      (``execution.py`` ``accrue``) and it is already *net* of frictions
      (``execution.py`` ``open``/``close``).  Both are added back, or the
      denominator would be a function of the numerator.
    * Interest is not trading PnL.  A book that does nothing still accrues it,
      so leaving it in makes an idle arm look like it traded profitably.
    * The sum is over ``|per-step|``, not ``|sum over steps|``.  Netting lets a
      strategy that made and lost the same amount report a denominator of zero
      and a cost share of infinity — which is exactly what the AM+PM smoke run
      did, at 140%.
    """
    parts = []
    for row in panel.rows:
        values = (
            row.mtm_pnl,
            row.realized_pnl,
            row.rf_accrual,
            row.cost_half_spread,
            row.cost_fees,
        )
        if any(v is None for v in values):
            return None
        mtm, realized, rf, spread, fees = values
        parts.append(abs(mtm + realized - rf + spread + fees))
    return sum(parts)


def cost_share(panel: NavPanel) -> float | None:
    """Frictions as a fraction of gross trading PnL.

    ``None`` when there is no gross PnL to take a share *of*.  A book whose
    trades netted to nothing has no meaningful cost ratio, and reporting one
    invites reading a division by noise as a finding.  Values above 1.0 are
    real and are not suppressed: they say frictions exceeded everything the
    strategy generated, which is a result rather than an error.
    """
    paid = frictions(panel)
    gross = gross_pnl(panel)
    if paid is None or gross is None:
        return None
    scale = mean([abs(r.nlv) for r in panel.rows]) if panel.rows else 0.0
    if gross <= max(1e-6, scale * 1e-9):
        return None
    return paid / gross


def mean_stale_mark_share(panel: NavPanel) -> float | None:
    values = [r.stale_mark_share for r in panel.rows]
    if any(v is None for v in values):
        return None
    return mean(values)


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def compute(panel: NavPanel, rf_annual: float = 0.0) -> ArmMetrics:
    """Reduce a validated panel to one Table 1 row.

    ``rf_annual`` is a continuously-compounded annual rate. It must be supplied
    explicitly: defaulting to zero yields a *raw* Sharpe, which the report
    labels as such rather than passing off as an excess-return Sharpe.
    """

    pm = panel.pm_rows()
    if len(pm) < 2:
        raise ValueError(f"{panel.arm}/{panel.track}: need >= 2 PM observations")

    daily = log_returns(pm)
    n_obs = len(daily)
    cumulative = cumulative_log_return(panel)
    ann_log = annualized_log_return(cumulative, n_obs)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR

    sharpe_ann = sharpe(daily, rf_daily)
    mdd_daily = max_drawdown_simple(log_levels(pm))
    mdd_native = (
        max_drawdown_simple(log_levels(panel.rows)) if panel.has_am else None
    )
    annualized_simple = math.exp(ann_log) - 1.0

    return ArmMetrics(
        arm=panel.arm,
        track=panel.track,
        n_obs=n_obs,
        grid=panel.provenance.grid,
        universe=panel.provenance.universe,
        cumulative_log_return=cumulative,
        cumulative_return=math.exp(cumulative) - 1.0,
        annualized_return=annualized_simple,
        annualized_vol=annualized_vol(daily),
        sharpe=sharpe_ann,
        sharpe_se=sharpe_standard_error(sharpe_ann, n_obs),
        sortino=sortino(daily, rf_daily),
        max_drawdown_daily=mdd_daily,
        max_drawdown_native=mdd_native,
        calmar=(annualized_simple / mdd_daily) if mdd_daily > 0.0 else None,
        skewness=skewness(daily),
        excess_kurtosis=excess_kurtosis(daily),
        positive_day_share=sum(1 for d in daily if d > 0.0) / n_obs,
        turnover=turnover(panel),
        frictions=frictions(panel),
        gross_pnl=gross_pnl(panel),
        cost_share=cost_share(panel),
        mean_stale_mark_share=mean_stale_mark_share(panel),
    )


def selftest_telescoping(panel: NavPanel, tolerance: float = 1e-9) -> None:
    """Assert that the summed reward equals the endpoint criterion.

    This is the identity the whole design rests on (protocol 1.1). It is cheap,
    so it runs on every panel rather than only in tests.
    """

    summed = sum(log_returns(panel.rows))
    endpoints = cumulative_log_return(panel)
    if abs(summed - endpoints) > tolerance:
        raise AssertionError(
            f"{panel.arm}/{panel.track}: telescoping violated, "
            f"sum={summed!r} endpoints={endpoints!r}"
        )
