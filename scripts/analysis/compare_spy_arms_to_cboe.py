"""SPY fixed-strategy arms against the Cboe SPX strategy indices, same window.

Both sides are scored with ``performance_metrics`` from ``clairvoyant_oracle`` --
the metric set the SFT trajectories use -- so the two halves of the table are
the same arithmetic on two NAV paths.

**The two halves are not the same experiment, and three differences dominate any
reading of the gap:**

1. *Leverage.* Our arms size to ``nav_fraction`` = 10% of NAV in gross premium.
   ``CNDR`` holds Treasuries equal to **10x** the strategy's maximum loss and
   ``BFLY`` 10x its worst payoff; both are fully collateralized by construction.
   A return difference is therefore mostly a capital-base difference.
2. *Tenor.* Our arms roll weekly at the ``0_7`` anchor (5 DTE). Every SPX index
   here except ``WPUT`` rolls monthly on the third Friday at ~28 DTE.
3. *Costs.* Ours pay a modelled half-spread plus per-contract fees on every leg.
   The Cboe levels are struck at a VWAP or bid/ask midpoint with no cost term.

Only three pairs are genuine methodology matches -- ``ic``/CNDR, ``ib``/BFLY and,
loosely, ``dg``/RXM (RXM is an undefined-risk 25-delta risk reversal, ours is
defined-risk). Everything else is context.

The NAV series prepends ``start_nav``: the panel's first row is written *after*
that step's fills, so a series taken from the panel alone loses the opening
trade (``sweep_algorithmic_policies.summarize`` documents the same trap).

Usage::

    .venv/bin/python scripts/analysis/compare_spy_arms_to_cboe.py \\
        --runs "$PM_ARM_ROOT/baselines_spy_full"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clairvoyant_oracle import performance_metrics  # noqa: E402

from portfolio_monkey.eval.builders.from_series import read_level_table  # noqa: E402

TRADING_DAYS = 252

#: ``our family -> the Cboe index whose methodology it replicates``.
MATCHES = {"fix.icn": "CNDR", "fix.ibn": "BFLY", "fix.dgr": "RXM", "fix.dgb": "RXM"}

SPX_ROWS = (
    ("SPX", "price return"),
    ("SPXTR", "total return"),
    ("BXM", "total return"),
    ("PUT", "collateralized"),
    ("WPUT", "collateralized"),
    ("CNDR", "collateralized"),
    ("BFLY", "collateralized"),
    ("CLL", "total return"),
    ("PPUT", "total return"),
    ("RXM", "collateralized"),
    ("CMBO", "TR + collateral"),
    ("SVRPO", "collateralized"),
)


def _d(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def arm_series(run: Path) -> tuple[list[float], dict]:
    """Daily NAV (start anchored) plus the trade-level aggregates."""

    rows = [json.loads(l) for l in (run / "nav_panel.jsonl").read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step_ts"])
    manifest = json.loads((run / "manifest.json").read_text())

    # One NAV per calendar date: the panel carries several marks per step.
    per_day: dict[str, float] = {}
    costs = notional = 0.0
    for r in rows:
        per_day[r["step_ts"][:10]] = r["nlv"]
        costs += (r.get("cost_half_spread") or 0.0) + (r.get("cost_fees") or 0.0)
        notional += abs(r.get("notional_traded") or 0.0)
    levels = [manifest["start_nav"], *[per_day[d] for d in sorted(per_day)]]

    fills = []
    fp = run / "fills.jsonl"
    if fp.exists():
        fills = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    opened: dict[str, str] = {}
    holds: list[int] = []
    for f in fills:
        pid = f.get("position_id")
        if f.get("kind") == "open":
            opened[pid] = f["step_ts"][:10]
        elif pid in opened:
            holds.append((_d(f["step_ts"][:10]) - _d(opened.pop(pid))).days)
    fill_cost = sum((f.get("half_spread") or 0.0) + (f.get("fees") or 0.0) for f in fills)
    gross = manifest["end_nav"] - manifest["start_nav"] + fill_cost
    return levels, {
        "terminal_nav": manifest["end_nav"],
        "cost": fill_cost,
        "turnover": sum(abs(f.get("mid_value") or 0.0) for f in fills),
        "gross_pnl": gross,
        "cost_over_gross": (fill_cost / abs(gross)) if gross else None,
        "hp": (sum(holds) / len(holds)) if holds else None,
        "n_open": sum(1 for f in fills if f.get("kind") == "open"),
    }


def fmt(v, scale=1.0, places=2, width=9):
    return f"{'--':>{width}}" if v is None else f"{v * scale:>{width}.{places}f}"


def main() -> None:
    root = Path(
        os.environ["PM_DATA_ROOT"]
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument(
        "--levels",
        type=Path,
        default=root / "work" / "cboe_strategy_indices" / "index_levels.csv",
    )
    p.add_argument("--dates", type=Path, default=Path("configs/dates_eval.txt"))
    p.add_argument("--rf", type=float, default=0.0)
    args = p.parse_args()

    grid = [_d(l) for l in args.dates.read_text().splitlines() if l.strip()]
    print(f"window   {grid[0]} .. {grid[-1]}  ({len(grid)} decision dates)")
    print(f"metrics  performance_metrics() -- the SFT set; rf={args.rf:g} (raw)")
    print("NOTE     leverage, tenor and costs differ between the halves; see module docstring\n")

    head = (
        f"{'arm':<18}{'match':<7}{'TR%':>9}{'termNAV':>12}{'logNAV':>9}{'CAGR%':>9}"
        f"{'AVOL%':>8}{'Sharpe':>8}{'MDD%':>8}{'turnover':>13}{'cost':>10}{'c/gross':>9}{'HP':>6}"
    )
    print(head)
    print("-" * len(head))

    arms = sorted(d for d in args.runs.iterdir() if d.is_dir() and (d / "nav_panel.jsonl").exists())
    scored = []
    for run in arms:
        name = run.name
        if name.endswith("s0001"):
            continue  # identical to s0000 on a single name; verified, not assumed
        levels, extra = arm_series(run)
        m = performance_metrics(levels, trading_days_per_year=TRADING_DAYS, risk_free_annual=args.rf)
        scored.append((name, m, extra))
    scored.sort(key=lambda t: -(t[1]["total_return"] or -9))

    for name, m, extra in scored:
        stem = name.rsplit(".", 2)[0] if name.startswith("fix.") else name
        print(
            f"{name.replace('.0_7',''):<18}{MATCHES.get(stem,''):<7}"
            f"{fmt(m['total_return'],100)}{extra['terminal_nav']:>12,.0f}"
            f"{fmt(m['log_return'],1,4)}{fmt(m['cagr'],100)}"
            f"{fmt(m['annual_volatility'],100,2,8)}{fmt(m['annual_sharpe'],1,2,8)}"
            f"{fmt(m['max_drawdown'],100,2,8)}{extra['turnover']:>13,.0f}"
            f"{extra['cost']:>10,.0f}{fmt(extra['cost_over_gross'],1,2,9)}"
            f"{fmt(extra['hp'],1,1,6)}"
        )

    print(f"\n{'-- Cboe SPX indices, same window --':<18}")
    print(head)
    print("-" * len(head))
    dates, columns = read_level_table(args.levels)
    keep = {d for d in grid}
    idx = {d: i for i, d in enumerate(dates)}
    for ticker, basis in SPX_ROWS:
        col = columns.get(ticker)
        if col is None:
            continue
        lv = [col[idx[d]] for d in grid if d in idx and col[idx[d]]]
        if len(lv) < 2:
            continue
        m = performance_metrics(lv, trading_days_per_year=TRADING_DAYS, risk_free_annual=args.rf)
        print(
            f"{ticker:<18}{basis[:6]:<7}{fmt(m['total_return'],100)}{'--':>12}"
            f"{fmt(m['log_return'],1,4)}{fmt(m['cagr'],100)}"
            f"{fmt(m['annual_volatility'],100,2,8)}{fmt(m['annual_sharpe'],1,2,8)}"
            f"{fmt(m['max_drawdown'],100,2,8)}{'--':>13}{'--':>10}{'--':>9}{'--':>6}"
        )


if __name__ == "__main__":
    main()
