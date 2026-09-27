"""Per-name fixed-strategy baselines: the family x name matrix, and Cboe refs.

This is the table that answers "could a learned dynamic policy's performance be
explained by one unconditionally good fixed strategy?" It is answered by the
*spread across names*, not by any single cell: a family that wins on one name
and loses on nine is not an unconditional strategy, it is a name effect.

**On the Cboe side for the nine single stocks there is no matched pair.** Cboe's
single-stock series are `*CW` (a short OTM weekly call, delta-hedged daily) and
`*DI` (the same with the stock held). Neither has a counterpart in our action
space: a delta-hedged short call holds `delta x 100` shares against a 100-share
obligation, so `payoff.is_unbounded` refuses it, and correctly -- the loss is
genuinely unbounded. The `*CW`/`*DI` rows are therefore printed as **context,
not as methodology matches**. The only genuine matches in the whole study are
`ic`/CNDR and `ib`/BFLY, and both are SPX-only.

Usage::

    .venv/bin/python scripts/analysis/summarize_pername_baselines.py \\
        --runs "$PM_ARM_ROOT/baselines_pername"
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
from compare_spy_arms_to_cboe import arm_series  # noqa: E402

from portfolio_monkey.eval.builders.from_series import read_level_table  # noqa: E402

TRADING_DAYS = 252
NAMES = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA", "SPY")
FAMILIES = (
    "olb", "olr", "dvb", "dvr", "cvb", "cvr", "dgb", "dgr",
    "bfb", "bfr", "lsn", "lgn", "ibn", "icn",
)


def _d(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


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
    args = p.parse_args()

    grid = [_d(l) for l in args.dates.read_text().splitlines() if l.strip()]

    # arm dir name is "<NAME>.fix.<fam>.<tenor>.s0000"
    tr: dict[tuple[str, str], float] = {}
    extras: dict[tuple[str, str], dict] = {}
    hold_tr = None
    for run in sorted(d for d in args.runs.iterdir() if d.is_dir()):
        if not (run / "nav_panel.jsonl").exists():
            continue
        parts = run.name.split(".")
        levels, extra = arm_series(run)
        m = performance_metrics(levels, trading_days_per_year=TRADING_DAYS)
        if run.name == "hold":
            hold_tr = m["total_return"]
            continue
        if len(parts) < 3 or parts[1] != "fix":
            continue
        name, fam = parts[0], parts[2]
        tr[(name, fam)] = m["total_return"]
        extras[(name, fam)] = extra

    print(f"window  {grid[0]} .. {grid[-1]}  ({len(grid)} dates)")
    print(f"hold    TR {100 * hold_tr:+.2f}%" if hold_tr is not None else "hold    --")
    print("\nTotal return % by family (rows) x name (cols)\n")
    head = f"{'family':<8}" + "".join(f"{n:>8}" for n in NAMES) + f"{'mean':>9}{'min':>8}{'max':>8}{'>hold':>7}"
    print(head)
    print("-" * len(head))
    for fam in FAMILIES:
        vals = [tr.get((n, fam)) for n in NAMES]
        got = [v for v in vals if v is not None]
        if not got:
            continue
        beat = sum(1 for v in got if hold_tr is not None and v > hold_tr)
        cells = "".join("      --" if v is None else f"{100 * v:>8.1f}" for v in vals)
        print(
            f"{fam:<8}{cells}{100 * sum(got) / len(got):>9.1f}"
            f"{100 * min(got):>8.1f}{100 * max(got):>8.1f}{beat:>4}/{len(got)}"
        )

    print("\nBest family per name\n")
    for n in NAMES:
        row = [(f, tr[(n, f)]) for f in FAMILIES if (n, f) in tr]
        if not row:
            continue
        row.sort(key=lambda t: -t[1])
        best, worst = row[0], row[-1]
        print(
            f"  {n:<6} best {best[0]:<4} {100 * best[1]:+7.1f}%   "
            f"worst {worst[0]:<4} {100 * worst[1]:+7.1f}%   "
            f"spread {100 * (best[1] - worst[1]):6.1f}pp"
        )

    print("\nCboe reference, same window (CONTEXT, not matched pairs for the nine)\n")
    dates, columns = read_level_table(args.levels)
    idx = {d: i for i, d in enumerate(dates)}
    print(f"{'index':<10}{'TR%':>9}{'AVOL%':>9}{'MDD%':>8}   what")
    print("-" * 60)
    refs = [(f"{n}CW", "short weekly call, delta-hedged") for n in NAMES if n != "SPY"]
    refs += [(f"{n}DI", "the same, stock held") for n in NAMES if n != "SPY"]
    refs += [
        ("CNDR", "iron condor  <- MATCHES our ic"),
        ("BFLY", "iron butterfly  <- MATCHES our ib"),
        ("RXM", "risk reversal 25d (undefined-risk)"),
        ("SPX", "index, price return"),
        ("SPXTR", "index, total return"),
    ]
    for ticker, what in refs:
        col = columns.get(ticker)
        if col is None:
            continue
        lv = [col[idx[d]] for d in grid if d in idx and col[idx[d]]]
        if len(lv) < 2:
            continue
        m = performance_metrics(lv, trading_days_per_year=TRADING_DAYS)
        vol = m["annual_volatility"]
        print(
            f"{ticker:<10}{100 * m['total_return']:>9.2f}"
            f"{'--' if vol is None else f'{100 * vol:.2f}':>9}"
            f"{100 * m['max_drawdown']:>8.2f}   {what}"
        )


if __name__ == "__main__":
    main()
