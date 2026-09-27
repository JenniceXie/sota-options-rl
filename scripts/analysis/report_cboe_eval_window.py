"""Table 1's published-series rows, computed on the evaluation window.

These are the ``SeriesArm`` rows of ``eval/arms.py`` — Cboe strategy indices
whose levels are published — reduced to metrics by the same ``eval.metrics``
code every other arm uses. Nothing here is a replication: no order is placed,
no cost model is applied, and the ``RuleArm`` replications on our own universe
are *not* included, because they are still ``status="pending"``.

Read the ``basis`` column before comparing two rows. A ``total return`` level
carries the whole equity risk premium and an ``option overlay`` level does not,
so their returns are not two measurements of the same thing.

Usage::

    .venv/bin/python scripts/analysis/report_cboe_eval_window.py
    .venv/bin/python scripts/analysis/report_cboe_eval_window.py --end 2025-08-29
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

from portfolio_monkey.eval import arms
from portfolio_monkey.eval.builders.from_series import (
    SeriesCoverageError,
    build_composite_panel,
    build_series_panel,
    read_level_table,
)
from portfolio_monkey.eval.metrics import compute

SOURCE = "cboe_strategy_indices/index_levels.csv"


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _window(path: Path) -> tuple[date, date, int, set[date]]:
    """First and last date of the evaluation window, read from the config."""
    dates = [
        _parse_date(line.strip())
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return dates[0], dates[-1], len(dates), set(dates)


def main() -> None:
    default_root = Path(
        os.environ["PM_DATA_ROOT"]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels", type=Path, default=default_root / "work" / SOURCE
    )
    parser.add_argument("--dates", type=Path, default=Path("configs/dates_eval.txt"))
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument(
        "--exact",
        action="store_true",
        help="measure on the decision dates only, not every exchange close",
    )
    args = parser.parse_args()

    start, end, n_dates, decision_dates = _window(args.dates)
    start = args.start or start
    end = args.end or end
    dates, columns = read_level_table(args.levels)

    if args.exact:
        # Intersect the level table with the decision grid, so the series are
        # measured on exactly the dates the policy trades rather than on every
        # exchange close in the interval. Levels are dropped, not forward
        # filled: the resulting return spans a gap and that is the honest
        # number for a date the policy never saw.
        keep = [i for i, d in enumerate(dates) if d in decision_dates]
        dates = [dates[i] for i in keep]
        columns = {t: [col[i] for i in keep] for t, col in columns.items()}

    in_window = sum(1 for d in dates if start <= d <= end)
    print(f"window      {start} .. {end}  ({n_dates} decision dates in {args.dates})")
    print(f"levels      {args.levels}")
    print(f"grid        {'decision dates only' if args.exact else 'every exchange close'}"
          f" -> {in_window} level dates in window")
    print("sharpe/sortino are RAW (rf = 0); see protocol section 8\n")

    # Every series row the registry declares, in registry order, plus the two
    # equal-weight composites. Rule replications are deliberately absent.
    rows: list[tuple[str, str, str, str]] = []
    for spec in arms.REGISTRY:
        if spec.kind == "series":
            rows.append((spec.name, spec.underlying, spec.strategy, spec.basis))

    header = (
        f"{'arm':<10}{'underlying':<20}{'basis':<16}"
        f"{'n':>5}{'TR%':>9}{'AR%':>9}{'VOL%':>8}"
        f"{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MDD%':>8}"
    )
    print(header)
    print("-" * len(header))

    missing: list[str] = []
    for name, underlying, _strategy, basis in rows:
        try:
            panel = build_series_panel(
                name,
                dates,
                columns,
                start,
                end,
                universe="published",
                source=SOURCE,
            )
        except SeriesCoverageError as exc:
            missing.append(f"{name}: {exc}")
            continue
        m = compute(panel)
        print(
            f"{name:<10}{underlying:<20}{basis:<16}{m.n_obs:>5}"
            f"{100 * m.cumulative_return:>9.2f}{100 * m.annualized_return:>9.2f}"
            f"{100 * m.annualized_vol:>8.2f}"
            f"{_fmt(m.sharpe):>8}{_fmt(m.sortino):>9}{_fmt(m.calmar):>8}"
            f"{100 * m.max_drawdown_daily:>8.2f}"
        )

    for spec in arms.COMPOSITE_ARMS:
        tickers = arms.CW_TICKERS if spec.name.endswith("cw_ew") else arms.DI_TICKERS
        try:
            panel = build_composite_panel(
                spec.name,
                tickers,
                dates,
                columns,
                start,
                end,
                universe="9 of ours",
                source=SOURCE,
            )
        except SeriesCoverageError as exc:
            missing.append(f"{spec.name}: {exc}")
            continue
        m = compute(panel)
        print(
            f"{spec.name:<10}{spec.underlying:<20}{spec.basis:<16}{m.n_obs:>5}"
            f"{100 * m.cumulative_return:>9.2f}{100 * m.annualized_return:>9.2f}"
            f"{100 * m.annualized_vol:>8.2f}"
            f"{_fmt(m.sharpe):>8}{_fmt(m.sortino):>9}{_fmt(m.calmar):>8}"
            f"{100 * m.max_drawdown_daily:>8.2f}"
        )

    if missing:
        print("\nunavailable over this window:")
        for line in missing:
            print(f"  {line}")


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
