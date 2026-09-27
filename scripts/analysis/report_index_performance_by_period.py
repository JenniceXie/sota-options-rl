"""Index and Cboe strategy-index performance over the SFT / RL / eval windows.

Scored with **the same six metrics the SFT trajectories are scored with** --
``performance_metrics`` is imported from ``clairvoyant_oracle`` rather than
reimplemented, so TR, AVOL, MDD, ASR, ACR and ASoR here mean exactly what they
mean in the trajectory tables. That matters because the convention is not the
one ``eval/metrics.py`` uses: these are simple returns, the annualized return is
a **CAGR**, ``ASR`` is the daily Sharpe times ``sqrt(252)`` and not CAGR/AVOL,
and ``ACR`` is CAGR/MDD.

Series are measured on the **decision dates** of each window, not on every
exchange close, so the grid matches the trajectories'. Cumulative return is
unaffected by that choice (shared endpoints); the annualized and risk figures
move slightly.

``SPY`` is the universe's index slot and is a **price** series -- the close
marks carry no dividend reinvestment -- so it is comparable to ``SPX`` and not
to ``SPXTR``. Cboe publishes no SPY strategy index, which is why the strategy
rows below are all SPX.

Usage::

    .venv/bin/python scripts/analysis/report_index_performance_by_period.py
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
PERIODS = ("sft", "rl", "eval")

#: The index sleeve: the universe's own index slot, the two passive references,
#: then the SPX strategy suite in the registry's order.
INDEX_ROWS = (
    ("SPY", "universe index slot", "price return"),
    ("SPX", "passive reference", "price return"),
    ("SPXTR", "passive reference", "total return"),
    ("BXM", "BuyWrite", "total return"),
    ("PUT", "PutWrite, cash-secured", "collateralized"),
    ("PUTY", "PutWrite, 2% OTM", "collateralized"),
    ("WPUT", "PutWrite, weekly", "collateralized"),
    ("CNDR", "Iron condor", "collateralized"),
    ("BFLY", "Iron butterfly", "collateralized"),
    ("CLL", "95-110 collar", "total return"),
    ("CLLZ", "Zero-cost PS collar", "total return"),
    ("PPUT", "5% put protection", "total return"),
    ("RXM", "Risk reversal 25d", "collateralized"),
    ("CMBO", "Covered combo", "TR + collateral"),
    ("SVRPO", "Market-neutral VRP", "collateralized"),
)

#: One row each for the nine-name single-stock sleeve, for continuity with the
#: per-name table. Built as equal-weight composites of the published levels.
COMPOSITES = (
    ("cw_ew", "9-name CW composite", "option overlay"),
    ("di_ew", "9-name DI composite", "total return"),
)

SINGLE_STOCKS = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA")


def _parse_date(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def read_period_dates(config_dir: Path) -> dict[str, list[date]]:
    out: dict[str, list[date]] = {}
    for name in PERIODS:
        path = config_dir / f"dates_{name}.txt"
        out[name] = [
            _parse_date(line) for line in path.read_text().splitlines() if line.strip()
        ]
    return out


def read_spy_closes(root: Path, wanted: set[date]) -> dict[date, float]:
    """Daily SPY close marks from ``raw/crsp/stock_price_points``.

    Only the partitions in ``wanted`` are opened -- the dataset spans years and
    this report needs 246 days of it.
    """
    closes: dict[date, float] = {}
    base = root / "raw" / "crsp" / "stock_price_points"
    for day in sorted(wanted):
        part = base / f"date={day.isoformat()}" / "part-000.jsonl"
        if not part.exists():
            continue
        with part.open() as handle:
            for line in handle:
                if '"SPY"' not in line:
                    continue
                record = json.loads(line)
                if record.get("record_type") != "stock_close":
                    continue
                payload = record.get("payload", {})
                if payload.get("symbol") != "SPY":
                    continue
                price = payload.get("price")
                if price:
                    closes[day] = float(price)
    return closes


def composite_levels(
    tickers: tuple[str, ...],
    grid: list[date],
    dates: list[date],
    columns: dict[str, list[float | None]],
) -> list[float] | None:
    """Equal-weight, daily-rebalanced composite over ``grid``.

    Rebalancing daily makes the composite's return the arithmetic mean of the
    constituents' simple returns, which is why this cannot be done by averaging
    levels.
    """
    index = {d: i for i, d in enumerate(dates)}
    series: list[list[float]] = []
    for ticker in tickers:
        column = columns.get(ticker)
        if column is None:
            return None
        values = [column[index[d]] if d in index else None for d in grid]
        if any(v is None or v <= 0.0 for v in values):
            return None
        series.append([float(v) for v in values])  # type: ignore[arg-type]
    levels = [1.0]
    for step in range(1, len(grid)):
        mean_return = sum(s[step] / s[step - 1] - 1.0 for s in series) / len(series)
        levels.append(levels[-1] * (1.0 + mean_return))
    return levels


def series_levels(
    ticker: str,
    grid: list[date],
    dates: list[date],
    columns: dict[str, list[float | None]],
) -> list[float] | None:
    column = columns.get(ticker)
    if column is None:
        return None
    index = {d: i for i, d in enumerate(dates)}
    values = [column[index[d]] if d in index else None for d in grid]
    usable = [v for v in values if v is not None and v > 0.0]
    if len(usable) < len(values):
        return None
    return [float(v) for v in usable]


def _fmt(value: float | None, scale: float = 1.0, places: int = 2) -> str:
    return "--" if value is None else f"{value * scale:.{places}f}"


def main() -> None:
    default_root = Path(
        os.environ["PM_DATA_ROOT"]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=Path,
        default=default_root / "work" / "cboe_strategy_indices" / "index_levels.csv",
    )
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--rf", type=float, default=0.0)
    args = parser.parse_args()

    periods = read_period_dates(args.configs)
    dates, columns = read_level_table(args.levels)
    all_wanted = {d for days in periods.values() for d in days}
    spy = read_spy_closes(args.data_root, all_wanted)
    columns["SPY"] = [spy.get(d) for d in dates]

    print(f"metrics     performance_metrics() from clairvoyant_oracle -- the SFT set")
    print(f"convention  simple returns; AR is CAGR; ASR = daily Sharpe * sqrt(252);")
    print(f"            ACR = CAGR/MDD; rf = {args.rf:.4f} (raw if 0)")
    print(f"levels      {args.levels}")
    print(f"SPY         {len(spy)} closes from raw/crsp/stock_price_points (price, no divs)\n")

    for name in PERIODS:
        grid = periods[name]
        print(
            f"=== {name.upper():<4} {grid[0]} .. {grid[-1]}  "
            f"({len(grid)} decision dates) ==="
        )
        header = (
            f"{'arm':<8}{'what':<24}{'basis':<16}{'n':>4}"
            f"{'TR%':>9}{'CAGR%':>10}{'AVOL%':>8}{'MDD%':>8}"
            f"{'ASR':>7}{'ACR':>8}{'ASoR':>8}"
        )
        print(header)
        print("-" * len(header))
        rows: list[tuple[str, str, str, list[float] | None]] = [
            (t, what, basis, series_levels(t, grid, dates, columns))
            for t, what, basis in INDEX_ROWS
        ]
        for key, what, basis in COMPOSITES:
            tickers = tuple(
                f"{s}{'CW' if key.startswith('cw') else 'DI'}" for s in SINGLE_STOCKS
            )
            rows.append(
                (key, what, basis, composite_levels(tickers, grid, dates, columns))
            )

        for ticker, what, basis, levels in rows:
            if levels is None or len(levels) < 2:
                print(f"{ticker:<8}{what:<24}{basis:<16}{'--':>4}   (no coverage)")
                continue
            m = performance_metrics(
                levels, trading_days_per_year=TRADING_DAYS, risk_free_annual=args.rf
            )
            print(
                f"{ticker:<8}{what:<24}{basis:<16}{m['trading_days']:>4}"
                f"{_fmt(m['total_return'], 100):>9}{_fmt(m['cagr'], 100):>10}"
                f"{_fmt(m['annual_volatility'], 100):>8}"
                f"{_fmt(m['max_drawdown'], 100):>8}"
                f"{_fmt(m['annual_sharpe']):>7}{_fmt(m['annual_calmar']):>8}"
                f"{_fmt(m['annual_sortino']):>8}"
            )
        print()


if __name__ == "__main__":
    main()
