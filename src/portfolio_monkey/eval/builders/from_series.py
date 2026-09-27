"""Build NAV panels from published index level series (``SeriesArm``).

An external index is a degenerate strategy that buys one unit at ``t=0`` and
holds, so its NAV panel is the rebased level series. Levels are daily closes
only, hence ``session="PM"`` and ``grid="PM"``; no AM-grid metric exists for
these arms (protocol section 6.1).
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from portfolio_monkey.eval.schema import (
    NavPanel,
    NavRow,
    PanelProvenance,
    build_panel,
)


class SeriesCoverageError(ValueError):
    """Raised when a requested ticker cannot be built over the window."""


def _parse_date(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def read_level_table(path: Path) -> tuple[list[date], dict[str, list[float | None]]]:
    """Read a wide ``date,TICKER,...`` level CSV into columns.

    Blank cells become ``None`` and are never forward-filled: a gap is a data
    fact and must surface as a coverage failure rather than a flat return.
    """

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        tickers = header[1:]
        dates: list[date] = []
        columns: dict[str, list[float | None]] = {t: [] for t in tickers}
        for row in reader:
            if not row or not row[0].strip():
                continue
            dates.append(_parse_date(row[0]))
            for ticker, cell in zip(tickers, row[1:]):
                text = cell.strip()
                columns[ticker].append(float(text) if text else None)
    return dates, columns


def build_series_panel(
    ticker: str,
    dates: list[date],
    columns: dict[str, list[float | None]],
    start: date,
    end: date,
    *,
    universe: str,
    source: str,
    track: str = "full",
    notes: tuple[str, ...] = (),
    rebase_to: float = 1.0,
) -> NavPanel:
    """Slice one ticker to ``[start, end]`` and rebase it to ``rebase_to``.

    Rebasing is cosmetic: the criterion ``log(V_T / V_0)`` and every metric
    derived from log differences are scale-invariant.
    """

    if ticker not in columns:
        raise SeriesCoverageError(f"{ticker}: not present in level table")

    pairs = [
        (d, v)
        for d, v in zip(dates, columns[ticker])
        if start <= d <= end and v is not None and v > 0.0
    ]
    if len(pairs) < 2:
        raise SeriesCoverageError(
            f"{ticker}: {len(pairs)} usable observations in [{start}, {end}]"
        )

    window_dates = [d for d in dates if start <= d <= end]
    missing = len(window_dates) - len(pairs)
    coverage_notes = list(notes)
    if missing:
        coverage_notes.append(
            f"{missing} of {len(window_dates)} window dates missing or non-positive"
        )

    base = pairs[0][1]
    rows = [
        NavRow(
            arm=ticker,
            track=track,
            trade_date=d,
            session="PM",
            nlv=value / base * rebase_to,
        )
        for d, value in pairs
    ]
    provenance = PanelProvenance(
        source=source,
        grid="PM",
        universe=universe,
        cost_level="published",
        notes=tuple(coverage_notes),
    )
    return build_panel(rows, provenance)


def build_composite_panel(
    name: str,
    tickers: Sequence[str],
    dates: list[date],
    columns: dict[str, list[float | None]],
    start: date,
    end: date,
    *,
    universe: str,
    source: str,
    track: str = "full",
    notes: tuple[str, ...] = (),
    rebase_to: float = 1.0,
) -> NavPanel:
    """Equal-weight, daily-rebalanced composite of several level series.

    Daily rebalancing means the composite return is the *arithmetic* mean of
    constituent simple returns, not the mean of log returns: rebalancing to
    equal weights each close is what makes the simple-return average the
    portfolio's realized return. Averaging log returns instead would silently
    report a buy-and-hold-of-logs quantity that no portfolio achieves.

    A date is used only if **every** constituent has a usable level on that date
    and on the prior used date, so the composite never rebalances into a name
    whose level is missing.
    """

    missing_tickers = [t for t in tickers if t not in columns]
    if missing_tickers:
        raise SeriesCoverageError(f"{name}: missing constituents {missing_tickers}")

    usable: list[tuple[date, list[float]]] = []
    for index, d in enumerate(dates):
        if not (start <= d <= end):
            continue
        values = [columns[t][index] for t in tickers]
        if any(v is None or v <= 0.0 for v in values):
            continue
        usable.append((d, [float(v) for v in values]))  # type: ignore[arg-type]

    if len(usable) < 2:
        raise SeriesCoverageError(
            f"{name}: {len(usable)} dates with all {len(tickers)} constituents "
            f"present in [{start}, {end}]"
        )

    window_dates = [d for d in dates if start <= d <= end]
    coverage_notes = list(notes)
    dropped = len(window_dates) - len(usable)
    if dropped:
        coverage_notes.append(
            f"{dropped} of {len(window_dates)} window dates dropped for incomplete "
            "constituent coverage"
        )

    level = rebase_to
    rows = [NavRow(arm=name, track=track, trade_date=usable[0][0], session="PM", nlv=level)]
    for (_, previous), (d, current) in zip(usable, usable[1:]):
        gross = sum(c / p for p, c in zip(previous, current)) / len(tickers)
        level *= gross
        rows.append(NavRow(arm=name, track=track, trade_date=d, session="PM", nlv=level))

    provenance = PanelProvenance(
        source=source,
        grid="PM",
        universe=universe,
        cost_level="published",
        notes=tuple(coverage_notes)
        + (f"equal-weight daily-rebalanced composite of {len(tickers)} series",),
    )
    return build_panel(rows, provenance)
