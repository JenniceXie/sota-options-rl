"""NAV panel schema: the contract between panel builders and metric computation.

This module is the boundary described in ``docs/evaluation_protocol.md`` section
0.1. Builders write rows conforming to this schema; ``metrics`` reads nothing
else. State variables (features, prompts, tool schemas, greek representations)
must never appear here, so that changing the state design cannot silently move a
reported number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Iterable, Sequence

SCHEMA_VERSION = "nav_panel.v1"

#: Sessions on the mark grid. ``PM`` is the common/metric grid (section 3).
SESSIONS = ("AM", "PM")


class SchemaError(ValueError):
    """Raised when a panel violates the schema contract."""


@dataclass(frozen=True, slots=True)
class NavRow:
    """One ``(arm, track, step_ts)`` observation of net liquidating value.

    Only ``arm``, ``track``, ``trade_date``, ``session`` and ``nlv`` are
    required. Ledger-backed arms populate the rest; ``SeriesArm`` rows built
    from published index levels leave them ``None``, which propagates to ``N/A``
    in the report rather than to a silently wrong zero.
    """

    arm: str
    track: str
    trade_date: date
    session: str
    nlv: float

    cash: float | None = None
    rf_accrual: float | None = None
    mtm_pnl: float | None = None
    realized_pnl: float | None = None
    cost_half_spread: float | None = None
    cost_fees: float | None = None
    net_dollar_delta: float | None = None
    gross_dollar_delta: float | None = None
    dollar_gamma: float | None = None
    vega_90: float | None = None
    notional_traded: float | None = None
    n_positions: int | None = None
    stale_mark_share: float | None = None

    def key(self) -> tuple[str, str, date, str]:
        return (self.arm, self.track, self.trade_date, self.session)


@dataclass(frozen=True, slots=True)
class PanelProvenance:
    """Everything needed to reproduce a reported number (section 11)."""

    schema_version: str = SCHEMA_VERSION
    source: str = ""
    grid: str = "PM"
    universe: str = ""
    cost_level: str = "n/a"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NavPanel:
    """Validated rows for a single ``(arm, track)`` plus their provenance."""

    arm: str
    track: str
    rows: tuple[NavRow, ...]
    provenance: PanelProvenance

    def pm_rows(self) -> tuple[NavRow, ...]:
        return tuple(r for r in self.rows if r.session == "PM")

    @property
    def has_am(self) -> bool:
        return any(r.session == "AM" for r in self.rows)


def validate(rows: Sequence[NavRow]) -> None:
    """Raise :class:`SchemaError` on any contract violation.

    Checks are deliberately strict: a panel that does not validate must not be
    silently repaired, because every repair is an undocumented modelling choice.
    """

    if not rows:
        raise SchemaError("empty panel")

    arms = {r.arm for r in rows}
    tracks = {r.track for r in rows}
    if len(arms) != 1 or len(tracks) != 1:
        raise SchemaError(f"panel must hold one (arm, track); got {arms} x {tracks}")

    seen: set[tuple[str, str, date, str]] = set()
    previous: tuple[date, str] | None = None
    for row in rows:
        if row.session not in SESSIONS:
            raise SchemaError(f"unknown session {row.session!r}")
        if row.nlv is None or row.nlv <= 0.0:
            raise SchemaError(
                f"non-positive nlv {row.nlv!r} at {row.trade_date} {row.session}; "
                "ruin must be handled by the reward floor, not by the panel"
            )
        if row.key() in seen:
            raise SchemaError(f"duplicate observation {row.key()}")
        seen.add(row.key())

        current = (row.trade_date, row.session)
        if previous is not None and _order(current) <= _order(previous):
            raise SchemaError(f"rows not strictly increasing at {current}")
        previous = current


def _order(item: tuple[date, str]) -> tuple[date, int]:
    trade_date, session = item
    return (trade_date, SESSIONS.index(session))


def build_panel(
    rows: Iterable[NavRow],
    provenance: PanelProvenance,
) -> NavPanel:
    """Sort, validate and freeze rows into a :class:`NavPanel`."""

    ordered = tuple(sorted(rows, key=lambda r: _order((r.trade_date, r.session))))
    validate(ordered)
    return NavPanel(
        arm=ordered[0].arm,
        track=ordered[0].track,
        rows=ordered,
        provenance=provenance,
    )


def restrict(panel: NavPanel, dates: Iterable[date]) -> NavPanel:
    """Drop observations whose ``trade_date`` is not in ``dates``.

    Used to put every arm in a book on one date grid before any of them is
    reported, so that a difference between two rows is a difference in strategy
    and not in sampling. Restriction only ever narrows: a date the panel does
    not have cannot be created here, and a builder that interpolated one would
    be inventing data.

    The cumulative log return survives unchanged whenever the endpoints do,
    because it telescopes. Volatility and drawdown do not: both read the
    retained increments, and dropping interior marks biases drawdown shallow.
    That is the same bias the daily grid already carries against an AM+PM grid,
    which is why the report labels its drawdown columns with their grid.
    """

    keep = set(dates)
    rows = tuple(r for r in panel.rows if r.trade_date in keep)
    if len(rows) == len(panel.rows):
        return panel
    notes = panel.provenance.notes + (
        f"restricted to {len(keep)} reference dates, "
        f"dropped {len(panel.rows) - len(rows)} observation(s)",
    )
    return build_panel(rows, replace(panel.provenance, notes=notes))
