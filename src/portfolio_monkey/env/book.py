"""The book: positions, cash, collateral, and the episode handoff.

``docs/env_contract.md`` section 8.  Two properties do all the work here.

**Cash is the only funding source.**  Naked shorts are not admitted
(``⟨Q10⟩``), so every package has a bounded max loss and collateral is that
loss held in cash.  There is no margin engine and no borrowing: cash is floored
at zero and an order that would breach the floor is refused by ``SizeResolver``
before it reaches here.  That is what makes ``nav`` a quantity this class can
compute rather than approximate.

**The book crosses episode boundaries; the context does not.**  Section 1.4:
monthly episodes are a context-window reset, not an economic one, so the reward
telescopes across the whole six-month run.  ``BookState`` is the serialized
handoff, and it is checked rather than trusted — ``nav`` is stored and then
*asserted* against a recomputation at load, because a handoff that silently
restates NAV is a handoff that can silently mint money between episodes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any

from .statespace import BookView, PositionView

__all__ = [
    "BookError",
    "InsufficientCash",
    "Leg",
    "Position",
    "PositionMark",
    "Book",
    "BookState",
    "STATE_VERSION",
]

#: ``v2`` adds ``share_entry_cost``.  A ``v1`` state carries no share basis, and
#: the NAV assert in :meth:`BookState.restore` would *not* catch its absence —
#: NAV is a function of the mark, not of the basis — so the handoff would
#: silently mis-attribute every later hedge unwind between mtm and realized.
STATE_VERSION = "book_state.v2"

#: NAV is asserted to this tolerance on load.  A dollar on a $1M book is 1e-6
#: relative; anything larger is a real discrepancy and not float noise.
NAV_TOLERANCE = 1.0


class BookError(RuntimeError):
    """Raised when the book is asked to do something it cannot represent."""


class InsufficientCash(BookError):
    """Raised when a debit would drive cash below zero.

    Reaching this is a bug in ``SizeResolver``, not a legitimate outcome: the
    cash constraint is one of the five limits that sizing minimizes over, so by
    the time an order arrives here it has already been made affordable.  The
    exception exists to make that invariant fail loudly instead of quietly
    going negative.
    """


@dataclass(frozen=True, slots=True)
class Leg:
    """One option leg of a package, at the quantity of a single package."""

    contract_id: str
    right: str
    strike: float
    expiry: date
    ratio: int
    multiplier: int = 100
    entry_price: float = 0.0
    entry_delta: float | None = None
    #: Per-contract half-spread paid on the way in.  Kept on the leg so that a
    #: leg with no live quote at exit can be charged what it actually cost to
    #: cross, rather than a constant that has no relationship to this contract.
    entry_half_spread: float = 0.0

    @property
    def is_short(self) -> bool:
        return self.ratio < 0


@dataclass(frozen=True, slots=True)
class PositionMark:
    """What ``MarketResolver`` returns for one position at one grid point."""

    mark: float
    dollar_delta: float = 0.0
    dollar_gamma: float = 0.0
    dollar_vega: float = 0.0
    dollar_theta: float = 0.0
    quality: str = "mid"
    min_dte: int = 0


@dataclass(frozen=True, slots=True)
class Position:
    """One open package.

    ``quantity`` is the package count decided by ``SizeResolver``; the legs
    carry per-package ratios, so a 4-lot credit vertical is one ``Position``
    with ``quantity = 4`` and never four positions.  Collapsing them would make
    the stop-loss and the per-underlying position cap count wrong.

    ``entry_cost`` is signed in cash terms: negative for a debit paid, positive
    for a credit received.  ``collateral`` is always non-negative.

    ``coordinates`` is the order's coordinate map, carried so that a roll
    (``X``) can inherit the coordinates it does not restate.  It is stored rather
    than parsed back out of ``strategy_handle`` because the handle renders deltas
    as integer percent and is therefore lossy below a 0.01 ``delta_step``.

    **The roll lineage is two fields, not one.**  ``rolled_from`` is the
    position this one replaced — the adjacent edge — and ``roll_root`` is the
    first position in the chain.  Both are needed: ``rolled_from`` answers "which
    two positions are linked by this action", which is what the receipt and the
    ``P`` block show, while ``roll_root`` answers "what did this *view* cost over
    its whole life", which is the attribution question and which the adjacent
    edge cannot answer without a graph walk over closed positions that the book
    no longer holds.  For a position that was opened outright, ``rolled_from`` is
    empty and ``roll_root`` is its own id, so every position is in a chain of at
    least one and no reader needs a null branch.
    """

    position_id: str
    underlying: str
    family: str
    orientation: str
    strategy_handle: str
    legs: tuple[Leg, ...]
    quantity: int
    opened_at: datetime
    entry_cost: float
    collateral: float
    max_loss_per_package: float
    coordinates: Mapping[str, float] = field(default_factory=dict)
    mark: float = 0.0
    dollar_delta: float = 0.0
    dollar_gamma: float = 0.0
    dollar_vega: float = 0.0
    dollar_theta: float = 0.0
    mark_quality: str = "mid"
    min_dte: int = 0
    entry_mark: float = 0.0
    rolled_from: str = ""
    roll_root: str = ""
    #: How many rolls precede this position.  Stored rather than derived because
    #: the chain's earlier links are closed and gone from ``Book.positions``, so
    #: at step N there is nothing left to count.
    roll_generation: int = 0

    def __post_init__(self) -> None:
        # Enforced here rather than at the one call site in ``ExecutionModel``
        # so that ``replace()`` — which ``with_mark`` runs on every position on
        # every step — cannot drop it, and so that a position decoded from a
        # book written before rolls existed comes back rooted at itself instead
        # of rooted at "".  Idempotent, which is what makes it safe under
        # ``replace``.
        if not self.roll_root:
            object.__setattr__(self, "roll_root", self.position_id)

    @property
    def unrealized_pnl(self) -> float:
        """Mark minus entry, in cash terms.

        ``entry_cost`` already carries the sign of the cash flow, so a credit
        spread that has decayed to zero shows ``+entry_cost``: the credit is
        earned.  Costs paid at entry are *not* netted out here — they are a
        realized cash outflow and belong in ``realized_pnl``, or they would be
        double-counted when the position closes.
        """
        return self.mark * self.quantity + self.entry_cost

    def view(self, as_of: datetime) -> PositionView:
        return PositionView(
            position_id=self.position_id,
            underlying=self.underlying,
            family=self.family,
            orientation=self.orientation,
            dte=self.min_dte,
            quantity=self.quantity,
            mark=self.mark,
            entry_cost=self.entry_cost,
            unrealized_pnl=self.unrealized_pnl,
            dollar_delta=self.dollar_delta,
            dollar_gamma=self.dollar_gamma,
            dollar_vega=self.dollar_vega,
            dollar_theta=self.dollar_theta,
            collateral=self.collateral,
            mark_quality=self.mark_quality,
            opened_at=self.opened_at,
            rolled_from=self.rolled_from,
            roll_generation=self.roll_generation,
        )

    def with_mark(self, mark: PositionMark) -> "Position":
        """Adopt a fresh mark, converting its greeks from per-package to per-position.

        ``MarketResolver`` sums the dollar greeks over ``self.legs``, and a leg
        is held "at the quantity of a single package" (see ``Leg`` above), so
        every greek arriving in ``mark`` is *per package*.  ``ExecutionModel``
        writes the same fields at open as ``package.delta * quantity`` — *per
        position*.  Marking runs before hedging on every step, so without this
        multiplication the per-package number is what every downstream consumer
        sees: ``HedgeResolver`` sums the field across a ticker and trades
        against the sum, which under-hedges by a factor of ``quantity``.

        Per-position is the canonical convention: ``Book.net_dollar_delta``
        sums across positions with no weights, and hedge bands apply at the
        name level, so the summands have to be true exposures.

        ``mark`` is deliberately *not* scaled.  ``Book.position_value`` already
        computes ``mark * quantity``; scaling here as well would square the
        package count.
        """
        return replace(
            self,
            mark=mark.mark,
            dollar_delta=mark.dollar_delta * self.quantity,
            dollar_gamma=mark.dollar_gamma * self.quantity,
            dollar_vega=mark.dollar_vega * self.quantity,
            dollar_theta=mark.dollar_theta * self.quantity,
            mark_quality=mark.quality,
            min_dte=mark.min_dte,
        )

    @property
    def stop_loss_reference(self) -> float:
        """The denominator of the stop-loss test (section 8.7).

        Max loss per package times the package count.  For a debit structure
        that equals the premium paid; for a credit structure it is the width
        less the credit, which is why the reference is not simply
        ``abs(entry_cost)``.
        """
        return self.max_loss_per_package * self.quantity


@dataclass(slots=True)
class Book:
    """Mutable account state.  One instance per run, not per episode."""

    cash: float
    initial_cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    peak_nav: float = 0.0
    _next_position_ordinal: int = 1

    def __post_init__(self) -> None:
        if self.peak_nav == 0.0:
            self.peak_nav = self.nav

    # -- valuation -------------------------------------------------------

    @property
    def collateral_used(self) -> float:
        return sum(p.collateral for p in self.positions.values())

    @property
    def position_value(self) -> float:
        return sum(p.mark * p.quantity for p in self.positions.values())

    @property
    def share_value(self) -> float:
        return sum(self._share_marks.get(t, 0.0) * q for t, q in self.shares.items())

    _share_marks: dict[str, float] = field(default_factory=dict)
    #: Signed cash paid for the current share balance, at reference price.
    #: Same convention as ``Position.entry_cost``: negative for a long.
    _share_entry_cost: dict[str, float] = field(default_factory=dict)

    def set_share_mark(self, ticker: str, price: float) -> None:
        self._share_marks[ticker] = price

    @property
    def share_unrealized_pnl(self) -> float:
        return self.share_value + sum(self._share_entry_cost.values())

    @property
    def nav(self) -> float:
        """Cash plus the mark of everything open.

        Collateral is *not* subtracted: it is held cash, already inside
        ``self.cash``, and subtracting it would double-count.  ``buying_power``
        is the quantity that nets it out.
        """
        return self.cash + self.position_value + self.share_value

    @property
    def buying_power(self) -> float:
        return max(0.0, self.cash - self.collateral_used)

    @property
    def collateral_utilization(self) -> float:
        nav = self.nav
        return 0.0 if nav <= 0 else self.collateral_used / nav

    @property
    def unrealized_pnl(self) -> float:
        """Paper PnL on everything open, options *and* hedge shares.

        Shares are included because a delta hedge works by offsetting the option
        leg: reporting the option leg here and leaving the share leg only in
        ``share_value`` would show a hedged book carrying its full unhedged
        swing.
        """
        return (
            sum(p.unrealized_pnl for p in self.positions.values())
            + self.share_unrealized_pnl
        )

    @property
    def drawdown(self) -> float:
        """Signed, from the running peak.  Zero or negative, never positive."""
        if self.peak_nav <= 0:
            return 0.0
        return min(0.0, self.nav / self.peak_nav - 1.0)

    def mark_peak(self) -> None:
        self.peak_nav = max(self.peak_nav, self.nav)

    # -- mutation --------------------------------------------------------

    def next_position_id(self) -> str:
        """Position ids are stable for the life of the *run*, not the episode.

        Section 1.4 requires this: a position carried across a monthly boundary
        must keep its id, or a close order written against the episode header
        refers to nothing.  The counter is therefore part of ``BookState``.
        """
        position_id = f"p{self._next_position_ordinal:02d}"
        self._next_position_ordinal += 1
        return position_id

    def debit(self, amount: float, *, allow_overdraft: bool = False) -> None:
        if amount < 0:
            raise BookError("debit takes a non-negative amount; use credit()")
        if not allow_overdraft and amount > self.cash + 1e-9:
            raise InsufficientCash(f"debit of {amount:.2f} against cash {self.cash:.2f}")
        self.cash -= amount

    def credit(self, amount: float) -> None:
        if amount < 0:
            raise BookError("credit takes a non-negative amount; use debit()")
        self.cash += amount

    def open(self, position: Position) -> None:
        if position.position_id in self.positions:
            raise BookError(f"position {position.position_id} is already open")
        if position.collateral > self.buying_power + 1e-9:
            raise InsufficientCash(
                f"collateral {position.collateral:.2f} exceeds buying power {self.buying_power:.2f}"
            )
        self.positions[position.position_id] = position

    def close(self, position_id: str, *, proceeds: float, cost: float) -> float:
        """Remove a position and realize its PnL.  Returns the realized amount.

        ``proceeds`` is the signed cash from unwinding the package (negative if
        closing a short structure costs money) and ``cost`` is the execution
        charge, always non-negative.  Both entry and exit costs land in
        ``realized_pnl``, which is what makes the T2 NAV panel's
        cost-attributed columns add up to the NAV change.
        """
        try:
            position = self.positions.pop(position_id)
        except KeyError as exc:
            raise BookError(f"no open position {position_id}") from exc
        realized = proceeds - cost + position.entry_cost
        self.cash += proceeds - cost
        self.realized_pnl += realized
        return realized

    def apply_marks(self, marks: Mapping[str, PositionMark]) -> None:
        for position_id, mark in marks.items():
            position = self.positions.get(position_id)
            if position is None:
                raise BookError(f"mark for unknown position {position_id}")
            self.positions[position_id] = position.with_mark(mark)

    def trade_shares(
        self, ticker: str, quantity: float, *, cash_delta: float, reference_price: float
    ) -> float:
        """Hedge fills.  ``quantity`` is signed; short stock is allowed here.

        Section 9 permits a short stock hedge even though short *options* are
        not admitted, because a short share position against a long call is
        covered by the call and carries a borrow cost rather than unbounded
        risk.

        Returns the PnL realized by the part of the trade that *reduces* an
        existing balance, on an average-cost basis.  Shares carry an
        ``entry_cost`` in the same signed-cash convention as ``Position``, so
        ``share_value + share_entry_cost`` is unrealized share PnL and has the
        same shape as ``Position.unrealized_pnl``.  Without this a hedge's PnL
        reached NAV through ``share_value`` but appeared in neither the
        ``mtm_pnl`` nor the ``realized_pnl`` column, so a hedged book's option
        leg showed its full swing with the offset invisible.

        The half-spread and commission are *not* handled here — like an option
        entry cost they are realized immediately by ``ExecutionModel``, so the
        basis is struck at ``reference_price`` and a freshly opened hedge starts
        at exactly zero unrealized.

        The mark is set here too.  ``share_value`` defaults an unmarked ticker
        to zero, so a balance opened and not separately marked would value the
        whole hedge at nothing; the fill is itself a price observation, and
        making the caller remember to say so is a NAV bug waiting to happen.
        """
        held = self.shares.get(ticker, 0.0)
        entry = self._share_entry_cost.get(ticker, 0.0)
        realized = 0.0

        if held != 0.0 and (quantity < 0.0) != (held < 0.0):
            # The order opposes the balance, so part of it closes.  A reversing
            # order does both: it closes all of ``held`` and opens the rest.
            closed = math.copysign(min(abs(quantity), abs(held)), held)
            fraction = abs(closed) / abs(held)
            realized = fraction * (held * reference_price + entry)
            entry *= 1.0 - fraction
            opening = quantity + closed
        else:
            opening = quantity
        entry += -opening * reference_price

        new = held + quantity
        if abs(new) < 1e-9:
            self.shares.pop(ticker, None)
            self._share_entry_cost.pop(ticker, None)
            self._share_marks.pop(ticker, None)
        else:
            self.shares[ticker] = new
            self._share_entry_cost[ticker] = entry
            self._share_marks[ticker] = reference_price
        self.cash += cash_delta
        self.realized_pnl += realized
        return realized

    # -- views -----------------------------------------------------------

    def view(self, as_of: datetime) -> BookView:
        nav = self.nav
        gross = sum(abs(p.mark * p.quantity) for p in self.positions.values())
        stale = sum(
            abs(p.mark * p.quantity)
            for p in self.positions.values()
            if p.mark_quality != "mid"
        )
        ordered = sorted(self.positions.values(), key=lambda p: p.position_id)
        return BookView(
            nav=nav,
            cash=self.cash,
            buying_power=self.buying_power,
            collateral_used=self.collateral_used,
            collateral_utilization=self.collateral_utilization,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            drawdown=self.drawdown,
            net_dollar_delta=sum(p.dollar_delta for p in ordered)
            + sum(self._share_marks.get(t, 0.0) * q for t, q in self.shares.items()),
            net_dollar_gamma=sum(p.dollar_gamma for p in ordered),
            net_dollar_vega=sum(p.dollar_vega for p in ordered),
            net_dollar_theta=sum(p.dollar_theta for p in ordered),
            stale_mark_share=0.0 if gross <= 0 else stale / gross,
            shares=dict(self.shares),
            positions=tuple(p.view(as_of) for p in ordered),
        )

    def positions_for(self, underlying: str) -> tuple[Position, ...]:
        return tuple(p for p in self.positions.values() if p.underlying == underlying)

    def risk_for(self, underlying: str) -> float:
        """Open risk on one name, as a fraction of NAV.

        Uses ``max_loss`` rather than mark, because the per-name cap of
        section 6A.4 is a budget over what can still be lost, not over what is
        currently at stake.
        """
        nav = self.nav
        if nav <= 0:
            return float("inf")
        return sum(p.stop_loss_reference for p in self.positions_for(underlying)) / nav

    @property
    def total_open_risk(self) -> float:
        nav = self.nav
        if nav <= 0:
            return float("inf")
        return sum(p.stop_loss_reference for p in self.positions.values()) / nav


# ---------------------------------------------------------------------------
# The episode handoff
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookState:
    """Serialized book at an episode boundary (``docs/env_contract.md`` 1.4).

    Everything needed to restate ``t=0`` of the next episode exactly, plus a
    ``state_sha256`` so that a run manifest can prove which handoff it consumed.
    """

    version: str
    as_of: datetime
    cash: float
    initial_cash: float
    realized_pnl: float
    peak_nav: float
    nav: float
    next_position_ordinal: int
    positions: tuple[Mapping[str, Any], ...]
    shares: Mapping[str, float]
    share_marks: Mapping[str, float]
    share_entry_cost: Mapping[str, float] = field(default_factory=dict)
    resolver_versions: Mapping[str, str] = field(default_factory=dict)
    env_fingerprint: str | None = None

    @classmethod
    def capture(
        cls,
        book: Book,
        *,
        as_of: datetime,
        resolver_versions: Mapping[str, str] | None = None,
        env_fingerprint: str | None = None,
    ) -> "BookState":
        return cls(
            version=STATE_VERSION,
            as_of=as_of,
            cash=book.cash,
            initial_cash=book.initial_cash,
            realized_pnl=book.realized_pnl,
            peak_nav=book.peak_nav,
            nav=book.nav,
            next_position_ordinal=book._next_position_ordinal,
            positions=tuple(_encode_position(p) for p in sorted(book.positions.values(), key=lambda p: p.position_id)),
            shares=dict(book.shares),
            share_marks=dict(book._share_marks),
            share_entry_cost=dict(book._share_entry_cost),
            resolver_versions=dict(resolver_versions or {}),
            env_fingerprint=env_fingerprint,
        )

    def restore(self) -> Book:
        """Rebuild the book and *assert* the NAV rather than restating it.

        Section 1.4 requirement 2.  If the recomputed NAV disagrees with the
        stored one, the handoff has lost or invented value, and continuing
        would silently corrupt the reward — which telescopes across episodes
        and so would carry the error to the end of the run.
        """
        if self.version != STATE_VERSION:
            raise BookError(f"cannot restore {self.version!r}; this build writes {STATE_VERSION!r}")
        book = Book(
            cash=self.cash,
            initial_cash=self.initial_cash,
            positions={},
            shares=dict(self.shares),
            realized_pnl=self.realized_pnl,
            peak_nav=self.peak_nav,
            _next_position_ordinal=self.next_position_ordinal,
        )
        book._share_marks = dict(self.share_marks)
        book._share_entry_cost = dict(self.share_entry_cost)
        for payload in self.positions:
            position = _decode_position(payload)
            book.positions[position.position_id] = position
        if abs(book.nav - self.nav) > NAV_TOLERANCE:
            raise BookError(
                f"restored NAV {book.nav:.2f} disagrees with the stored {self.nav:.2f}; "
                "the handoff is not value-preserving"
            )
        return book

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "as_of": self.as_of.isoformat(),
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "realized_pnl": self.realized_pnl,
            "peak_nav": self.peak_nav,
            "nav": self.nav,
            "next_position_ordinal": self.next_position_ordinal,
            "positions": [dict(p) for p in self.positions],
            "shares": dict(self.shares),
            "share_marks": dict(self.share_marks),
            "share_entry_cost": dict(self.share_entry_cost),
            "resolver_versions": dict(self.resolver_versions),
            "env_fingerprint": self.env_fingerprint,
        }
        payload["state_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BookState":
        body = {k: v for k, v in payload.items() if k != "state_sha256"}
        expected = payload.get("state_sha256")
        if expected is not None:
            actual = hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if actual != expected:
                raise BookError("book state checksum does not match its contents")
        return cls(
            version=body["version"],
            as_of=datetime.fromisoformat(body["as_of"]),
            cash=body["cash"],
            initial_cash=body["initial_cash"],
            realized_pnl=body["realized_pnl"],
            peak_nav=body["peak_nav"],
            nav=body["nav"],
            next_position_ordinal=body["next_position_ordinal"],
            positions=tuple(body["positions"]),
            shares=dict(body["shares"]),
            share_marks=dict(body.get("share_marks", {})),
            share_entry_cost=dict(body.get("share_entry_cost", {})),
            resolver_versions=dict(body.get("resolver_versions", {})),
            env_fingerprint=body.get("env_fingerprint"),
        )


def _encode_position(position: Position) -> dict[str, Any]:
    return {
        "position_id": position.position_id,
        "underlying": position.underlying,
        "family": position.family,
        "orientation": position.orientation,
        "strategy_handle": position.strategy_handle,
        "quantity": position.quantity,
        "opened_at": position.opened_at.isoformat(),
        "entry_cost": position.entry_cost,
        "entry_mark": position.entry_mark,
        "collateral": position.collateral,
        "max_loss_per_package": position.max_loss_per_package,
        "mark": position.mark,
        "dollar_delta": position.dollar_delta,
        "dollar_gamma": position.dollar_gamma,
        "dollar_vega": position.dollar_vega,
        "dollar_theta": position.dollar_theta,
        "mark_quality": position.mark_quality,
        "min_dte": position.min_dte,
        "coordinates": dict(position.coordinates),
        "rolled_from": position.rolled_from,
        "roll_root": position.roll_root,
        "roll_generation": position.roll_generation,
        "legs": [
            {
                "contract_id": leg.contract_id,
                "right": leg.right,
                "strike": leg.strike,
                "expiry": leg.expiry.isoformat(),
                "ratio": leg.ratio,
                "multiplier": leg.multiplier,
                "entry_price": leg.entry_price,
                "entry_delta": leg.entry_delta,
                "entry_half_spread": leg.entry_half_spread,
            }
            for leg in position.legs
        ],
    }


def _decode_position(payload: Mapping[str, Any]) -> Position:
    return Position(
        position_id=payload["position_id"],
        underlying=payload["underlying"],
        family=payload["family"],
        orientation=payload["orientation"],
        strategy_handle=payload["strategy_handle"],
        legs=tuple(
            Leg(
                contract_id=leg["contract_id"],
                right=leg["right"],
                strike=leg["strike"],
                expiry=date.fromisoformat(leg["expiry"]),
                ratio=leg["ratio"],
                multiplier=leg["multiplier"],
                entry_price=leg["entry_price"],
                entry_delta=leg["entry_delta"],
                entry_half_spread=leg.get("entry_half_spread", 0.0),
            )
            for leg in payload["legs"]
        ),
        quantity=payload["quantity"],
        opened_at=datetime.fromisoformat(payload["opened_at"]),
        entry_cost=payload["entry_cost"],
        entry_mark=payload.get("entry_mark", 0.0),
        collateral=payload["collateral"],
        max_loss_per_package=payload["max_loss_per_package"],
        mark=payload["mark"],
        dollar_delta=payload["dollar_delta"],
        dollar_gamma=payload["dollar_gamma"],
        dollar_vega=payload["dollar_vega"],
        dollar_theta=payload["dollar_theta"],
        mark_quality=payload["mark_quality"],
        min_dte=payload["min_dte"],
        # ``.get`` on the roll fields, not ``[]``: books carried across the
        # monthly boundary are on disk right now, written before these columns
        # existed, and a resume is not the place to discover a schema change.
        # ``roll_root`` is repaired by ``__post_init__`` rather than defaulted
        # here, so an old position comes back rooted at itself.
        coordinates=dict(payload.get("coordinates") or {}),
        rolled_from=payload.get("rolled_from", ""),
        roll_root=payload.get("roll_root", ""),
        roll_generation=payload.get("roll_generation", 0),
    )
