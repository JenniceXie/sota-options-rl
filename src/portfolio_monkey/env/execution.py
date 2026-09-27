"""ExecutionModel: the one place the book actually changes.

``docs/env_contract.md`` section 7.  The resolvers all return value objects and
none of them touches cash; every mutation funnels through here, which is what
makes the ledger complete by construction — a fill that did not go through
``ExecutionModel`` has no ``Fill`` record, and the T2 reconciliation of section
8.1 would not balance.

Three rules the whole cost model rests on:

**The half-spread is charged on entry and again on exit.**  Marks are taken at
mid (``resolvers/market.py``), so a book that never pays the spread earns the
full bid-ask on every round trip for free.  ``CostModel.half_spread_multiplier``
scales this and ``docs/evaluation_protocol.md`` section 10 sweeps it over
``(0, 0.25, 0.5, 1.0)`` precisely to measure how much of a result is that
artifact.

**Entry costs are realized immediately, not amortized into the position.**
``Position.entry_cost`` carries only the mid value, so a position marked at the
same snapshot at which it was filled has *exactly* zero unrealized PnL and the
entry cost shows up once, as cash out.  Amortizing the cost into the entry price
would hide it inside a mark and make the cost columns fail to add up to the NAV
change.

**Expiry is settlement, not a trade.**  Intrinsic value is exchanged with no
half-spread, because nobody crosses a market at expiry.  A short leg finishing
in the money pays ``assignment_fee`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from .actions import build_strategy_handle
from .book import Book, Leg, Position
from .chain import ChainQuote
from .resolvers.contract import ResolvedPackage
from .resolvers.hedge import ShareOrder
from .resolvers.size import SizeDecision
from .spec import EnvConfig

__all__ = ["Fill", "OpenPrice", "ExecutionModel", "EXECUTION_VERSION"]

EXECUTION_VERSION = "execution.v1"


@dataclass(frozen=True, slots=True)
class OpenPrice:
    """What opening a package would exchange, before it is opened.

    This exists so that the ``Q`` quote path and :meth:`ExecutionModel.open`
    cannot disagree.  The three numbers were computed inline in ``open`` until
    2026-09-24; a quote that recomputed them would have been a *second*
    implementation of the cost model, and the failure mode is the worst kind --
    the policy is shown a price, trades on it, and is charged a different one,
    with nothing in the ledger saying the two ever differed.  Both callers now
    read :meth:`ExecutionModel.price`, so quoted cost equals filled cost by
    construction rather than by agreement.

    Signs follow ``Fill``: ``mid_value`` is signed cash (a debit is negative)
    and the two frictions are non-negative.
    """

    mid_value: float
    half_spread: float
    fees: float

    @property
    def cost(self) -> float:
        return self.half_spread + self.fees

    @property
    def cash_delta(self) -> float:
        return self.mid_value - self.half_spread - self.fees


@dataclass(frozen=True, slots=True)
class Fill:
    """One cash-moving event, in the shape the T1 ledger needs.

    ``mid_value`` is the signed cash the package would have exchanged at mid,
    and ``half_spread``/``fees`` are the non-negative frictions on top.  Keeping
    them separate rather than reporting one net number is what lets
    ``docs/evaluation_protocol.md`` section 8.1 attribute NAV change to cost.
    """

    kind: str
    position_id: str
    underlying: str
    family: str
    strategy_handle: str
    quantity: int
    mid_value: float
    half_spread: float
    fees: float
    cash_delta: float
    realized: float
    mark_quality: str = "mid"
    reason: str = ""
    detail: Mapping[str, object] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return self.half_spread + self.fees


class ExecutionModel:
    """Applies resolved intents to the book and reports what it cost."""

    version = EXECUTION_VERSION

    def __init__(self, config: EnvConfig) -> None:
        self._config = config

    # -- opening ---------------------------------------------------------

    def price(self, package: ResolvedPackage, quantity: int) -> OpenPrice:
        """Price ``quantity`` packages without touching the book.

        Pure in both arguments and in ``config.cost``, which is what lets the
        ``Q`` path quote a candidate and :meth:`open` fill it from the same
        arithmetic.  Nothing here reads or writes ``Book``, so quoting has no
        effect on sizing, collateral or cash.
        """
        cost = self._config.cost
        return OpenPrice(
            # Signed cash: a debit is negative.
            mid_value=-package.mid_cost * quantity,
            half_spread=package.half_spread_cost * quantity * cost.half_spread_multiplier,
            fees=cost.option_fee_per_contract * self._contracts(package, quantity),
        )

    def open(
        self,
        book: Book,
        package: ResolvedPackage,
        decision: SizeDecision,
        *,
        as_of: datetime,
        replaces: Position | None = None,
    ) -> tuple[Position, Fill]:
        """Open one package, optionally as the replacement half of a roll.

        ``replaces`` is the position an ``X`` just closed.  It is passed as the
        whole ``Position`` and not as an id because the lineage is three fields
        deep — root, parent, generation — and deriving them needs the parent's
        own root and generation, which by the time this runs is no longer in
        ``book.positions``.  Passing the id would force a lookup that cannot
        succeed.
        """
        quantity = decision.quantity
        # The same call the ``Q`` verb makes, so a quote the policy acted on and
        # the fill it gets cannot be different numbers.  See ``OpenPrice``.
        priced = self.price(package, quantity)
        mid_value = priced.mid_value
        half_spread = priced.half_spread
        fees = priced.fees
        cash_delta = priced.cash_delta

        position = Position(
            position_id=book.next_position_id(),
            underlying=package.order.underlying,
            family=package.order.family,
            orientation=package.order.orientation,
            strategy_handle=build_strategy_handle(
                package.order.underlying,
                package.order.family,
                package.order.orientation,
                package.order.tenor_bucket,
                package.order.coordinates,
            ),
            legs=tuple(
                Leg(
                    contract_id=leg.contract_id,
                    right=leg.quote.right,
                    strike=leg.quote.strike,
                    expiry=leg.quote.expiry,
                    ratio=leg.ratio,
                    multiplier=leg.quote.multiplier,
                    entry_price=leg.quote.mid,
                    entry_delta=leg.quote.delta,
                    entry_half_spread=leg.quote.half_spread,
                )
                for leg in package.legs
            ),
            quantity=quantity,
            opened_at=as_of,
            # Only the mid, so that the mark taken at this same snapshot leaves
            # the position at exactly zero unrealized PnL.
            entry_cost=mid_value,
            collateral=decision.collateral,
            max_loss_per_package=package.max_loss,
            mark=package.mid_cost,
            dollar_delta=package.delta * quantity,
            dollar_gamma=package.gamma * package.underlying_price**2 * quantity / 100.0,
            dollar_vega=package.vega * quantity,
            dollar_theta=package.theta * quantity,
            entry_mark=package.mid_cost,
            min_dte=package.dte,
            coordinates=dict(package.order.coordinates),
            rolled_from="" if replaces is None else replaces.position_id,
            # "" is repaired to this position's own id by ``__post_init__``, so
            # an outright open roots itself and the two cases share one line.
            roll_root="" if replaces is None else replaces.roll_root,
            roll_generation=0 if replaces is None else replaces.roll_generation + 1,
        )

        book.open(position)
        book.cash += cash_delta
        book.realized_pnl -= half_spread + fees

        return position, Fill(
            kind="open",
            position_id=position.position_id,
            underlying=position.underlying,
            family=position.family,
            strategy_handle=position.strategy_handle,
            quantity=quantity,
            mid_value=mid_value,
            half_spread=half_spread,
            fees=fees,
            cash_delta=cash_delta,
            realized=-(half_spread + fees),
            reason=decision.binding,
            detail={
                "dte": package.dte,
                "max_loss": package.max_loss,
                "limits": dict(decision.limits),
                # Which vol the spot shock was taken from (``env_contract``
                # 6A.1.1).  Recorded here rather than left on the decision so
                # that an audit over ``fills.jsonl`` can *assert* the per-name
                # surface was read on every open, instead of trusting that the
                # fallback to the package's own vol never quietly fired --
                # which would reintroduce the package-dependent shock the
                # 2026-09-22 ruling removed, invisibly.
                "spot_sigma_source": decision.spot_sigma_source,
                # The link, on the side that knows it.  The close half of a roll
                # runs first and cannot name a replacement that does not exist
                # yet, so it only carries ``reason="roll"``; this is where the
                # pair is actually joined in ``fills.jsonl``.  A ``roll`` close
                # with no open naming it is a roll whose reopen was refused —
                # countable, which is the point of recording it this way round.
                "rolled_from": position.rolled_from,
                "roll_root": position.roll_root,
                "roll_generation": position.roll_generation,
            },
        )

    # -- closing ---------------------------------------------------------

    def close(
        self,
        book: Book,
        position: Position,
        *,
        reason: str,
        quotes: Mapping[str, ChainQuote] | None = None,
    ) -> Fill:
        """Unwind at the mark already taken this step.

        The mark comes from ``MarketResolver`` at the *same* snapshot rather
        than being re-fetched, which is what section 10 means by steps 5 and 9
        using one snapshot: re-reading the chain here would let a position be
        marked at one price and closed at another within a single instant.
        ``quotes`` is that same snapshot, passed in only for the spread.

        ``mark_quality`` rides along on the fill.  Closing a stale package is
        not an error — the position has to go somewhere — but it is a fill
        against a carried price, and a run whose exits are disproportionately
        stale has a PnL that is partly bookkeeping.
        """
        cost = self._config.cost
        proceeds = position.mark * position.quantity
        half_spread = self._exit_half_spread(position, quotes) * cost.half_spread_multiplier
        fees = cost.option_fee_per_contract * self._position_contracts(position)
        realized = book.close(
            position.position_id, proceeds=proceeds, cost=half_spread + fees
        )
        return Fill(
            kind="close",
            position_id=position.position_id,
            underlying=position.underlying,
            family=position.family,
            strategy_handle=position.strategy_handle,
            quantity=position.quantity,
            mid_value=proceeds,
            half_spread=half_spread,
            fees=fees,
            cash_delta=proceeds - half_spread - fees,
            realized=realized,
            mark_quality=position.mark_quality,
            reason=reason,
        )

    def settle_expiry(self, book: Book, position: Position, *, spot: float) -> Fill:
        """Settle expired legs at intrinsic, with no spread charged.

        A package whose legs expire on different dates cannot be settled
        piecewise here without splitting the position, so settlement is
        all-or-nothing at the *last* expiry.  Every family in the registry is
        single-expiry (``ContractResolver`` picks one expiry for all legs), so
        this is a statement of that invariant rather than a limitation.
        """
        cost = self._config.cost
        intrinsic = sum(
            leg.ratio * _intrinsic(leg.right, leg.strike, spot) * leg.multiplier
            for leg in position.legs
        )
        proceeds = intrinsic * position.quantity
        assigned = [
            leg
            for leg in position.legs
            if leg.is_short and _intrinsic(leg.right, leg.strike, spot) > 0
        ]
        fees = cost.assignment_fee * len(assigned) * position.quantity
        realized = book.close(position.position_id, proceeds=proceeds, cost=fees)
        return Fill(
            kind="expire",
            position_id=position.position_id,
            underlying=position.underlying,
            family=position.family,
            strategy_handle=position.strategy_handle,
            quantity=position.quantity,
            mid_value=proceeds,
            half_spread=0.0,
            fees=fees,
            cash_delta=proceeds - fees,
            realized=realized,
            mark_quality="intrinsic",
            reason="expiry",
            detail={"spot": spot, "assigned_legs": len(assigned)},
        )

    # -- hedging ---------------------------------------------------------

    def hedge(self, book: Book, order: ShareOrder) -> Fill:
        """Fill a share order.  ``HedgeResolver`` already priced the slippage.

        The slippage is reported as ``half_spread`` so that hedge frictions land
        in the same cost column as option frictions.  A hedge programme that
        eats its own alpha in crossing costs should be visible in the cost
        attribution, not buried in the mark.

        Three things move, not one.  ``trade_shares`` returns the P&L released
        by whatever part of this order *closes* an existing balance, on average
        cost at the reference price; slippage and commission are charged on top;
        and the surviving balance keeps a basis, so the rest of its P&L stays
        unrealized until it too is traded out.  Booking only the frictions —
        which is what this method used to do — left the entire price move on a
        hedge invisible to both ``mtm_pnl`` and ``realized_pnl``, so a hedge
        could lose money that appeared nowhere but in NAV.
        """
        slippage = abs(order.quantity) * abs(order.fill_price - order.reference_price)
        closed = book.trade_shares(
            order.ticker,
            order.quantity,
            cash_delta=order.cash_delta,
            reference_price=order.reference_price,
        )
        book.realized_pnl -= slippage + order.commission
        return Fill(
            kind="hedge",
            position_id="-",
            underlying=order.ticker,
            family="hedge",
            strategy_handle=f"{order.ticker}:hedge:shares",
            quantity=0,
            mid_value=-order.quantity * order.reference_price,
            half_spread=slippage,
            fees=order.commission,
            cash_delta=order.cash_delta,
            realized=closed - slippage - order.commission,
            reason="delta_band",
            detail={
                "shares": order.quantity,
                "exposure_before": order.exposure_before,
                "exposure_after": order.exposure_after,
                "attributed_to": list(order.attributed_to),
            },
        )

    # -- interest --------------------------------------------------------

    def accrue(self, book: Book, *, years: float, rate: float | None) -> float:
        """Credit the risk-free rate on cash and charge borrow on short stock.

        ``docs/evaluation_protocol.md`` section 5 is emphatic about this: an
        options book sitting on 90% cash that is not credited ``r_f`` carries a
        4-5% annual headwind that the index it is compared against does not pay,
        and every risk-adjusted comparison downstream inherits the error.

        Returns the net accrual so the caller can put it in the ledger rather
        than inferring it from a cash difference.
        """
        if rate is None or years <= 0:
            return 0.0
        interest = book.cash * rate * years
        borrow = sum(
            abs(quantity) * book._share_marks.get(ticker, 0.0)
            for ticker, quantity in book.shares.items()
            if quantity < 0
        ) * self._config.cost.borrow_rate_annual * years
        net = interest - borrow
        book.cash += net
        book.realized_pnl += net
        return net

    # -- pieces ----------------------------------------------------------

    @staticmethod
    def _contracts(package: ResolvedPackage, quantity: int) -> int:
        return sum(abs(leg.ratio) for leg in package.legs) * quantity

    @staticmethod
    def _position_contracts(position: Position) -> int:
        return sum(abs(leg.ratio) for leg in position.legs) * position.quantity

    @staticmethod
    def _exit_half_spread(
        position: Position, quotes: Mapping[str, ChainQuote] | None
    ) -> float:
        """Exit spread from the live quote, falling back to what entry paid.

        Live is preferred because the spread at exit is the spread that will
        actually be crossed, and it is exactly the quantity that widens in the
        stressed markets where a stop-loss fires — using the entry spread would
        make forced exits look cheapest precisely when they are dearest.

        The fallback is the leg's own ``entry_half_spread`` rather than a
        constant, so an unquoted leg is charged what *this contract* cost to
        cross rather than an average of contracts unlike it.  A leg with no
        quote is by definition one whose current spread is unobservable, and
        the position is already flagged ``stale`` on the fill.
        """
        total = 0.0
        for leg in position.legs:
            quote = (quotes or {}).get(leg.contract_id)
            per_contract = quote.half_spread if quote is not None else leg.entry_half_spread
            total += abs(leg.ratio) * per_contract * leg.multiplier
        return total * position.quantity


def _intrinsic(right: str, strike: float, spot: float) -> float:
    if spot <= 0:
        return 0.0
    return max(0.0, spot - strike) if right == "call" else max(0.0, strike - spot)


def expired_positions(book: Book, trade_date: date) -> tuple[Position, ...]:
    """Positions whose last leg has reached expiry."""
    return tuple(
        position
        for position in sorted(book.positions.values(), key=lambda p: p.position_id)
        if max(leg.expiry for leg in position.legs) <= trade_date
    )


def breached_positions(
    book: Book, config: EnvConfig, trade_date: date
) -> tuple[tuple[Position, str], ...]:
    """Positions the risk controls force out, with the reason for each.

    Two triggers, checked in this order because they mean different things:
    ``stop_loss`` is a risk decision and ``force_close_dte`` is a mechanical one
    (section 8.7 closes near expiry rather than carrying pin risk into
    settlement).  A position that trips both is reported as a stop-loss, since
    that is the fact worth counting when the stop-loss rate is reviewed.
    """
    risk = config.risk
    out: list[tuple[Position, str]] = []
    for position in sorted(book.positions.values(), key=lambda p: p.position_id):
        reference = position.stop_loss_reference
        if (
            risk.stop_loss_enabled
            and reference > 0
            and position.unrealized_pnl <= -risk.stop_loss_fraction * reference
        ):
            out.append((position, "stop_loss"))
        elif position.min_dte <= risk.force_close_dte:
            out.append((position, "force_close_dte"))
    return tuple(out)


def total_cost(fills: Sequence[Fill]) -> tuple[float, float]:
    """``(half_spread, fees)`` over a set of fills."""
    return (sum(f.half_spread for f in fills), sum(f.fees for f in fills))
