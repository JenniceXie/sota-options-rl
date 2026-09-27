"""MarketResolver: mark the open book to market at a grid point.

``docs/env_contract.md`` section 5.  This runs on the **mark grid**, which is
fixed at AM+PM and is deliberately not configurable: if changing when the
policy may act could also change when NAV is measured, then a decision-grid
experiment and a measurement change would be confounded, and the reward is the
measurement.

Marks are taken at **mid**, and the spread is charged in full on entry and on
exit by ``ExecutionModel``.  Mark-to-mid with no spread charged is the standard
way a backtest manufactures alpha out of a wide market, which is exactly what
``docs/evaluation_protocol.md`` section 10 sweeps ``half_spread_multiplier``
to detect.

``mark_quality`` is the honesty channel and it is why marking is a resolver
rather than a dictionary lookup:

``mid``
    Every leg priced from a two-sided quote at this grid point, within the
    staleness bound — from the chain, or from the out-of-chain NBBO described
    below.
``stale``
    The contract could not be priced here at all; the last mark is carried.
    Carried marks are a NAV that has stopped moving, which reads as low
    volatility rather than as missing data, so the share of book marked this
    way is tracked on every step.  **It is not yet enforced:**
    ``RiskControls.max_stale_mark_share`` is computed and recorded but never
    compared against, so this is a diagnostic, not a gate.
``intrinsic``
    Past expiry, or close enough that only intrinsic remains.  Not an
    approximation: at expiry intrinsic *is* the value.

**Out-of-chain marking.**  A leg drops out of the chain slice exactly when its
quote goes wide, thin or stale — which is when its value is moving most — so
carrying ``entry_price`` is a NAV error that is *systematically* signed rather
than merely noisy.  When a ``MarkQuotes`` source is configured, such a leg is
marked at the last NBBO at or before the grid point instead, and its Greeks are
recovered by inverting that mid through the same binomial the chain builder
uses (``data/features/iv_surface.py``), borrowing rate, dividends, spot and
time-to-expiry from a sibling contract on the same expiry.  Greeks recovered
this way are on the chain's convention and are therefore summable with it.

This path is **marking only**.  The chain stays the single source of tradeable
prices, with its build-time quality tiers and its env-time ``true_age`` gate,
and neither applies here — see ``env/markquotes.py`` for why conflating the two
is how a backtest fills at prices no one could have traded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

from ...data.features.iv_surface import (
    implied_volatility_from_midpoint,
    model_theta,
    model_vega,
)
from ..book import Book, Leg, Position, PositionMark
from ..chain import ChainQuote, ChainSlice, NoChainData, OptionChain
from ..markquotes import MarkQuotes, NullMarkQuotes
from ..payoff import PayoffLeg, value_bounds
from ..spec import EnvConfig

__all__ = ["MarketResolver", "MarkReport", "MARKET_RESOLVER_VERSION"]

MARKET_RESOLVER_VERSION = "market_resolver.v2"


@dataclass(frozen=True, slots=True)
class MarkReport:
    """Marks for every open position, plus what it cost in quality."""

    marks: Mapping[str, PositionMark]
    spots: Mapping[str, float]
    stale_positions: tuple[str, ...]
    expired_positions: tuple[str, ...]
    quality_counts: Mapping[str, int]
    #: Positions holding at least one leg valued from an out-of-chain NBBO.
    #: Reported separately rather than as a ``mark_quality`` tier: the mark is a
    #: real two-sided quote and degrading it would understate the book, but a
    #: run where most of the NAV came from this path is a different experiment
    #: from one where none of it did, and that has to be visible.
    repriced_positions: tuple[str, ...] = ()
    #: Positions whose mark fell outside the package's own structural value
    #: range and was clamped to it.  This must be empty.  It is not an
    #: advisory: the value of a single-expiry package provably cannot lie
    #: outside ``payoff.value_bounds``, so a non-empty tuple means the
    #: valuation is wrong, not that the market is unusual.  Carried rather
    #: than raised because a mid-episode exception destroys the ledger that
    #: would diagnose it.
    unbounded_positions: tuple[str, ...] = ()
    resolver_version: str = MARKET_RESOLVER_VERSION


@dataclass(frozen=True, slots=True)
class _Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float


def _quote_index(chain_slice: ChainSlice | None) -> dict[str, ChainQuote]:
    """``contract_id -> quote`` for everything this slice can value.

    ``markable`` first so that ``quotes`` overwrites it: a contract which is
    both tradeable and priced must be read through the stronger admission.
    Marking deliberately sees more of the chain than opening does -- the book
    can be holding a contract it could not open today (deep in the money, or
    inside ``min_dte``), and refusing to value it is what sent the old code to
    the entry price.
    """
    if chain_slice is None:
        return {}
    index = {q.contract_id: q for q in chain_slice.markable}
    index.update({q.contract_id: q for q in chain_slice.quotes})
    return index


class MarketResolver:
    """Values the book.  Never trades, never mutates cash."""

    version = MARKET_RESOLVER_VERSION

    def __init__(
        self, config: EnvConfig, chain: OptionChain, marks: MarkQuotes | None = None
    ) -> None:
        self._config = config
        self._chain = chain
        # Defaults to the null source so there is one code path.  A run
        # configured without it values the book exactly as it did before
        # out-of-chain marking existed, which is what makes the two comparable.
        self._marks: MarkQuotes = marks if marks is not None else NullMarkQuotes()

    def mark(
        self,
        book: Book,
        *,
        trade_date: date,
        session: str,
        decision_time: datetime,
    ) -> MarkReport:
        slices: dict[str, ChainSlice | None] = {}
        marks: dict[str, PositionMark] = {}
        spots: dict[str, float] = {}
        stale: list[str] = []
        expired: list[str] = []
        repriced: list[str] = []
        unbounded: list[str] = []
        counts: dict[str, int] = {}

        for position in book.positions.values():
            underlying = position.underlying
            if underlying not in slices:
                slices[underlying] = self._slice(underlying, trade_date, session, decision_time)

        # One lookup per *underlying*, built here rather than inside the per-
        # position loop.  A book of eight positions cannot tell the difference,
        # but ``clairvoyant_oracle`` marks thousands of packages against the same
        # ten slices on every step, and rebuilding a slice-sized dict once per
        # position is the entire cost of that pass.  Nothing about the merge
        # changes; it only stopped being repeated.
        quotes = {name: _quote_index(sl) for name, sl in slices.items()}

        # Warm every out-of-chain quote in one batch before marking anything.
        # Fetched lazily leg by leg these are serial round trips; the source
        # parallelises a batch, and marking is the only place that knows the
        # whole set in advance.
        self._marks.prefetch(self._unpriceable(book, quotes, trade_date), decision_time)

        for position in book.positions.values():
            chain_slice = slices[position.underlying]
            mark, used_marks = self._mark_position(
                position,
                chain_slice,
                quotes[position.underlying],
                trade_date,
                decision_time,
            )
            mark, violated = _within_structural_bounds(position, mark)
            if violated:
                unbounded.append(position.position_id)
            marks[position.position_id] = mark
            counts[mark.quality] = counts.get(mark.quality, 0) + 1
            if mark.quality == "stale":
                stale.append(position.position_id)
            if mark.quality == "intrinsic":
                expired.append(position.position_id)
            if used_marks:
                repriced.append(position.position_id)
            if chain_slice is not None and chain_slice.quotes:
                spots[position.underlying] = chain_slice.quotes[0].underlying_price

        # A share balance can outlive the option position it hedged.  Price it
        # anyway: the caller marks shares from ``spots``, so a ticker missing
        # here would keep its last mark forever and quietly freeze that part of
        # NAV at a stale price.
        for ticker in book.shares:
            if ticker in spots:
                continue
            if ticker not in slices:
                slices[ticker] = self._slice(ticker, trade_date, session, decision_time)
            chain_slice = slices[ticker]
            if chain_slice is not None and chain_slice.quotes:
                spots[ticker] = chain_slice.quotes[0].underlying_price

        return MarkReport(
            marks=marks,
            spots=spots,
            stale_positions=tuple(stale),
            expired_positions=tuple(expired),
            quality_counts=counts,
            repriced_positions=tuple(repriced),
            unbounded_positions=tuple(unbounded),
        )

    # -- pieces ----------------------------------------------------------

    @staticmethod
    def _unpriceable(
        book: Book, quotes: Mapping[str, Mapping[str, ChainQuote]], trade_date: date
    ) -> list[str]:
        """Contracts the chain cannot price here, so need an out-of-chain NBBO.

        Expired legs are excluded: they are worth intrinsic regardless of any
        quote, so fetching one would be a request that cannot change an answer.
        """
        wanted: list[str] = []
        for position in book.positions.values():
            quoted = quotes.get(position.underlying, {})
            for leg in position.legs:
                if leg.contract_id not in quoted and (leg.expiry - trade_date).days > 0:
                    wanted.append(leg.contract_id)
        return wanted

    def _slice(
        self, underlying: str, trade_date: date, session: str, decision_time: datetime
    ) -> ChainSlice | None:
        try:
            return self._chain.slice_for(
                underlying, trade_date=trade_date, session=session, decision_time=decision_time
            )
        except NoChainData:
            return None

    def _mark_position(
        self,
        position: Position,
        chain_slice: ChainSlice | None,
        quotes: Mapping[str, ChainQuote],
        trade_date: date,
        decision_time: datetime,
    ) -> tuple[PositionMark, bool]:
        """Mark one package, falling back leg by leg rather than all-or-nothing.

        A four-leg iron condor with one unquoted wing is still 75% observable,
        and treating the whole package as stale would both overstate the stale
        share and freeze three legs that did move.  So the fallback is
        per-leg — but the *quality* is the worst of the legs, because a package
        containing one carried leg is not a live mark.

        ``quotes`` is this underlying's merged lookup, built once per slice by
        ``_quote_index`` and handed in rather than rebuilt here.
        """
        total = 0.0
        delta = gamma = vega = theta = 0.0
        quality = "mid"
        min_dte = 10**6
        spot = chain_slice.quotes[0].underlying_price if chain_slice and chain_slice.quotes else 0.0
        used_marks = False

        for leg in position.legs:
            dte = (leg.expiry - trade_date).days
            min_dte = min(min_dte, dte)
            quote = quotes.get(leg.contract_id)

            if dte <= 0:
                intrinsic = self._intrinsic(leg.right, leg.strike, spot)
                total += leg.ratio * intrinsic * leg.multiplier
                quality = _worst(quality, "intrinsic")
                continue

            if quote is None:
                priced = self._reprice(leg, chain_slice, decision_time)
                if priced is None:
                    # Carry the leg's share of the last package mark.  Using the
                    # last *package* mark divided by leg count would smear a
                    # moved leg across an unmoved one, so the entry price is
                    # used instead: it is the last price this specific leg is
                    # known to have had.
                    total += leg.ratio * leg.entry_price * leg.multiplier
                    quality = _worst(quality, "stale")
                    continue
                price, greeks = priced
                used_marks = True
                total += leg.ratio * price * leg.multiplier
                if greeks is None:
                    # Priced but not differentiated: no sibling on this expiry
                    # to borrow pricing inputs from, or the inversion did not
                    # converge.  The mark is sound and is kept, but a leg
                    # contributing value and no risk would understate the
                    # book's exposure, so the package is not called ``mid``.
                    quality = _worst(quality, "stale")
                    continue
                delta += leg.ratio * greeks.delta * leg.multiplier
                gamma += leg.ratio * greeks.gamma * leg.multiplier
                vega += leg.ratio * greeks.vega * leg.multiplier
                theta += leg.ratio * greeks.theta * leg.multiplier
                continue

            total += leg.ratio * quote.mid * quote.multiplier
            if not quote.greeks_known:
                # A priced leg whose Greeks could not be established.  Its zero
                # delta means "unknown", so adding it would report a naked leg
                # as flat and invite the hedge resolver to trade against it.
                # The price is real, so the mark is kept and downgraded.
                quality = _worst(quality, "stale")
                continue
            delta += leg.ratio * quote.delta * quote.multiplier
            gamma += leg.ratio * quote.gamma * quote.multiplier
            vega += leg.ratio * quote.vega * quote.multiplier
            theta += leg.ratio * quote.theta * quote.multiplier

        mark = PositionMark(
            mark=total,
            dollar_delta=delta * spot,
            dollar_gamma=gamma * spot * spot / 100.0,
            dollar_vega=vega,
            dollar_theta=theta,
            quality=quality,
            min_dte=max(0, min_dte if min_dte < 10**6 else 0),
        )
        return mark, used_marks

    def _reprice(
        self, leg: Leg, chain_slice: ChainSlice | None, decision_time: datetime
    ) -> tuple[float, _Greeks | None] | None:
        """Value one out-of-chain leg at the last NBBO on or before the step.

        Returns ``(price, greeks)``, with ``greeks`` ``None`` when the mid is
        trustworthy but nothing on this expiry is available to invert it
        against, or ``None`` when there is no usable quote at all.
        """
        quote = self._marks.quote(leg.contract_id, decision_time)
        if quote is None:
            return None
        bound = self._config.marking.max_mark_quote_age_seconds
        if quote.age_seconds < 0.0 or quote.age_seconds > bound:
            # A negative age is a quote printed after the step: not staleness
            # but its opposite, and marking against it would be look-ahead.
            return None
        price = quote.mid
        if price <= 0.0:
            return None
        donor = _pricing_donor(chain_slice, leg)
        if donor is None:
            return price, None
        return price, _invert_greeks(
            price, leg, donor, self._config.marking.american_tree_steps
        )

    @staticmethod
    def _intrinsic(right: str, strike: float, spot: float) -> float:
        if spot <= 0:
            return 0.0
        return max(0.0, spot - strike) if right == "call" else max(0.0, strike - spot)


#: Tolerance on the structural bound, in account currency per package.  A cent
#: absorbs float error on a sum of leg values; anything the defect produced was
#: three to five orders of magnitude larger than this.
_BOUND_TOLERANCE = 0.01


def _tolerance(position: Position) -> float:
    """How far outside its structural range a package's *midpoint* may sit.

    A midpoint is not a price.  ``value_bounds`` constrains what the package can
    be *worth*, and what is observed is a sum of independent per-leg midpoints,
    each of which may sit anywhere inside its own bid-ask.  A vertical whose
    long leg is quoted wide and short leg tight therefore has a mid a tick or
    two outside the strike width without anything being wrong: the tradeable
    interval still straddles the bound.  So the honest test is whether the
    package's *quoted interval* clears the bound, and the half-spread is the
    half-width of that interval.

    Measured before this was spread-aware, on the full menu enumerated at
    2025-04-01 (scripts/analysis/clairvoyant_oracle.py): a flat 0.01 flagged 249
    exits, with a **median excess of $25 per package** -- one or two cents a
    share across three or four legs, which is one tick. Clamping those and
    stamping them ``stale`` would have buried the real signal in noise a
    hundred times its size, on the exact families (``dv``, ``cv``, ``bf``,
    ``ic``) the defect lives in.

    The floor keeps the check meaningful when the spread is unobserved: an
    all-zero-spread package still gets float-error room and nothing more.
    """
    spread = sum(
        abs(leg.ratio) * leg.entry_half_spread * leg.multiplier
        for leg in position.legs
    )
    return max(_BOUND_TOLERANCE, spread)


def _within_structural_bounds(
    position: Position, mark: PositionMark
) -> tuple[PositionMark, bool]:
    """Clamp a package mark into the range its own leg set permits.

    The last line of defence, and deliberately independent of every mechanism
    that could produce a bad mark: it re-derives the bound from the legs rather
    than trusting any price, so it catches a mispriced leg, a mixed-provenance
    package and an arithmetic error alike.

    Clamping rather than raising because the mark is consumed inside an episode
    and an exception there destroys the ledger needed to diagnose it.  The
    caller records the position id, and a run that reports any is not a run
    with an unusual market — it is a run with a broken valuation.
    """
    legs = [
        PayoffLeg(
            right=leg.right,
            strike=leg.strike,
            ratio=leg.ratio,
            entry_price=0.0,
            multiplier=leg.multiplier,
        )
        for leg in position.legs
    ]
    low, high = value_bounds(legs)
    if low - _tolerance(position) <= mark.mark <= high + _tolerance(position):
        return mark, False
    clamped = min(max(mark.mark, low), high)
    return replace(mark, mark=clamped, quality=_worst(mark.quality, "stale")), True


def _pricing_donor(chain_slice: ChainSlice | None, leg: Leg) -> ChainQuote | None:
    """A contract whose pricing inputs stand in for a leg missing from the slice.

    Restricted to the **same expiry**, because that is the boundary the inputs
    actually respect.  Measured over 848 ``(underlying, expiry, decision_time)``
    groups on 2025-06-04, ``underlying_price`` and ``continuous_dividend`` were
    constant in every group and ``interest_rate`` varied by at most 7e-06, so
    borrowing them across strikes costs nothing.  ``time_to_expiry`` is the
    reason for the restriction: it is expiry-specific, and it differs between
    calls and puts on the same expiry (AM- versus PM-settled roots), so the
    same right is preferred and the nearest strike breaks the tie.
    """
    if chain_slice is None or not chain_slice.quotes:
        return None
    same_expiry = [q for q in chain_slice.quotes if q.expiry == leg.expiry]
    if not same_expiry:
        return None
    candidates = [q for q in same_expiry if q.right == leg.right] or same_expiry
    return min(candidates, key=lambda q: abs(q.strike - leg.strike))


def _invert_greeks(
    price: float, leg: Leg, donor: ChainQuote, american_steps: int
) -> _Greeks | None:
    """Recover Greeks from a mid, the way the chain builder does.

    Same functions, same ``american_steps``, same signed-Black-Scholes delta
    convention, so a repriced leg can be added to a chain-priced one without
    reconciling two models.
    """
    if donor.time_to_expiry <= 0.0 or donor.underlying_price <= 0.0:
        return None
    style = donor.exercise_style if donor.exercise_style in {"american", "european"} else "american"
    common: dict[str, Any] = {
        "spot": donor.underlying_price,
        "strike": leg.strike,
        "time_to_expiry": donor.time_to_expiry,
        "rate": donor.interest_rate,
        "dividend_yield": donor.continuous_dividend,
        "option_right": leg.right,
        "exercise_style": style,
        "discrete_dividend": donor.discrete_dividend,
    }
    try:
        inverted = implied_volatility_from_midpoint(
            midpoint=price,
            price_tolerance=max(1.0e-6, price * 1.0e-6),
            max_iterations=36,
            american_steps=american_steps,
            **common,
        )
        iv = inverted.implied_volatility
        # A mid below intrinsic or above the expanded bracket resolves to no IV.
        # That is a statement about the quote, not an error, and the caller
        # keeps the price while dropping the risk.
        if iv is None or iv <= 0.0 or inverted.delta is None or inverted.gamma is None:
            return None
        vega = model_vega(volatility=iv, american_steps=american_steps, **common)
        theta = model_theta(volatility=iv, american_steps=american_steps, **common)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return _Greeks(
        delta=float(inverted.delta),
        gamma=float(inverted.gamma),
        vega=float(vega),
        theta=float(theta),
    )


#: Worst-first, so ``_worst`` is a max over this order.
_QUALITY_RANK = {"mid": 0, "intrinsic": 1, "stale": 2}


def _worst(current: str, candidate: str) -> str:
    return current if _QUALITY_RANK[current] >= _QUALITY_RANK[candidate] else candidate
