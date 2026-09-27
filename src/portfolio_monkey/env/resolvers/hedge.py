"""HedgeResolver: bring net dollar delta back inside its band with shares.

``docs/env_contract.md`` section 9.  This resolver runs *after* the policy has
acted (step 8 of the section 10 order), so a position opened this step is
hedged in the same step rather than carrying its delta overnight.

Three decisions are worth stating, because each one has a cheaper wrong version:

**The band is tested on total exposure, not on option delta alone.**  Shares
held from an earlier hedge are themselves delta, so testing the option leg in
isolation would re-hedge a book that is already flat and churn the spread every
step.  Testing the total gives the band its hysteresis for free: nothing trades
until the *combined* position leaves the band, and when option delta decays the
same rule unwinds the shares.

**Hedging to the band edge, not to zero** (``⟨Q25⟩``).  Hedging flat means the
next tick is immediately outside the band again in whichever direction it
moved, so a flat hedge pays the spread roughly twice as often for an exposure
bound that is no tighter — the bound is the band either way.

**The band may be one number or one per group** (``HedgeSpec.band_rule``).  The
fixed rule tests every group against ``delta_band * nav``; Whalley-Wilmott sizes
each group from its own gamma and its underlying's session half-spread, so the
band widens where delta runs away fastest and where correcting it costs most.
The rule is a config switch rather than a replacement because every trajectory
recorded so far was produced under the fixed band, and the two arms have to be
runnable side by side for either to mean anything.

**The resolver does not touch the book.**  It returns orders and the caller
fills them, exactly as ``ContractResolver`` returns a package and
``SizeResolver`` returns a count.  Keeping the mutation in one place is what
lets the ledger record a hedge fill the same way it records any other fill.

**Orphaned share balances are unwound to flat, unconditionally.**  When the
option position a hedge was carrying dies — closed, expired, or simply no
longer in a hedged family — the shares survive it, and a rule that only walks
``book.positions`` never looks at them again.  Left alone they are naked
directional stock, held forever, that no band ever tests.  So they are unwound
to zero rather than to the band edge (there is nothing left to hedge, so the
edge has no meaning) and without the deadband (a residue too small to trade is
exactly the residue that would otherwise be permanent).

Borrow cost on short share balances is *not* charged here.  It accrues over an
interval rather than at a fill (section 8.3), so charging it in the resolver
would attach an interval cost to an instant and would miss the intervals in
which no hedge trades at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from ..book import Book, Position
from ..spec import EnvConfig
from ..spreads import SpreadTable

__all__ = ["ShareOrder", "HedgePlan", "HedgeResolver", "HEDGE_RESOLVER_VERSION"]

HEDGE_RESOLVER_VERSION = "hedge_resolver.v1"

#: Index underlyings have no share to trade (``⟨Q6⟩``).  A position on one of
#: these is hedged through ``HedgeSpec.proxy_instruments`` if that names a
#: ticker for it, and reported as unhedgeable if it does not — the substitution
#: is a modelling choice, so it lives in the config rather than as a default
#: buried in a resolver.
#:
#: No name in the shipped ``spec.UniverseSpec`` is in this set: SPY holds the
#: index slot and hedges itself in its own shares.  That is not an accident of
#: the universe, it is one of the reasons SPY took the slot -- being unhedgeable
#: was SPX's worst property, and the alternative to swapping it out was to hedge
#: an index through a proxy.
#:
#: The proxy machinery is kept anyway, and ``proxy_instruments`` still ships with
#: ``SPX -> SPY``, so that re-admitting SPX is a one-line universe change rather
#: than a resolver change.  The entry is measured, not conventional: SPY is the
#: densest series in ``normalized/option_trades`` over the window (248,994,504
#: prints against SPX's 197,789,446), so the proxy is priceable wherever the
#: index is.  Note that this is the *normalized* layer -- an earlier reading of
#: this file concluded no SPY series existed at all, which was true only of
#: ``features/underlying_market_features``.  Back then that was the state layer
#: holding ten names none of which was SPY; now SPY is one of the ten, so the
#: distinction has stopped being load-bearing for SPY specifically and remains
#: load-bearing for any future hedge instrument outside the universe.
#:
#: What bounds an index position that has no proxy is
#: ``SizeBounds.max_net_dollar_delta``, which is charged on the book's *net*
#: delta and therefore counts the name whether or not the hedger can act on it.
#:
#: Exercise style cuts the other way from intuition here, and it is why the
#: universe swap did not make the book's life harder.  All ten names are now
#: American, so a short leg can be assigned early and hand the book 100 shares it
#: did not ask for -- a delta discontinuity this resolver does not model
#: (``payoff.py`` declines American exercise; section 8.7 treats assignment as an
#: event).  It is survivable precisely *because* shares are the hedge instrument:
#: assignment delivers the thing the hedge is already made of, so the band rule
#: absorbs it on the next pass.  SPX was European and cash-settled, so its delta
#: path was continuous to expiry with no assignment to absorb -- the easier
#: problem -- but it terminated against a settlement print (SET, struck on the
#: constituents' opening trades for AM-settled expiries) that is not a tradeable
#: price, and it had no shares with which to absorb anything.  Trading SPY takes
#: the discontinuity the resolver can at least see over the one it cannot.
#:
#: For the proxy path that remains, the same seam is the reason shares must be
#: unwound with the option leg rather than carried across it: SPY keeps trading
#: through an expiry that SPX settles against a print no one can trade.  That is
#: what ``_unwind_orphans`` does, for a reason that predates the proxy.
NON_SHARE_UNDERLYINGS = frozenset({"SPX", "XSP", "NDX", "RUT", "VIX"})

#: Orders correcting less than this fraction of the band are dropped.
#:
#: Hedging to the band edge lands exactly *on* the boundary, and the hedge's own
#: cost then lowers NAV, which shrinks the band, which puts the freshly hedged
#: exposure a few dollars outside it again — so a naive edge rule emits a
#: sub-share order on every subsequent grid point forever.  The deadband makes
#: the effective bound ``band × (1 + MIN_HEDGE_FRACTION)`` instead of ``band``,
#: which is the honest way to state the cost of not churning.  A fraction of
#: the band rather than a share count or a dollar amount, so the rule does not
#: change meaning with the price of the underlying or the size of the account.
MIN_HEDGE_FRACTION = 0.01


@dataclass(frozen=True, slots=True)
class ShareOrder:
    """A signed share order and the cash it moves.

    ``quantity`` is signed: negative is a short hedge, which section 9.3 admits
    even though short *options* are not, because the shares are covered by the
    long option delta they offset.
    """

    ticker: str
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    cash_delta: float
    exposure_before: float
    exposure_after: float
    attributed_to: tuple[str, ...] = ()

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.fill_price


@dataclass(frozen=True, slots=True)
class HedgePlan:
    """What the hedge grid wants done, and what it could not do."""

    orders: tuple[ShareOrder, ...]
    band_dollars: float
    skipped: Mapping[str, str]
    resolver_version: str = HEDGE_RESOLVER_VERSION
    #: ``(ticker, band_dollars, half_spread, share_gamma)`` per group tested.
    #:
    #: Under ``band_rule='fixed'`` every group shares ``band_dollars`` and this
    #: adds nothing.  Under Whalley-Wilmott the band is a different number for
    #: every group at every step, so ``band_dollars`` alone stops being able to
    #: explain why a hedge fired -- and "the band was 3x wider because gamma had
    #: decayed" is exactly the question a sweep over ``risk_aversion`` asks.
    #: Recording the two inputs beside the output makes the band reproducible
    #: from the log rather than only re-derivable by re-running.
    bands: tuple[tuple[str, float, float, float], ...] = ()
    #: ``(ticker, exposure)`` for every group the band rule actually tested.
    #:
    #: An empty plan is ambiguous without this and the ambiguity is expensive:
    #: a run whose exposure never left the band and a run where the hedge was
    #: never invoked both produce zero orders and zero fills, and telling them
    #: apart afterwards meant summing per-position deltas out of
    #: ``position_steps.jsonl`` by hand.  Recording what was *examined* makes
    #: "the hedger looked and declined" a positive statement.
    examined: tuple[tuple[str, float], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.orders

    @property
    def peak_exposure_ratio(self) -> float:
        """Largest tested exposure as a fraction of the band.

        Below 1.0 the hedge could not have fired; at or above it, it should
        have.  This is the one number that says whether a window exercised the
        hedge path at all, which is what a null result needs before it can be
        read as evidence that hedging does not matter.

        Each exposure is divided by *its own* band.  Dividing them all by
        ``band_dollars`` would be correct only under the fixed rule; under
        Whalley-Wilmott the bands differ per group, and a single denominator
        would report a ratio above 1.0 for a group that never came close and
        below 1.0 for one that traded.
        """
        if not self.examined:
            return 0.0
        if self.bands:
            ratios = [
                abs(exposure) / band
                for (_, exposure), (_, band, _, _) in zip(self.examined, self.bands, strict=True)
                if band > 0
            ]
            return max(ratios) if ratios else 0.0
        if self.band_dollars <= 0:
            return 0.0
        return max(abs(exposure) for _, exposure in self.examined) / self.band_dollars

    @property
    def total_cost(self) -> float:
        return sum(o.commission + abs(o.quantity) * abs(o.fill_price - o.reference_price) for o in self.orders)


class HedgeResolver:
    """Emits share orders.  Never mutates the book."""

    version = HEDGE_RESOLVER_VERSION

    def __init__(self, config: EnvConfig) -> None:
        self._config = config

    def hedge(
        self,
        book: Book,
        *,
        spots: Mapping[str, float],
        as_of: datetime,
        spreads: SpreadTable | None = None,
        session: str = "PM",
        nav: float | None = None,
    ) -> HedgePlan:
        """``nav`` overrides the band's reference equity.  Leave it ``None``.

        The environment never passes it: ``None`` means ``book.nav``, which is
        the only correct answer when a real account is being hedged, and the
        production path is unchanged by this argument existing.

        It exists for the clairvoyant oracle's upper bound, where the band has to
        be a *constant*.  ``band = delta_band * book.nav`` makes one position's
        hedge depend on every other position's marks, and a min-cost flow needs
        each trip's cost fixed before selection -- so an arc cost computed
        against a running NAV is not an arc cost and the "bound" it produces
        bounds nothing.  Freezing the reference equity is what buys the
        exactness; it is a deliberate misstatement of the account, valid only
        because the number it feeds is a bound and not a result.
        """
        spec = self._config.hedge
        nav = book.nav if nav is None else nav
        band = spec.delta_band * nav
        if not spec.enabled or nav <= 0:
            return HedgePlan(orders=(), band_dollars=band, skipped={})
        if spec.band_rule == "whalley_wilmott" and spreads is None:
            spreads = SpreadTable.flat(spec.fallback_half_spread)

        # Keyed by the ticker whose shares do the hedging, which is the
        # underlying itself for the nine single names and the proxy for an
        # index.  Grouping on the *instrument* rather than the position's
        # underlying is what makes the proxy work at all: the share balance,
        # the spot and the orphan test all have to agree on one ticker, and
        # ``book.shares`` is keyed by the thing that trades.
        hedgeable: dict[str, list[Position]] = {}
        skipped: dict[str, str] = {}
        for position in book.positions.values():
            if position.family not in spec.hedged_families:
                continue
            ticker = spec.hedge_ticker_for(position.underlying)
            if ticker is None:
                skipped[position.position_id] = "no_share_instrument"
                continue
            spot = spots.get(ticker, 0.0)
            if spot <= 0:
                skipped[position.position_id] = "no_price"
                continue
            hedgeable.setdefault(ticker, []).append(position)

        orders: list[ShareOrder] = []
        examined: list[tuple[str, float]] = []
        bands: list[tuple[str, float, float, float]] = []
        for ticker, positions in sorted(hedgeable.items()):
            spot = spots[ticker]
            held = book.shares.get(ticker, 0.0)
            # Looked up once per ticker and used by both consumers, so the band
            # cannot be sized against one spread while the fill pays another.
            # Independent of ``band_rule``: ``measured_spread_costs`` is its own
            # flag precisely so it can be the control for a WW arm.
            half_spread = (
                spreads.half_spread(ticker, as_of.date(), session=session)
                if spreads is not None
                else None
            )
            if spec.portfolio_level:
                requests = [
                    (
                        tuple(p.position_id for p in positions),
                        sum(p.dollar_delta for p in positions),
                        sum(p.dollar_gamma for p in positions),
                        held,
                    )
                ]
            else:
                requests = self._per_position(positions, held, spot)

            for owners, option_delta, dollar_gamma, share_count in requests:
                exposure = option_delta + share_count * spot
                # Shares carry delta but no gamma, so the group's gamma is the
                # option gamma alone and the band does not move as the hedge
                # fills.  That matters: a band that shrank when it was acted on
                # would re-open the group it had just closed.
                group_band, share_gamma = self._band(
                    nav=nav,
                    spot=spot,
                    dollar_gamma=dollar_gamma,
                    half_spread=half_spread,
                    fixed=band,
                )
                examined.append((ticker, exposure))
                bands.append((ticker, group_band, half_spread or 0.0, share_gamma))
                if abs(exposure) <= group_band:
                    continue
                target = math.copysign(group_band, exposure) if spec.hedge_to_band_edge else 0.0
                order = self._order(
                    ticker,
                    shares=(target - exposure) / spot,
                    spot=spot,
                    band=group_band,
                    exposure_before=exposure,
                    owners=owners,
                    half_spread=half_spread,
                )
                if order is not None:
                    orders.append(order)

        orders.extend(
            self._unwind_orphans(
                book,
                spots=spots,
                hedged=hedgeable,
                band=band,
                skipped=skipped,
                spreads=spreads,
                as_of=as_of,
                session=session,
            )
        )
        return HedgePlan(
            orders=tuple(orders),
            band_dollars=band,
            skipped=skipped,
            examined=tuple(examined),
            bands=tuple(bands),
        )

    def _band(
        self,
        *,
        nav: float,
        spot: float,
        dollar_gamma: float,
        half_spread: float | None,
        fixed: float,
    ) -> tuple[float, float]:
        """Band half-width in dollars, and the share-gamma that produced it.

        Under ``fixed`` the gamma is reported as zero rather than as the value
        it would have had, so a log cannot be read as if the fixed arm had
        consulted a gamma it never looked at.
        """
        spec = self._config.hedge
        if spec.band_rule != "whalley_wilmott" or half_spread is None:
            return fixed, 0.0

        # ``dollar_gamma`` is the change in dollar delta per 1% move,
        # ``Gamma * S^2 / 100`` (resolvers/market.py).  The band formula wants
        # the raw share-gamma, delta per $1 move, so undo exactly that.
        share_gamma = 100.0 * dollar_gamma / (spot * spot) if spot > 0 else 0.0

        # H_shares = m * (3/2 * k * S * Gamma^2 * nav / rra)^(1/3), in delta
        # units; multiplying by the spot puts it in the dollar-delta units the
        # exposure test speaks.
        inner = 1.5 * half_spread * spot * share_gamma * share_gamma * nav / spec.risk_aversion
        band = spec.band_multiple * (inner ** (1.0 / 3.0)) * spot
        return (
            min(max(band, spec.min_band_fraction * nav), spec.max_band_fraction * nav),
            share_gamma,
        )

    def _unwind_orphans(
        self,
        book: Book,
        *,
        spots: Mapping[str, float],
        hedged: Mapping[str, list[Position]],
        band: float,
        skipped: dict[str, str],
        spreads: SpreadTable | None = None,
        as_of: datetime | None = None,
        session: str = "PM",
    ) -> list[ShareOrder]:
        """Sell out share balances that no hedgeable position stands behind.

        ``hedged`` is keyed by hedge *ticker*, which is what makes this safe for
        a proxy.  Were it keyed by the positions' underlyings, a SPY balance
        carrying an SPX hedge would match nothing, read as an orphan, and be
        unwound to flat on the very step it was opened — the hedge would pay the
        spread twice per step and never hold.  Keyed by instrument, the same one
        line both protects a live proxy balance and retires it the moment the
        index position behind it goes away.

        Keyed ``shares:<ticker>`` in ``skipped`` so that an unpriceable orphan is
        distinguishable from an unhedgeable position, whose key is a position id.
        """
        out: list[ShareOrder] = []
        for ticker, held in sorted(book.shares.items()):
            if held == 0.0 or ticker in hedged:
                continue
            spot = spots.get(ticker, 0.0)
            if spot <= 0:
                skipped[f"shares:{ticker}"] = "no_price"
                continue
            order = self._order(
                ticker,
                shares=-held,
                spot=spot,
                band=band,
                exposure_before=held * spot,
                owners=(),
                deadband=False,
                half_spread=(
                    spreads.half_spread(ticker, as_of.date(), session=session)
                    if spreads is not None and as_of is not None
                    else None
                ),
            )
            if order is not None:
                out.append(order)
        return out

    # -- pieces ----------------------------------------------------------

    @staticmethod
    def _per_position(
        positions: list[Position], held: float, spot: float
    ) -> list[tuple[tuple[str, ...], float, float, float]]:
        """Split the existing share balance across the packages that need it.

        Per-position hedging (``⟨Q26⟩``) keeps hedge PnL attributable, which
        ``docs/evaluation_protocol.md`` section 9.2 needs for paired per-name
        inference.  But shares are fungible and the book stores one balance per
        ticker, so *which* package a share belongs to is not a fact the book
        holds.  Splitting pro rata to ``|dollar_delta|`` is a stated convention:
        it is the split under which every package is hedged to the same
        fraction of its own exposure, which is the split the band rule would
        have produced had it been applied package by package from flat.

        Gamma is *not* split -- each package keeps its own, because unlike the
        share balance it is a property of the package rather than of the
        ticker.  Splitting it would be the same error as testing option delta
        in isolation, one level down.
        """
        weights = [abs(p.dollar_delta) for p in positions]
        total = sum(weights)
        if total <= 0:
            return [((p.position_id,), p.dollar_delta, p.dollar_gamma, 0.0) for p in positions]
        return [
            ((p.position_id,), p.dollar_delta, p.dollar_gamma, held * w / total)
            for p, w in zip(positions, weights, strict=True)
        ]

    def _order(
        self,
        ticker: str,
        *,
        shares: float,
        spot: float,
        band: float,
        exposure_before: float,
        owners: tuple[str, ...],
        deadband: bool = True,
        half_spread: float | None = None,
    ) -> ShareOrder | None:
        """Price a share trade, rounding **toward** the current position.

        Whole-share rounding truncates the magnitude rather than rounding to
        nearest, so the hedge can undershoot the band edge but can never
        overshoot it and flip the sign of the exposure.  An overshoot would
        create delta in the opposite direction that the same band rule then has
        to trade back, which is a spread charged twice for no change in the
        bound.
        """
        cost = self._config.cost
        if not self._config.hedge.allow_fractional_shares:
            shares = math.trunc(shares)
        if shares == 0.0:
            return None
        if deadband and abs(shares * spot) < MIN_HEDGE_FRACTION * band:
            return None

        # The flat ``stock_half_spread_bps`` is one number for ten names and two
        # sessions; measured, it is 0.09 bp for SPY and 6.8 bp for META at the
        # open, a 75x range that a single default cannot straddle.  Behind a
        # flag because it changes realized PnL, so an arm that turns it on is
        # not comparable to one that does not.
        if half_spread is not None and self._config.flags.measured_spread_costs:
            slip = spot * half_spread
        else:
            slip = spot * cost.stock_half_spread_bps / 10_000.0
        fill_price = spot + slip if shares > 0 else spot - slip
        commission = cost.stock_commission_per_share * abs(shares)
        return ShareOrder(
            ticker=ticker,
            quantity=shares,
            reference_price=spot,
            fill_price=fill_price,
            commission=commission,
            # Buying shares spends cash; selling raises it.  Commission is paid
            # either way, which is why it is subtracted outside the sign.
            cash_delta=-shares * fill_price - commission,
            exposure_before=exposure_before,
            exposure_after=exposure_before + shares * spot,
            attributed_to=owners,
        )
