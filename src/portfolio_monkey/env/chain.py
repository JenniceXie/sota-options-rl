"""Point-in-time access to the option chain, with the liquidity gates applied.

This is the only place in the environment that touches
``features/option_chain_snapshots``.  Everything above it sees ``ChainQuote``
objects that have already passed every gate in ``EnvConfig.resolver``, so a
resolver never has to ask whether a quote is tradeable — if it is in the chain
it returned, it is.

It may still be a *poor* market, and that is the deliberate half of the
contract.  Liquidity quality — open interest, leg spread, quote tier — is
reported on the quote as ``warnings`` rather than used to remove it, because
this module's callers cannot tell an absent contract from a nonexistent one.
``ChainSlice.nearest_delta`` has no tolerance: it returns the closest survivor
at any distance, so every row removed here becomes a strike the resolver lands
on instead, silently.  See ``ResolverBounds`` for what that cost.

**The staleness gate is computed, not read.**  The dataset's
``quote_age_seconds`` column is measured against the snapshot the quote came
from, not against the decision time.  Measured 2026-09-09: every one of the
2,095,768 ``market_open`` rows in the window is built from the *previous* day's
close snapshot (``quote_source_date = T-1``), yet reports
``quote_age_seconds ≈ 60``.  The true age on 2025-06-04 AM is 17.5 hours.

That matters more than a mislabeled column.  At the AM step the state block
already carries the realized overnight gap in ``ret`` (mean absolute gap over
the window: 43 bp AAPL/MSFT, 69 bp NVDA, 101 bp PLTR, 129 bp TSLA), so filling
against pre-gap mids hands the policy a free option worth on the order of a
sixth of a two-week ATM premium.  ``true_age_seconds`` is therefore derived
from ``quote_available_time`` and gated on, which makes an AM decision fail
resolution rather than succeed profitably.  ``EnvConfig.grid.decision_sessions``
defaults to PM-only for the same reason; this is the belt to that pair of
braces, and it survives someone re-enabling AM in a config.
"""

from __future__ import annotations

import os
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import (
    SESSION_TO_STEP,
    DatasetError,
    available_dates,
    data_root,
    read_partition,
)
from .spec import ResolverBounds

__all__ = [
    "ChainError",
    "NoChainData",
    "ChainQuote",
    "ChainSlice",
    "OptionChain",
    "REJECTION_REASONS",
    "MARK_ONLY_REASONS",
    "WARNING_REASONS",
    "CHAIN_COLUMNS",
]

DATASET = "option_chain_snapshots"

#: Projected out of a ~120-column table.  Everything here is either a gate
#: input, a pricing input, or provenance that has to survive into the ledger.
CHAIN_COLUMNS = (
    "underlying",
    "step",
    "decision_time",
    "contract_id",
    "option_ticker",
    "right",
    "strike",
    "expiry",
    "multiplier",
    "adjusted_contract",
    "deliverable_status",
    # The conjunct that makes ``adjusted_contract`` mean something.  Omitting it
    # here silently restores the old behaviour, because the gate reads a column
    # that was never projected and gets ``None``.
    "deliverable_complete",
    "exercise_style",
    "bid",
    "ask",
    "midpoint",
    "relative_spread",
    "quote_status",
    "quote_quality_tier",
    "quote_quality_reasons",
    "quote_age_seconds",
    "quote_available_time",
    "quote_source_date",
    "open_interest",
    "open_interest_economic_date",
    "project_iv",
    "project_iv_status",
    "model_delta",
    "model_gamma",
    "model_vega",
    "model_theta",
    "model_theta_units",
    "underlying_price",
    "time_to_expiry",
    "interest_rate",
    # Carried so that a contract *absent* from this slice can still be priced
    # off a sibling's inputs; see ``resolvers/market.py``.  They are pricing
    # inputs in the sense the header describes, just consumed at env time
    # rather than at build time.
    "continuous_dividend",
    "discrete_dividend",
    "pricing_model",
    "available_time",
    "max_input_available_time",
    "greek_available_time",
)

#: Factor that converts a stored ``model_theta`` into the env's convention,
#: which is per trading day (``iv_surface.THETA_DAYS_PER_YEAR``).
#:
#: The builder stamped the divisor it used into ``model_theta_units``, and this
#: is what that column is for.  Partitions written before 2026-09-20 carry
#: ``option_price_per_calendar_day`` and are 1.4484x too small for the scenario
#: arithmetic in ``resolvers/size.py``; converting here means the 246 built
#: dates do not have to be repriced to change the convention, and means a
#: mixed-vintage read cannot silently average the two.  A partition whose units
#: string is unrecognised raises rather than defaulting, because the failure
#: mode of a default is a scale error that every downstream number absorbs.
THETA_UNIT_TO_TRADING_DAY = {
    "option_price_per_trading_day": 1.0,
    "option_price_per_calendar_day": 365.0 / 252.0,
}

#: A row failing any of these is not tradeable: a package built on it would be
#: fiction.  Two of them are nonetheless *priced* — see ``MARK_ONLY_REASONS``.
REJECTION_REASONS = (
    "no_greeks",
    "quote_status",
    "one_sided",
    "stale_quote",
    "dte_window",
    "non_standard_deliverable",
    "not_point_in_time",
)

#: Not tradeable, but the row still carries a real two-sided point-in-time
#: quote, so it lands in ``ChainSlice.markable`` as well as in ``rejected``.
#:
#: The invariant is **anything the book can hold, the book can value.**  Both
#: members break it in the same direction and the damage is identical: a leg
#: that cannot be found falls back to its *entry* price
#: (``resolvers/market.py``), so a spread gets marked as a live leg minus a
#: months-old one and closes at a number outside its own structural range.
#:
#: ``no_greeks`` is the larger of the two and it fires hardest exactly when it
#: hurts most.  ``project_iv`` is inverted from the **midpoint**, and an
#: American option deep in the money quotes *through* parity — the bid is below
#: intrinsic because nobody pays parity for an option they would have to carry
#: short stock against.  No volatility reproduces such a midpoint, so the
#: builder returns a null IV and a null delta and the row is deleted.  Measured
#: on 2025-04-07, 39,769 valid rows: 10.4% have a midpoint below intrinsic and
#: **100.0% of those are dropped**, against 0.7% of out-of-the-money rows.  The
#: loss is therefore conditioned on having moved — a package is unpriceable
#: precisely when it has gone deep in the money, which is when its mark matters.
#:
#: There is no arbitrage to protect against.  Only 0.36% of rows have an *ask*
#: below intrinsic; the bound is real but it is being applied to a midpoint,
#: which is a synthetic average rather than a price anyone can trade.
#:
#: ``dte_window`` is the same mistake at the other end of a position's life:
#: ``min_dte`` is an opening constraint, and applying it to marking makes a
#: position unpriceable in the days before expiry when it is most decided.
MARK_ONLY_REASONS = ("no_greeks", "dte_window")

#: A row failing any of these *is* returned, carrying the flag.  These describe
#: how good the market is, not whether it exists; see ``ResolverBounds``.
WARNING_REASONS = (
    "low_quote_tier",
    "wide_spread",
    "thin_open_interest",
)


class ChainError(RuntimeError):
    """Base class for chain access failures."""


class NoChainData(ChainError):
    """Raised when the chain has no partition, or no rows for a name.

    Distinct from "every row was gated out": the first is a coverage problem
    that invalidates the step, the second is a market condition the resolver
    reports as a resolution failure and the episode survives.
    """


@dataclass(frozen=True, slots=True)
class ChainQuote:
    """One tradeable contract, past every gate.

    Past every *gate* is not past every check.  ``warnings`` carries the
    advisory failures — a thin prior-day open interest, a wide leg spread, a
    quote tier below ``primary``/``fallback`` — which used to remove the row and
    now travel with it.  Anything that sizes or scores a package should read
    them; nothing needs to refuse on them.
    """

    contract_id: str
    underlying: str
    right: str
    strike: float
    expiry: date
    dte: int
    bid: float
    ask: float
    mid: float
    relative_spread: float
    open_interest: int
    delta: float
    gamma: float
    vega: float
    theta: float
    iv: float
    underlying_price: float
    multiplier: int
    true_age_seconds: float
    quote_tier: str
    #: Pricing inputs, kept so that a leg missing from the slice can borrow them
    #: from a sibling contract and be repriced from an out-of-chain NBBO.  They
    #: are *not* gate inputs and nothing tradeable depends on them.
    time_to_expiry: float = 0.0
    interest_rate: float = 0.0
    continuous_dividend: float = 0.0
    discrete_dividend: float = 0.0
    #: ``american`` for every US listed equity/ETF option; the column is null on
    #: a minority of rows, and defaulting is safer than refusing to price.
    exercise_style: str = "american"
    #: Advisory failures, in ``WARNING_REASONS`` order.  Empty for a clean quote.
    warnings: tuple[str, ...] = ()
    #: ``False`` on a ``markable`` row whose Greeks could not be established.
    #: ``delta``/``gamma``/``vega``/``theta``/``iv`` are then zero and mean
    #: "unknown", not "flat" — a portfolio delta must skip such a leg rather
    #: than add its zero, because adding it silently reports a naked position
    #: as hedged.  Every quote in ``ChainSlice.quotes`` has this ``True``.
    greeks_known: bool = True

    @property
    def half_spread(self) -> float:
        return max(0.0, (self.ask - self.bid) / 2.0)

    def signed_delta(self, right_sign: int = 1) -> float:
        return self.delta * right_sign


@dataclass(frozen=True, slots=True)
class ChainSlice:
    """Every admitted contract for one name at one grid point.

    ``rejected`` is a histogram, not a list.  It is carried into the ledger so
    that a run where nothing resolved can be diagnosed without a rerun — "342
    contracts, 300 not_point_in_time" is a different problem from "342
    contracts, 340 stale_quote", and the second is what an AM step looks like.

    ``warned`` is the same histogram for the advisory checks, and counts rows
    that *are* in ``quotes``.  The two therefore do not sum to ``n_raw`` and are
    not meant to.

    ``underlying_price`` is read off the *raw* rows, before admission.  The
    price of the underlying is a property of the market, not of whether any
    option contract on it happens to be tradeable, and taking it from
    ``quotes[0]`` made it unreadable at exactly the points where the gate
    empties the slice -- every AM step, on every date, for every name.
    """

    underlying: str
    trade_date: date
    session: str
    decision_time: datetime
    quotes: tuple[ChainQuote, ...]
    rejected: Mapping[str, int]
    n_raw: int
    warned: Mapping[str, int] = field(default_factory=dict)
    underlying_price: float = 0.0
    #: Priced but **not tradeable** — see ``MARK_ONLY_REASONS``.  These rows are
    #: also counted in ``rejected``, so the two overlap by construction and
    #: ``len(quotes) + len(markable)`` is not a partition of ``n_raw``.
    #:
    #: Deliberately a separate field rather than a flag on ``quotes``: opening
    #: and marking are different admissions, and every selection path
    #: (``for_expiry``, ``nearest_delta``, the contract resolver) reads
    #: ``quotes`` only.  Merging the two would let the resolver land a strike on
    #: a contract it is not allowed to trade, which is the failure mode the
    #: no-tolerance ``nearest_delta`` search makes silent.
    markable: tuple[ChainQuote, ...] = ()

    @property
    def expiries(self) -> tuple[date, ...]:
        return tuple(sorted({q.expiry for q in self.quotes}))

    def priced(self, contract_id: str) -> ChainQuote | None:
        """Any point-in-time quote for this contract, tradeable or not.

        The mark path's lookup.  Prefers ``quotes`` so that a contract which is
        both tradeable and priced is never read through the weaker admission.
        """
        for quote in self.quotes:
            if quote.contract_id == contract_id:
                return quote
        for quote in self.markable:
            if quote.contract_id == contract_id:
                return quote
        return None

    def for_expiry(self, expiry: date, right: str | None = None) -> tuple[ChainQuote, ...]:
        return tuple(
            q for q in self.quotes if q.expiry == expiry and (right is None or q.right == right)
        )

    def nearest_delta(
        self, expiry: date, right: str, target: float, *, exclude: frozenset[str] = frozenset()
    ) -> ChainQuote | None:
        """The contract whose |delta| is closest to ``target``.

        ``target`` is always positive; the sign of a put delta is a convention
        of the pricing model, and the action space names moneyness, not sign.
        ``exclude`` prevents two legs of one package resolving to the same
        contract, which would silently collapse a vertical into nothing.
        """
        candidates = [
            q for q in self.for_expiry(expiry, right) if q.contract_id not in exclude
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda q: abs(abs(q.delta) - target))


class OptionChain:
    """Reads and gates ``option_chain_snapshots``.

    Caches one ``(underlying, date, session)`` slice at a time per name.  The
    partition is a whole trading day of every listed name, so re-reading it per
    order would dominate the wall clock of a rollout.
    """

    def __init__(
        self,
        bounds: ResolverBounds,
        *,
        root: str | os.PathLike[str] | None = None,
        prefix: str = "features",
        cache_size: int = 24,
    ) -> None:
        self._bounds = bounds
        self._root = data_root(root) / prefix
        self._cache: dict[tuple[date, str], dict[str, ChainSlice]] = {}
        self._cache_size = cache_size
        self._dates: tuple[date, ...] | None = None

    @property
    def coverage(self) -> tuple[date, ...]:
        if self._dates is None:
            self._dates = available_dates(self._root, DATASET)
        return self._dates

    def has(self, trade_date: date) -> bool:
        return trade_date in set(self.coverage)

    def name_coverage(
        self, names: Sequence[str], dates: Sequence[date]
    ) -> dict[str, list[date]]:
        """Which of ``dates`` carry any chain rows for each of ``names``.

        A date present in ``coverage`` says only that *some* name was resolved
        that day, and that is not the question the run asks.  On the shipped
        build, ``coverage`` reports 63 dates while the five tradeable names
        appear on eleven of 315 name-dates between them, because the chain was
        resolved from a score-ranked shortlist that the evaluation universe was
        never in.  Every order then fails ``E_NO_CHAIN`` and the arm looks like
        a policy that will not trade.

        Reads two columns rather than ``CHAIN_COLUMNS``: this runs before the
        first step, over every date in the window, and its answer is a
        membership test.
        """
        present: dict[str, list[date]] = {name: [] for name in names}
        wanted = set(names)
        for trade_date in dates:
            try:
                # Consumed inside the guard: ``read_partition`` is a generator,
                # so a missing partition raises on the first ``next`` and not on
                # the call that created it.
                rows = read_partition(self._root, DATASET, trade_date, ("underlying",))
                seen = {row.get("underlying") for row in rows} & wanted
            except DatasetError:
                continue
            for name in seen:
                present[name].append(trade_date)
        return present

    def previous_covered(self, trade_date: date) -> date | None:
        """The latest covered date at or before ``trade_date``.

        Used only for expiry/assignment settlement, never to source a decision
        quote: substituting an older chain at a decision point is precisely the
        stale-fill problem this module exists to prevent.
        """
        dates = self.coverage
        index = bisect_right(dates, trade_date)
        return dates[index - 1] if index else None

    def slice_for(
        self,
        underlying: str,
        *,
        trade_date: date,
        session: str,
        decision_time: datetime,
    ) -> ChainSlice:
        key = (trade_date, session)
        by_name = self._cache.get(key)
        if by_name is None:
            by_name = self._load(trade_date, session, decision_time)
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = by_name
        result = by_name.get(underlying)
        if result is None:
            raise NoChainData(
                f"{DATASET} has no rows for {underlying} on "
                f"{trade_date.isoformat()} {session}"
            )
        return result

    # -- loading ---------------------------------------------------------

    def _load(
        self, trade_date: date, session: str, decision_time: datetime
    ) -> dict[str, ChainSlice]:
        step = SESSION_TO_STEP.get(session, session)
        try:
            # Both sessions live in one date partition, and a slice is for one
            # of them.  Asking the reader for the session means the other one is
            # never boxed into Python dicts, which on this dataset is about half
            # the rows and, measured, about half the load time.
            rows = read_partition(
                self._root, DATASET, trade_date, CHAIN_COLUMNS, equals={"step": step}
            )
        except DatasetError as exc:
            raise NoChainData(str(exc)) from exc

        accepted: dict[str, list[ChainQuote]] = {}
        markable: dict[str, list[ChainQuote]] = {}
        rejected: dict[str, dict[str, int]] = {}
        warned: dict[str, dict[str, int]] = {}
        raw: dict[str, int] = {}
        spot: dict[str, float] = {}

        for row in rows:
            # Redundant with the ``equals`` above and kept anyway: it is one
            # dict lookup on an already-halved stream, and it is what makes the
            # pushdown provably unable to change a number.  If a future format
            # or reader ignores the predicate, this loop still sees one session.
            if row.get("step") != step:
                continue
            underlying = row.get("underlying")
            if not underlying:
                continue
            raw[underlying] = raw.get(underlying, 0) + 1
            if underlying not in spot:
                price = _number(row.get("underlying_price"))
                if price is not None and price > 0:
                    spot[underlying] = price
            quote, reason = self._admit(row, trade_date, decision_time)
            if reason:
                # Counted whether or not a quote came back, so the histogram
                # keeps meaning "rows this slice will not trade" and stays
                # comparable with every run recorded before ``markable``
                # existed.  A mark-only row is in both places on purpose.
                bucket = rejected.setdefault(underlying, {})
                bucket[reason] = bucket.get(reason, 0) + 1
                if quote is not None:
                    markable.setdefault(underlying, []).append(quote)
                continue
            accepted.setdefault(underlying, []).append(quote)
            if quote.warnings:
                flags = warned.setdefault(underlying, {})
                for flag in quote.warnings:
                    flags[flag] = flags.get(flag, 0) + 1

        return {
            name: ChainSlice(
                underlying=name,
                trade_date=trade_date,
                session=session,
                decision_time=decision_time,
                quotes=tuple(sorted(accepted.get(name, ()), key=lambda q: (q.expiry, q.right, q.strike))),
                rejected=dict(rejected.get(name, {})),
                n_raw=count,
                warned=dict(warned.get(name, {})),
                underlying_price=spot.get(name, 0.0),
                markable=tuple(
                    sorted(markable.get(name, ()), key=lambda q: (q.expiry, q.right, q.strike))
                ),
            )
            for name, count in raw.items()
        }

    def _admit(
        self, row: Mapping[str, Any], trade_date: date, decision_time: datetime
    ) -> tuple[ChainQuote | None, str]:
        bounds = self._bounds

        if row.get("quote_status") != "valid":
            return None, "quote_status"

        bid = _number(row.get("bid"))
        ask = _number(row.get("ask"))
        if bounds.require_two_sided_quote and (bid is None or ask is None or bid <= 0 or ask <= 0):
            return None, "one_sided"
        if bid is None or ask is None or ask < bid:
            return None, "one_sided"

        if bounds.require_standard_deliverable:
            # ``adjusted_contract`` records that an OCC memo touched the
            # contract, not that what it delivers is unusual.  Both facts are on
            # the row and only the second one disqualifies it.  Every NVDA
            # contract in the window carries the flag from the June 2024 ten-for-
            # one split, with ``deliverable_status`` ``occ_memo_exact_components``
            # and a deliverable of exactly 100 NVDA shares — an ordinary
            # contract, adjusted the ordinary way.  Gating on the flag alone
            # deleted 2,446 of NVDA's 4,230 rows on 2024-09-03, including the
            # 0.503-delta call at strike 109, and left a "straddle" resolved onto
            # a 0.08-delta call and a 0.93-delta put with no error raised.
            # ``option_candidate_resolver`` has always used both conjuncts.
            if (
                row.get("adjusted_contract") is True
                and row.get("deliverable_complete") is not True
            ):
                return None, "non_standard_deliverable"
            multiplier = _number(row.get("multiplier"))
            if multiplier is None or int(multiplier) != bounds.expected_multiplier:
                return None, "non_standard_deliverable"
        else:
            multiplier = _number(row.get("multiplier")) or bounds.expected_multiplier

        available = _parse_time(row.get("quote_available_time")) or _parse_time(
            row.get("available_time")
        )
        if available is None:
            return None, "not_point_in_time"
        true_age = (decision_time - available).total_seconds()
        if true_age < 0:
            # The quote was published after the decision.  Not staleness — the
            # opposite — and it must never be filled against.
            return None, "not_point_in_time"
        if true_age > bounds.max_quote_age_seconds:
            return None, "stale_quote"

        expiry = _parse_date(row.get("expiry"))
        if expiry is None:
            # No expiry is not a short-dated contract, it is an unusable row:
            # nothing downstream can decide whether it has settled.
            return None, "not_point_in_time"
        dte = (expiry - trade_date).days

        mid = _number(row.get("midpoint"))
        if mid is None:
            mid = (bid + ask) / 2.0

        # -- the two mark-only gates, from here down -----------------------
        #
        # Both have passed every check that says the quote is *real*: valid
        # status, two-sided, standard deliverable, published before the
        # decision and inside the staleness bound.  What they fail is a
        # condition on *opening*, so the row is returned for marking and
        # counted as rejected.  See ``MARK_ONLY_REASONS``.
        reason = ""
        if dte < bounds.min_dte:
            reason = "dte_window"

        delta = _number(row.get(bounds.delta_field))
        iv = _number(row.get("project_iv"))
        gamma = _number(row.get("model_gamma")) or 0.0
        vega = _number(row.get("model_vega")) or 0.0
        theta = (_number(row.get("model_theta")) or 0.0) * _theta_scale(row)
        greeks_known = delta is not None and iv is not None
        if not greeks_known:
            # ``no_greeks`` outranks ``dte_window`` in the histogram so that the
            # counts stay comparable with every run recorded before this split.
            reason = "no_greeks"
            spot = _number(row.get("underlying_price")) or 0.0
            right = str(row["right"])
            intrinsic = (
                max(0.0, float(row["strike"]) - spot)
                if right == "put"
                else max(0.0, spot - float(row["strike"]))
            )
            if spot > 0.0 and intrinsic > 0.0 and mid <= intrinsic:
                # At or through parity, which is why the midpoint inversion
                # failed.  The sigma -> 0 limit is then exact rather than an
                # approximation: the option is a share of stock, so delta is
                # +-1 and the second-order Greeks vanish.  Verified against the
                # builder's own discarded bracket on AAPL 2025-04-07 --
                # ``crr_american_price_delta`` at its minimum volatility
                # returns delta = -1.000000, gamma = -1.6e-09.
                delta = -1.0 if right == "put" else 1.0
                gamma = vega = theta = 0.0
                iv = 0.0
                greeks_known = True
            else:
                # Above the expanded volatility bracket, or null for some other
                # reason.  There is no defensible delta here, so the row is
                # priced and explicitly greekless.
                delta = gamma = vega = theta = iv = 0.0

        # Advisory from here down.  A missing value warns rather than rejects:
        # not knowing the spread is a gap in the row, and dropping the contract
        # for it hides the gap behind a plausible-looking neighbour.
        tier = row.get("quote_quality_tier")
        spread = _number(row.get("relative_spread"))
        open_interest = _number(row.get("open_interest"))
        warnings = tuple(
            flag
            for flag, failed in (
                ("low_quote_tier", tier not in bounds.warn_quote_tier_outside),
                (
                    "wide_spread",
                    spread is None or spread > bounds.warn_relative_spread_above,
                ),
                (
                    "thin_open_interest",
                    open_interest is None
                    or open_interest < bounds.warn_open_interest_below,
                ),
            )
            if failed
        )

        return (
            ChainQuote(
                contract_id=str(row.get("contract_id") or row.get("option_ticker")),
                underlying=str(row["underlying"]),
                right=str(row["right"]),
                strike=float(row["strike"]),
                expiry=expiry,
                dte=dte,
                bid=bid,
                ask=ask,
                mid=mid,
                # Both are advisory now and both can be absent.  The fallbacks
                # are the honest reading of the gap rather than a flattering
                # one: derive the spread from the quote that is present, and
                # call unknown open interest zero.  ``warnings`` says which.
                relative_spread=(
                    spread
                    if spread is not None
                    else (0.0 if mid <= 0 else (ask - bid) / mid)
                ),
                open_interest=int(open_interest or 0),
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                iv=iv,
                underlying_price=_number(row.get("underlying_price")) or 0.0,
                multiplier=int(multiplier),
                true_age_seconds=true_age,
                quote_tier=str(tier),
                warnings=warnings,
                time_to_expiry=_number(row.get("time_to_expiry")) or 0.0,
                interest_rate=_number(row.get("interest_rate")) or 0.0,
                continuous_dividend=_number(row.get("continuous_dividend")) or 0.0,
                discrete_dividend=_number(row.get("discrete_dividend")) or 0.0,
                exercise_style=str(row.get("exercise_style") or "american").strip().lower(),
                greeks_known=greeks_known,
            ),
            reason,
        )


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def _theta_scale(row: Mapping[str, Any]) -> float:
    """Factor taking this row's stored theta to the env's per-trading-day one.

    Absent means the builder predates the units column entirely, which is the
    calendar-day vintage.  An *unknown* string is a different thing and raises:
    a builder that started emitting a third convention must be noticed here and
    not absorbed as 1.0.
    """

    units = row.get("model_theta_units")
    if units is None or units != units:  # None or NaN
        return THETA_UNIT_TO_TRADING_DAY["option_price_per_calendar_day"]
    try:
        return THETA_UNIT_TO_TRADING_DAY[str(units)]
    except KeyError:
        raise ChainError(f"unrecognised model_theta_units: {units!r}") from None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
