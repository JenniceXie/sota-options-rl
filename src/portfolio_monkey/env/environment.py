"""OptionsEnv: the step transition of ``docs/env_contract.md`` section 10.

Order of operations is where a simulator quietly acquires look-ahead, so the
order here is fixed and the reasons are worth stating rather than inferring:

1.  advance the clock
2.  load the point-in-time snapshot at ``decision_time``
3.  settle expiries and assignments
4.  accrue the risk-free rate on cash
5.  ``MarketResolver.mark`` -> marks, greeks, ``mark_quality``
6.  compute pre-decision NAV and build the observation
7.  on the decision grid: policy acts -> validate -> resolve -> size -> execute
8.  on the hedge grid: ``HedgeResolver`` -> execute share orders
9.  re-mark whatever 7-8 touched, at the **same** snapshot
10. write the T1 rows and the T2 NAV row
11. check ruin and the risk limits

Steps 5 and 9 sharing a snapshot is what makes a new position's entry MTM
exactly zero: it is marked at the same mid at which it filled, so the whole
entry cost is the half-spread plus fees and nothing leaks into the mark.  And
hedging *after* the policy acts is what stops the policy's own fresh delta from
sitting unhedged overnight.

The state space is a constructor argument, not a branch.  ``docs/state_space.md``
is expected to change, and the seam is the requirement: nothing in this file
reads a feature name, a block name or a wire code.  It hands ``StepContext`` to
whatever ``EnvConfig.state_space_id`` names and appends the bytes it gets back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .actions import (
    HoldOrder,
    ParsedAction,
    QuoteOrder,
    RollBasis,
    RollOrder,
    has_quote_lines,
    parse_action,
    strip_real_name_orders,
)
from .book import Book, BookState, Position
from .chain import ChainSlice, NoChainData, OptionChain
from .execution import ExecutionModel, Fill, breached_positions, expired_positions
from .ledger import LedgerWriter
from .markquotes import MarkQuotes
from .pseudonyms import Pseudonyms, anonymize
from .resolvers import (
    ContractResolver,
    HedgePlan,
    HedgeResolver,
    MarketResolver,
    MarkReport,
    ResolutionFailure,
    ResolvedPackage,
    SizeDecision,
    SizeResolver,
    resolver_versions,
)
from .spec import ConfigError, EnvConfig
from .spreads import SpreadTable
from .statespace import (
    EpisodeContext,
    FeatureSource,
    Observation,
    StateSpace,
    StepContext,
    assert_point_in_time,
    build_state_space,
    estimate_tokens,
)

__all__ = [
    "GridPoint",
    "Episode",
    "StepView",
    "OptionsEnv",
    "DataGap",
    "build_grid",
    "monthly_episodes",
]


class DataGap(RuntimeError):
    """A snapshot the step needed is missing.

    ``docs/env_contract.md`` section 11.2: fail loudly, do not interpolate.
    Raised rather than defaulted because the defaults are all economically
    meaningful — a zero spot settles every call worthless and every put at full
    strike, which is indistinguishable in the ledger from a market that
    actually did that.
    """


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One point on the mark grid, and whether the policy may act on it.

    ``previous_trade_date`` is the grid's own predecessor, not a calendar's:
    it is filled from the coverage ``build_grid`` was handed, so a gap in the
    data is a gap here too.  It is carried on the point rather than derived in
    the environment because episodes are cut monthly and the first AM of a
    month needs the last PM of the previous one.
    """

    trade_date: date
    session: str
    timestamp: datetime
    is_decision: bool
    is_hedge: bool
    previous_trade_date: date | None = None


@dataclass(frozen=True, slots=True)
class Episode:
    """A contiguous run of grid points sharing one context window."""

    episode_id: str
    points: tuple[GridPoint, ...]

    @property
    def start_date(self) -> date:
        return self.points[0].trade_date

    @property
    def end_date(self) -> date:
        return self.points[-1].trade_date

    @property
    def n_decisions(self) -> int:
        return sum(1 for p in self.points if p.is_decision)


@dataclass(frozen=True, slots=True)
class StepView:
    """What the driver sees after ``reset`` or ``step``."""

    point: GridPoint | None
    observation: Observation | None
    nav: float
    done: bool
    terminated: str | None = None
    fills: tuple[Fill, ...] = ()
    results: tuple[Mapping[str, object], ...] = ()
    info: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------


def build_grid(
    dates: Sequence[date], config: EnvConfig, *, tzinfo: ZoneInfo | None = None
) -> tuple[GridPoint, ...]:
    """Expand a list of trading dates into mark grid points.

    Dates come from the data's own coverage rather than from a calendar.  A
    holiday calendar would say a date is a trading day and the chain would have
    no rows for it, and the difference between "the market was closed" and "the
    snapshot is missing" is exactly the distinction section 11.2 refuses to
    paper over.
    """
    zone = tzinfo or ZoneInfo(config.grid.timezone_name)
    times = {"AM": config.grid.am_time_et, "PM": config.grid.pm_time_et}
    decision = set(config.grid.decision_sessions)
    hedge = set(config.grid.hedge_sessions)
    points: list[GridPoint] = []
    previous: date | None = None
    for trade_date in sorted(dates):
        for session in config.grid.mark_sessions:
            points.append(
                GridPoint(
                    trade_date=trade_date,
                    session=session,
                    timestamp=datetime.combine(trade_date, times[session], tzinfo=zone),
                    is_decision=session in decision,
                    is_hedge=session in hedge,
                    previous_trade_date=previous,
                )
            )
        previous = trade_date
    points.sort(key=lambda p: p.timestamp)
    return tuple(points)


def monthly_episodes(points: Sequence[GridPoint], *, prefix: str = "ep") -> tuple[Episode, ...]:
    """Cut the grid into calendar months (``docs/env_contract.md`` section 1.4).

    The cut is a context-window reset, not an economic one: the book crosses
    the boundary via ``BookState`` and the reward telescopes across the whole
    run, so where the cut falls changes what the policy can remember and
    nothing else.
    """
    buckets: dict[tuple[int, int], list[GridPoint]] = {}
    for point in points:
        buckets.setdefault((point.trade_date.year, point.trade_date.month), []).append(point)
    return tuple(
        Episode(episode_id=f"{prefix}_{year:04d}_{month:02d}", points=tuple(group))
        for (year, month), group in sorted(buckets.items())
    )


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------


class OptionsEnv:
    """One run of one arm.  Episodes reset the context, not the book."""

    def __init__(
        self,
        config: EnvConfig,
        *,
        chain: OptionChain,
        features: FeatureSource,
        state_space: StateSpace | None = None,
        ledger: LedgerWriter | None = None,
        rates: Mapping[date, float] | None = None,
        marks: MarkQuotes | None = None,
        spreads: SpreadTable | None = None,
    ) -> None:
        self.config = config
        self.chain = chain
        self.features = features
        self.state_space = state_space or build_state_space(config)
        self.ledger = ledger
        self.rates = rates or {}
        # Marking only, and only for contracts the chain cannot price.  Left
        # unset the book is valued exactly as it was before out-of-chain marking
        # existed, so the two are comparable.  See ``resolvers/market.py``.
        self.marks = marks
        # ``config.mark_quote_source`` is hashed into the fingerprint, so it is
        # a *claim* about this environment, and a claim nobody checks is how
        # two runs came to share a fingerprint while one marked out of chain
        # and the other had no mark source at all.  Checked at construction
        # rather than read back off the manifest because a mismatch found after
        # the episode is an autopsy: the tokens are already spent and the
        # fingerprint is already on disk, mislabelling the run.
        declared = config.mark_quote_source
        actual = getattr(marks, "source_version", None)
        if declared != actual:
            raise ConfigError(
                "mark source disagrees with the fingerprinted config: "
                f"config.mark_quote_source={declared!r} but the source handed "
                f"to the environment is {actual!r}. These must match or the "
                "fingerprint labels an environment that was never run."
            )
        # Per-(name, session) half-spreads for the hedge band and, when
        # ``flags.measured_spread_costs`` is on, for the share fill.  Left unset
        # under the Whalley-Wilmott band rule the resolver substitutes a flat
        # table, which is the arm that isolates the band's gamma scaling from
        # its spread scaling rather than an accidental degradation.
        self.spreads = spreads

        self.contract = ContractResolver(config)
        self.size = SizeResolver(config)
        self.market = MarketResolver(config, chain, marks)
        self.hedge = HedgeResolver(config)
        self.execution = ExecutionModel(config)

        self.book = Book(cash=config.initial_cash, initial_cash=config.initial_cash)
        self.episode: Episode | None = None
        self._index = 0
        self._pending: GridPoint | None = None
        self._pending_fills: list[Fill] = []
        self._last_result: dict[str, object] | None = None
        self._last_stamp: datetime | None = None
        self._terminated: str | None = None
        self._accrual = 0.0
        self._hedge_stats: dict[str, Any] = {
            "points": 0,
            "examined": 0,
            "orders": 0,
            "peak_exposure_ratio": 0.0,
            "band_min": float("inf"),
            "band_max": 0.0,
            "skipped": {},
        }
        self._mark_stats: dict[str, Any] = {
            "points": 0,
            "unbounded_marks": 0,
            "unbounded_positions": [],
        }
        self._proposals: tuple[Mapping[str, Any], ...] = ()
        self._slices: dict[tuple[date, str, str], ChainSlice | None] = {}
        # Redrawn in ``reset`` when ``flags.anonymize`` is on, and ``None``
        # otherwise so that every other arm takes a plain identity path rather
        # than a bijection that happens to be the identity -- a map that is
        # always applied is a map whose failure is invisible.
        self._pseudo: Pseudonyms | None = None
        self._anon_stats: dict[str, Any] = {
            "steps": 0,
            "real_name_orders_refused": 0,
            "real_names": [],
        }

    # -- de-identification ------------------------------------------------

    @property
    def anonymization_stats(self) -> dict[str, Any] | None:
        """How often the policy wrote a real ticker, and which ones.

        ``None`` on an arm that is not anonymized, so a reader can tell "the
        guard never ran" from "the guard ran and caught nothing" -- a zero that
        means both is the failure this counter exists to prevent.  ``steps``
        rides along for the same reason: a refusal count with no denominator
        cannot say whether the guard saw the whole run.
        """
        if self._pseudo is None and not self._anon_stats["steps"]:
            return None
        return dict(self._anon_stats)

    @property
    def pseudonyms(self) -> Pseudonyms | None:
        """The episode's label draw, for the ledger and the audit.

        Exposed because a de-identified corpus that does not record its own map
        cannot be checked: without this, "the order named U03" and "the fill
        was in NVDA" are two unrelated claims and no reviewer can join them.
        """
        return self._pseudo

    def _mask(self, text: str, *, trade_date: date | None) -> str:
        if self._pseudo is None:
            return text
        return anonymize(text, self._pseudo, trade_date=trade_date)

    def _masked(self, observation: Observation, *, trade_date: date | None) -> Observation:
        """De-identify a rendered observation, wire and ledger copy alike.

        ``blocks`` is masked too, not just ``text``.  It is what the ledger
        records and what tests assert against, so a version that masked only the
        wire would write the real tickers into the very artifact an auditor
        would open to check that it had not.

        ``token_estimate`` is re-taken from the masked text because the count is
        the budget evidence: ``U07`` is not ``GOOGL`` and ``m17`` is not
        ``2024-10-01``, so the pre-mask number would be the size of a prompt
        that was never sent.

        **Every observation that reaches a policy must pass through here.**  It
        is a method rather than two copies of the same six lines because it did
        not used to be: the quote round rendered its price block straight off
        the state space, and for the whole astra main sample the ``R`` rows went
        onto the wire in real tickers while every other block of the same
        conversation was labelled -- 537 of 537 quote turns, measured.  Nothing
        failed, because a leak is not a crash.
        """
        if self._pseudo is None:
            return observation
        masked = self._mask(observation.text, trade_date=trade_date)
        return replace(
            observation,
            text=masked,
            blocks={
                name: self._mask(block, trade_date=trade_date)
                for name, block in observation.blocks.items()
            },
            token_estimate=estimate_tokens(masked),
        )

    def quote_observation(self, results: Sequence[Mapping[str, Any]], point: GridPoint) -> Observation:
        """The answer to a proposal turn: the prices, as the policy must see them.

        The runner used to call ``state_space.quote_block`` itself.  That reads
        as an innocent shortcut -- the quote block is pure rendering and the
        environment has nothing to add -- and it is exactly what put the real
        names on the wire: the de-identification is a property of the *channel*,
        not of the renderer, so anything that bypasses the environment bypasses
        it.
        """
        return self._masked(self.state_space.quote_block(results), trade_date=point.trade_date)

    # -- lifecycle -------------------------------------------------------

    @property
    def terminated(self) -> str | None:
        return self._terminated

    def state(self, *, as_of: datetime | None = None) -> BookState:
        """The handoff for the next episode."""
        return BookState.capture(
            self.book,
            as_of=as_of or self._last_stamp or datetime.now(tz=ZoneInfo("UTC")),
            resolver_versions=resolver_versions(),
            env_fingerprint=self.config.fingerprint(),
        )

    def load(self, carried_in: BookState) -> None:
        """Restore a previous episode's book.  NAV is asserted, not restated."""
        self.book = carried_in.restore()

    def reset(self, episode: Episode) -> StepView:
        """Begin an episode at its first grid point.

        The book is *not* reset.  That is the whole content of section 1.4: a
        fresh ``Book`` here would make each month an independent bet and the
        run's log return would no longer be the sum of the episodes'.
        """
        self.episode = episode
        self._index = 0
        self._pending = None
        self._pending_fills = []
        self._last_result = None
        self._terminated = None
        # Drawn here rather than once per run: the ruling of 2026-09-23 is that
        # a carried position crosses the boundary under the *new* episode's
        # label, so the draw has to change exactly where the context resets.
        # Nothing else moves -- the book still holds real tickers, and the
        # ``carried in:`` rows go through the same masking as any other row.
        if self.config.flags.anonymize:
            self._pseudo = Pseudonyms.draw(
                episode.episode_id, self.config.universe.observed
            )
        return self._advance()

    def step(self, action_text: str | None = None) -> StepView:
        """Apply an action to the pending grid point, then advance.

        ``action_text`` is ignored when the pending point is not on the decision
        grid, which is most of them: the mark grid is twice a day and the
        decision grid defaults to once.
        """
        point = self._pending
        if point is None:
            return StepView(point=None, observation=None, nav=self.book.nav, done=True,
                            terminated=self._terminated)

        results: tuple[Mapping[str, object], ...] = ()
        # Cleared unconditionally: a mark-only point that inherited the previous
        # decision's proposals would attribute them to a step the policy was
        # never asked about.
        self._proposals = ()
        if point.is_decision and action_text is not None:
            results = self._act(point, action_text)
        if point.is_hedge:
            self._run_hedge(point)
        self._remark(point)
        # Captured before ``_close_point`` drains them.  The returned view
        # describes the *next* pending point, but its ``fills`` and ``results``
        # belong to the point just applied: that is the only place the driver
        # can see what its own action did, and the decision ledger needs it.
        applied = tuple(self._pending_fills)
        self._close_point(point)
        return replace(self._advance(), fills=applied, results=results)

    # -- phases 1-6 ------------------------------------------------------

    def _advance(self) -> StepView:
        assert self.episode is not None, "call reset() before step()"
        if self._terminated is not None:
            self._pending = None
            return StepView(point=None, observation=None, nav=self.book.nav, done=True,
                            terminated=self._terminated)
        if self._index >= len(self.episode.points):
            self._pending = None
            return StepView(point=None, observation=None, nav=self.book.nav, done=True)

        point = self.episode.points[self._index]
        self._index += 1
        self._pending = point
        self._pending_fills = []

        self._settle_expiries(point)                     # 3
        accrual = self._accrue(point)                    # 4
        self._mark(point)                                # 5
        observation = self._observe(point) if point.is_decision else None  # 6

        ruin = self._check_ruin()
        if ruin is not None:
            self._terminated = ruin
            self._close_point(point, accrual=accrual)
            self._pending = None
            return StepView(point=point, observation=None, nav=self.book.nav, done=True,
                            terminated=ruin)

        self._accrual = accrual
        return StepView(
            point=point,
            observation=observation,
            nav=self.book.nav,
            done=False,
            info={"n_positions": len(self.book.positions), "cash": self.book.cash},
        )

    def _settle_expiries(self, point: GridPoint) -> None:
        """Settle every position whose last leg has reached expiry.

        The cash is never in doubt -- ``Book.close`` credits proceeds, charges
        the assignment fee and adds to ``realized_pnl`` -- but the *policy* was
        not told.  A package simply stopped appearing in the ``P`` block, and
        the only trace was an account row that had moved for no stated reason,
        which is the one event in the environment that looked like the position
        had never existed.  Stop-losses and DTE force-closes already report
        themselves this way; settlement is the third involuntary exit and it is
        reported in the same shape.

        ``force_close_dte`` normally closes a package a day before expiry, so
        this path is reachable only when the grid skips the date that check
        would have run on -- a missing chain partition, which is exactly the
        condition under which nobody is watching.
        """
        expiring = expired_positions(self.book, point.trade_date)
        if not expiring:
            return
        settled: list[Mapping[str, object]] = []
        for position in expiring:
            spot = self._spot(position.underlying, point)
            if spot <= 0:
                raise DataGap(
                    f"no underlying price for {position.underlying} on "
                    f"{point.trade_date.isoformat()}; cannot settle {position.position_id}"
                )
            fill = self.execution.settle_expiry(self.book, position, spot=spot)
            self._pending_fills.append(fill)
            settled.append(
                {"order": "-", "status": "EXPIRED", "position_id": fill.position_id}
            )
        # Appended to the previous action's results rather than replacing them:
        # ``_advance`` settles before it observes, so these rows join whatever
        # the policy's own orders did and both reach the same ``RES`` block.
        # ``_last_result`` is None at the first point of an episode, where a
        # carried-in position can expire before any action has been taken.
        if self._last_result is None:
            self._last_result = {"results": ()}
        self._last_result["results"] = (
            tuple(self._last_result.get("results", ())) + tuple(settled)
        )

    def _accrue(self, point: GridPoint) -> float:
        """Credit ``r_f`` on cash for the interval since the last grid point.

        The interval is calendar time, not trading time: cash earns over a
        weekend, and using trading-day fractions would under-credit the book by
        roughly the weekend share of the year against an index that does not
        have that headwind.
        """
        previous = self._last_stamp
        self._last_stamp = point.timestamp
        if previous is None:
            return 0.0
        years = (point.timestamp - previous).total_seconds() / (365.25 * 86_400.0)
        rate = self.rates.get(point.trade_date, self.config.risk_free_rate)
        return self.execution.accrue(self.book, years=years, rate=rate)

    def _mark(self, point: GridPoint) -> None:
        report = self.market.mark(
            self.book,
            trade_date=point.trade_date,
            session=point.session,
            decision_time=point.timestamp,
        )
        self._record_marks(report)
        self.book.apply_marks(report.marks)
        for underlying, spot in report.spots.items():
            if underlying in self.book.shares:
                self.book.set_share_mark(underlying, spot)
        self.book.mark_peak()

    def _record_marks(self, report: MarkReport) -> None:
        """Carry ``unbounded_positions`` out to the manifest.

        ``MarketResolver`` clamps a mark that falls outside the package's own
        structural value range and records which position it happened to, but
        the clamp is silent in the ledger: the only trace is a ``mark_quality``
        of ``stale``, which the same position would carry for merely being old.
        Dropped here, the field could only ever read empty to a consumer, which
        is indistinguishable from the case it exists to flag.

        Counted rather than raised for the reason the resolver gives for not
        raising -- a mid-episode exception destroys the ledger that would
        diagnose it -- and the ids are kept, not just the count, because the
        first question about a non-empty list is which package.
        """
        stats = self._mark_stats
        stats["points"] += 1
        if not report.unbounded_positions:
            return
        stats["unbounded_marks"] += len(report.unbounded_positions)
        known = stats["unbounded_positions"]
        for position_id in report.unbounded_positions:
            if position_id not in known:
                known.append(position_id)

    @property
    def mark_stats(self) -> dict[str, Any]:
        """Provenance for the manifest; see :meth:`_record_marks`."""
        return {
            **self._mark_stats,
            "unbounded_positions": list(self._mark_stats["unbounded_positions"]),
        }

    def _observe(self, point: GridPoint) -> Observation:
        assert self.episode is not None
        episode_ctx = EpisodeContext(
            episode_id=self.episode.episode_id,
            config=self.config,
            start_date=self.episode.start_date,
            end_date=self.episode.end_date,
            decision_points=self.episode.n_decisions,
            carried_in=self.book.view(point.timestamp) if self._index == 1 else None,
        )
        observation = self.state_space.step_block(
            StepContext(
                episode=episode_ctx,
                step_index=self._index - 1,
                trade_date=point.trade_date,
                session=point.session,
                decision_time=point.timestamp,
                book=self.book.view(point.timestamp),
                features=self.features,
                last_result=self._last_result,
                previous_trade_date=point.previous_trade_date,
            )
        )
        return self._masked(observation, trade_date=point.trade_date)

    def episode_header(self, episode: Episode, *, carried_in: bool) -> str:
        """The one-off block emitted before an episode's first step."""
        point = episode.points[0]
        header = self.state_space.episode_block(
            EpisodeContext(
                episode_id=episode.episode_id,
                config=self.config,
                start_date=episode.start_date,
                end_date=episode.end_date,
                decision_points=episode.n_decisions,
                carried_in=self.book.view(point.timestamp) if carried_in else None,
            )
        )
        # ``trade_date=None`` on purpose, so the header's ``start..end`` span
        # degrades to ``<d>..<d>`` rather than to a sawtooth.  A dte is a
        # meaningful answer to "when is this step"; it is not a meaningful
        # answer to "how long is this episode", and ``steps=`` already carries
        # the length.
        #
        # This runs before ``reset``, so the draw has to be made here when the
        # caller asks for the header first.  Drawing it twice is safe -- the map
        # is a pure function of the episode id.
        if self.config.flags.anonymize:
            self._pseudo = Pseudonyms.draw(
                episode.episode_id, self.config.universe.observed
            )
        return self._mask(header, trade_date=None)

    # -- phase 7 ---------------------------------------------------------

    def _act(self, point: GridPoint, action_text: str) -> tuple[Mapping[str, object], ...]:
        # Every fill on this point resolves against ``fill_point``'s chain,
        # which is ``point`` itself except on an AM decision.  See
        # ``_fill_point``.
        fill_point = self._fill_point(point)
        # The inbound half of the de-identification, and it is the *first* thing
        # that happens to a completion.  Everything downstream -- the parser's
        # tradeable check, the contract resolver, the book, the ledger -- keeps
        # working in real tickers, which is what makes this a view rather than a
        # change to the environment.  Unmasking any later would mean a second
        # code path that has to know about labels.
        #
        # ``E_UNKNOWN_NAME`` is still reachable and still means what it says: a
        # label outside the draw passes through unmasked, so a hallucinated
        # ``U44`` is refused rather than silently mapped onto a real name.
        #
        # The reverse is refuse-and-count, and it has to happen *here*, on the
        # raw completion: after unmasking, a guessed ``NVDA`` and an unmasked
        # ``U03`` are identical text, so a real name that reached the resolver
        # would fill and be indistinguishable in the ledger from a label-driven
        # order.  Refusing per line rather than per completion keeps the other,
        # properly masked orders on the same step tradeable.
        refusals: list[Mapping[str, object]] = []
        if self._pseudo is not None:
            action_text, named = strip_real_name_orders(
                action_text, self.config.universe.observed
            )
            refusals = [
                {
                    "order": line,
                    "status": "E_REAL_NAME",
                    "detail": f"{ticker} is a real name; this episode labels it "
                    f"{self._pseudo.to_label.get(ticker, '?')}",
                }
                for line, ticker in named
            ]
            self._anon_stats["steps"] += 1
            self._anon_stats["real_name_orders_refused"] += len(named)
            self._anon_stats["real_names"] = sorted(
                set(self._anon_stats["real_names"]) | {t for _, t in named}
            )
            action_text = self._pseudo.unmask(action_text)
        parsed = parse_action(
            action_text, self.config, open_positions=self._roll_bases()
        )
        results: list[Mapping[str, object]] = [
            *refusals,
            *(
                {"order": error.raw, "status": error.code, "detail": error.detail}
                for error in parsed.errors
            ),
        ]

        # Closes run before opens.  A policy that rotates out of one position
        # into another has to be able to do it in one step, and running opens
        # first would size the new position against buying power the close is
        # about to release.
        for close in parsed.closes:
            position = self.book.positions.get(close.position_id)
            if position is None:
                results.append(
                    {"order": close.raw, "status": "E_UNKNOWN_POS", "detail": close.position_id}
                )
                continue
            fill = self.execution.close(
                self.book,
                position,
                reason="policy",
                quotes=self._quotes(position.underlying, fill_point),
            )
            self._pending_fills.append(fill)
            results.append(
                {
                    "order": close.raw,
                    "status": "OK",
                    "position_id": fill.position_id,
                    "realized": round(fill.realized, 2),
                    # Reported on both sides of the round trip, because half of
                    # it is charged here and the policy has no other way to
                    # learn what closing costs.
                    "cost": round(fill.cost, 2),
                }
            )

        # A roll is two fills, and they are split across the two phases rather
        # than run back to back, so that *every* close on this step has released
        # its collateral before *any* open is sized.  Running each roll as a unit
        # would size the first roll's replacement against a book that still holds
        # the second roll's position, which makes the outcome depend on the order
        # the policy happened to write its lines in.
        rolled = [self._roll_close(roll, fill_point) for roll in parsed.rolls]
        for roll, closed, row in rolled:
            results.append(self._roll_open(roll, closed, row, fill_point))

        for order in parsed.opens:
            results.append(self._open(order, fill_point))

        # Risk controls run after the policy, so a stop-loss the policy chose
        # not to take is still taken.
        for position, reason in breached_positions(self.book, self.config, point.trade_date):
            fill = self.execution.close(
                self.book,
                position,
                reason=reason,
                quotes=self._quotes(position.underlying, fill_point),
            )
            self._pending_fills.append(fill)
            results.append(
                {"order": "-", "status": reason.upper(), "position_id": fill.position_id}
            )

        self._last_result = {"results": results, "rationale": parsed.rationale}
        self._record_proposals(point, parsed, results, refused=refusals)
        return tuple(results)

    def _record_proposals(
        self,
        point: GridPoint,
        parsed: ParsedAction,
        results: Sequence[Mapping[str, object]],
        *,
        refused: Sequence[Mapping[str, object]] = (),
    ) -> None:
        """Every strategy the policy named this step, filled or not.

        The ledger's ``fills`` table answers "what did the book do"; this
        answers "what did the policy *want*", and the two differ on most steps.
        A proposal can die at the parser, at the chain, at the resolver or at
        the size gate, and in every one of those cases it leaves no trace in
        any table that records positions -- so the arms that ask *why* a policy
        underperformed (was the intent wrong, or was the intent fine and the
        ladder refused it?) currently have to re-parse completion text and
        re-derive the disposition, which means re-implementing ``_act``.

        Kept per step and drained by the runner rather than written here: the
        environment does not own a ledger, and giving it one would make the
        write path depend on whether an arm asked for ledgers at all.

        Pairing is positional, not by order text.  ``_act`` builds ``results``
        as one row per parse error, then one per close, then one per **roll**,
        then one per open, in those orders, so the open results are a contiguous
        slice and the i-th open's outcome is the i-th element of it.  A roll
        contributes exactly one result row even though it is two fills, which is
        what keeps that slice arithmetic true.  Matching on ``raw`` instead
        looks equivalent and is not: a policy may emit the same line twice --
        the first live step of the SFT window emitted ``O TSLA ol b 8_30``
        three times -- and a by-text lookup gives all the duplicates the first
        one's fill, so the ledger reports three opened positions where the book
        has one.  There is no key on ``OpenOrder`` that separates them, because
        as *intents* they are genuinely identical; only their position in the
        sequence distinguishes them.
        """
        rows: list[dict[str, Any]] = []
        # ``refused`` never reached ``parse_action`` -- it was stripped from the
        # completion before unmasking, because after unmasking a guessed real
        # name is indistinguishable from an unmasked label.  It therefore has to
        # be counted into the offsets by hand, and it is passed in rather than
        # recovered from ``results`` so the arithmetic below states its own
        # premise instead of inferring it from a status string.
        for refusal in refused:
            rows.append({"kind": "reject", **refusal})
        for error in parsed.errors:
            rows.append(
                {
                    "kind": "reject",
                    "order": error.raw,
                    "status": error.code,
                    "detail": error.detail,
                }
            )
        for close in parsed.closes:
            rows.append({"kind": "close", "order": close.raw, "position_id": close.position_id})

        first_roll = len(refused) + len(parsed.errors) + len(parsed.closes)
        roll_results = list(results[first_roll : first_roll + len(parsed.rolls)])
        for index, roll in enumerate(parsed.rolls):
            result = roll_results[index] if index < len(roll_results) else {}
            rows.append(
                {
                    "kind": "roll",
                    "order": roll.raw,
                    "status": result.get("status", "E_UNRECORDED"),
                    "detail": result.get("detail", ""),
                    "underlying": roll.underlying,
                    "family": roll.family,
                    "orientation": roll.orientation,
                    "tenor_bucket": roll.tenor_bucket,
                    "coordinates": dict(roll.coordinates),
                    "strategy_handle": roll.strategy_handle,
                    # For a roll, ``defaulted`` means "inherited from the position
                    # I replaced", which is still not a coordinate the policy
                    # chose *this step*.  Same column, same meaning.
                    "defaulted": list(roll.defaulted),
                    "snapped": list(roll.snapped),
                    # Both halves of the link, so the proposals table answers
                    # "which two positions did this action join" without joining
                    # to ``fills``.  ``closed`` is filled in even when the
                    # replacement was refused — that is the row that says the
                    # book lost a position and gained nothing.
                    "closed_position_id": result.get("closed", roll.position_id),
                    "position_id": result.get("position_id", ""),
                    "quantity": result.get("quantity"),
                    "debit": result.get("debit"),
                    "cost": result.get("cost"),
                    "dte": result.get("dte"),
                    "binding": result.get("binding", ""),
                }
            )

        first_open = first_roll + len(parsed.rolls)
        open_results = list(results[first_open : first_open + len(parsed.opens)])
        for index, order in enumerate(parsed.opens):
            result = open_results[index] if index < len(open_results) else {}
            rows.append(
                {
                    "kind": "open",
                    "order": order.raw,
                    "status": result.get("status", "E_UNRECORDED"),
                    "detail": result.get("detail", ""),
                    "underlying": order.underlying,
                    "family": order.family,
                    "orientation": order.orientation,
                    "tenor_bucket": order.tenor_bucket,
                    "coordinates": dict(order.coordinates),
                    "strategy_handle": order.strategy_handle,
                    # What the policy left to the environment.  A coordinate it
                    # never named is not a coordinate it chose, and an offline
                    # reader that cannot tell the two apart will credit the
                    # policy with the default.
                    "defaulted": list(order.defaulted),
                    "snapped": list(order.snapped),
                    "position_id": result.get("position_id", ""),
                    "quantity": result.get("quantity"),
                    "debit": result.get("debit"),
                    "cost": result.get("cost"),
                    "dte": result.get("dte"),
                    "binding": result.get("binding", ""),
                }
            )

        # Restore the order the policy wrote, which the grouping above destroys.
        # It matters because the rejects are usually the *tail* of a completion
        # -- the order cap bites after eight lines -- and grouping them first
        # makes the ledger say the policy's first five choices were refused
        # when they were its last five.  An offline reader ranking a policy's
        # intent by where it placed it would invert the ranking.
        #
        # Recovered by walking the completion's own lines rather than tracked
        # through the parse, so that the index means "this line of what the
        # model emitted" even for lines the parser rejected before building an
        # order.  Duplicate lines are consumed in turn, so three identical
        # opens get three consecutive indices.  A line that does not match --
        # prose, or a raw the parser normalized -- sorts to the end rather than
        # claiming position 0.
        offsets: dict[str, list[int]] = {}
        for offset, line in enumerate(parsed.raw.splitlines()):
            stripped = line.strip()
            if stripped:
                offsets.setdefault(stripped, []).append(offset)

        unmatched = len(parsed.raw.splitlines()) + 1
        keyed: list[tuple[int, int, dict[str, Any]]] = []
        for position, row in enumerate(rows):
            queue = offsets.get(str(row["order"]).strip())
            offset = queue.pop(0) if queue else unmatched
            # ``position`` breaks ties so the sort stays total and the unmatched
            # tail keeps the grouped order instead of depending on sort stability.
            keyed.append((offset, position, row))
        ordered = [row for _, _, row in sorted(keyed, key=lambda item: item[:2])]
        for index, row in enumerate(ordered):
            row.update(
                {
                    "step_ts": point.timestamp.isoformat(),
                    "trade_date": point.trade_date.isoformat(),
                    "session": point.session,
                    "proposal_index": index,
                }
            )
        self._proposals = tuple(ordered)

    @property
    def proposals(self) -> tuple[Mapping[str, Any], ...]:
        """The strategies named at the most recent decision; see :meth:`_record_proposals`."""
        return self._proposals

    def _roll_bases(self) -> Mapping[str, RollBasis]:
        """What each open position would lend to a roll against it.

        Built from ``Book.positions`` so that "which ids are rollable" and "what
        rolling one would inherit" are one read of one structure.  ``coordinates``
        comes off the position rather than out of its ``strategy_handle``: the
        handle writes deltas as integer percent, so recovering 0.575 from ``d58``
        is not possible and a roll would quietly move a strike it was asked to
        leave alone.
        """
        return {
            position_id: RollBasis(
                underlying=position.underlying,
                family=position.family,
                orientation=position.orientation,
                coordinates=dict(position.coordinates),
            )
            for position_id, position in self.book.positions.items()
        }

    def _roll_close(
        self, roll: RollOrder, point: GridPoint
    ) -> tuple[RollOrder, Position | None, dict[str, object]]:
        """Unwind the position an ``X`` replaces, and start its result row.

        ``reason="roll"`` rather than ``"policy"`` on the fill: a roll's exit and
        a decided exit are the same cash but different intent, and an arm that
        reports "the policy closed 40 positions" without separating them is
        counting continuations as decisions to leave.
        """
        position = self.book.positions.get(roll.position_id)
        row: dict[str, object] = {"order": roll.raw, "closed": roll.position_id}
        if position is None:
            return roll, None, {**row, "status": "E_UNKNOWN_POS", "detail": roll.position_id}
        fill = self.execution.close(
            self.book,
            position,
            reason="roll",
            quotes=self._quotes(position.underlying, point),
        )
        self._pending_fills.append(fill)
        row["realized"] = round(fill.realized, 2)
        row["cost"] = round(fill.cost, 2)
        return roll, position, row

    def _roll_open(
        self,
        roll: RollOrder,
        closed: Position | None,
        row: Mapping[str, object],
        point: GridPoint,
    ) -> Mapping[str, object]:
        """Open the replacement half, and report the pair as one row.

        **A refused replacement leaves the close standing.**  There is no way to
        undo it: ``book.close`` has already realized the PnL and there is no
        transaction to roll back, and re-opening the original would be a second
        round trip at a second spread — the policy would pay twice to end up where
        it started.  So the row reports the replacement's own rejection code and
        keeps ``closed``, and the ``RES`` line reads ``E_SIZE ... p03-closed``.
        That distinction has to reach the policy: "I rolled p03" and "p03 is gone
        and nothing replaced it" are different books, and a bare rejection would
        let the policy keep reasoning about a position it no longer holds.
        """
        if closed is None:
            return row
        result = self._open(roll.as_open(), point, replaces=closed)
        merged = {**row, **result, "closed": roll.position_id}
        # ``_open`` reports the *entry* cost; the close charged one too, and the
        # policy has no other way to learn that a roll pays two spreads.
        merged["cost"] = round(float(row.get("cost") or 0.0) + float(result.get("cost") or 0.0), 2)
        if result.get("status") != "OK":
            merged["detail"] = f"{result.get('detail', '')} ({roll.position_id} closed)".strip()
        return merged

    #: The dataset and field the spot shock's per-name reference vol comes from.
    #: The same field ``statespace_v1``'s market block already shows the policy,
    #: deliberately: the policy can then reason about its own size from its own
    #: state, and there is only one number to audit rather than two that could
    #: disagree.
    SPOT_SIGMA_DATASET = "option_delta_point_surface"
    SPOT_SIGMA_FIELD = "atm_iv_30d"

    def _spot_sigma(self, underlying: str, point: GridPoint) -> float | None:
        """The 30-day ATM implied vol for ``underlying`` (``env_contract`` 6A.1.1).

        Read at the *fill* point, which is what ``_open`` is given, so the shock
        and the quotes the package was priced against come from the same instant.

        Fetched for the whole observed universe rather than for the one name
        being sized.  That looks wasteful and is the opposite: ``FeatureSource``
        caches by ``(dataset, date, session, keys)``, so asking for the same key
        tuple the state block asks for turns every open after the first into a
        cache hit, while a single-name fetch would be a distinct key and a second
        cold partition read at /ocean's ~8MB/s.

        ``None`` on any miss -- absent partition, absent name, absent field, or a
        value that fails the point-in-time gate.  The caller falls back to the
        package's own vol and records the substitution, so a miss degrades the
        measure rather than the run.
        """
        try:
            records = self.features.fetch(
                self.SPOT_SIGMA_DATASET,
                trade_date=point.trade_date,
                session=point.session,
                keys=tuple(self.config.universe.observed),
                decision_time=point.timestamp,
            )
        except (FileNotFoundError, KeyError):
            return None
        record = records.get(underlying)
        if record is None:
            return None
        assert_point_in_time(record, point.timestamp)
        value = record.values.get(self.SPOT_SIGMA_FIELD)
        if value is None:
            return None
        sigma = float(value)
        return sigma if sigma > 0.0 else None

    def _resolve_and_size(
        self, order, point: GridPoint
    ) -> tuple[ResolvedPackage, SizeDecision] | Mapping[str, object]:
        """Everything an open does *before* the book is touched.

        Split out on 2026-09-24 so the ``Q`` quote path and :meth:`_open` walk
        the identical chain slice, contract resolver and size resolver.  A quote
        that resolved separately could show a package, a quantity or a rejection
        that the subsequent fill disagreed with, and the divergence would be
        invisible: the ledger only ever records the fill.  Sharing the code is
        what makes ``quoted == filled`` a property of the design rather than
        something a test has to keep re-establishing.

        Returns the resolved pair, or the result row explaining the refusal.
        """
        try:
            chain_slice = self.chain.slice_for(
                order.underlying,
                trade_date=point.trade_date,
                session=point.session,
                decision_time=point.timestamp,
            )
        except NoChainData as exc:
            return {"order": order.raw, "status": "E_NO_CHAIN", "detail": str(exc)}

        package = self.contract.resolve(order, chain_slice)
        if isinstance(package, ResolutionFailure):
            return package.as_result()

        decision = self.size.resolve(
            package,
            self.book,
            as_of=point.timestamp,
            atm_iv=self._spot_sigma(order.underlying, point),
        )
        if not decision.approved:
            return {
                "order": order.raw,
                "status": decision.code,
                "detail": decision.detail,
                "binding": decision.binding,
            }
        return package, decision

    def _quote(self, quote, point: GridPoint) -> Mapping[str, object]:
        """Price one candidate without opening it.

        Deliberately reports the *same* keys an ``OK`` open reports, minus the
        ``position_id`` there is no position for, so the policy reads one
        vocabulary in both turns.  ``quantity`` is included because it is not
        the policy's to choose -- the size resolver picks it -- and premium and
        cost are meaningless without the multiplier they were computed at.

        Every refusal an open can give, a quote gives too, for the same reason
        the parser is shared: a candidate that cannot be opened must not come
        back looking priceable.
        """
        resolved = self._resolve_and_size(quote.as_open(), point)
        if not isinstance(resolved, tuple):
            return resolved
        package, decision = resolved
        priced = self.execution.price(package, decision.quantity)
        return {
            "order": quote.raw,
            "status": "OK",
            "quantity": decision.quantity,
            "binding": decision.binding,
            # ``premium`` is the mid and ``cost`` the friction on top; ``debit``
            # is what leaves the account, and equals ``-cash_delta`` on the fill
            # this previews.  All three are reported because the policy has been
            # shown to reason about the ratio, not the level.
            "premium": round(-priced.mid_value, 2),
            "debit": round(-priced.cash_delta, 2),
            "cost": round(priced.cost, 2),
            "dte": package.dte,
            "leg_deltas": [leg.quote.delta for leg in package.legs],
        }

    def quote(self, action_text: str, point: GridPoint) -> tuple[Mapping[str, object], ...]:
        """Answer a proposal turn.  Changes nothing.

        The book, the cash, the ledger and the grid are all untouched: this runs
        between ``policy.act`` and ``env.step``, on the decision point that has
        not been stepped yet, so every quote prices against exactly the chain
        slice the fill will use one turn later.

        Empty when the completion asked for nothing, and the caller must treat
        that as "this was not a quote turn" and hand the same text to ``step``.
        The check is first because everything below it has a cost the ordinary
        path must not pay: the ``E_QUOTE_TURN`` refusals would reject a normal
        completion's orders outright, and the anonymization counters would tick
        twice for a completion that ``_act`` is about to count again.
        """
        if not (self.config.quotes_enabled and has_quote_lines(action_text)):
            return ()
        refusals: list[Mapping[str, object]] = []
        if self._pseudo is not None:
            # The same inbound guard ``_act`` applies, and for the same reason:
            # a ``Q NVDA ...`` is a real-name leak whether or not it trades.
            action_text, named = strip_real_name_orders(
                action_text, self.config.universe.observed
            )
            refusals = [
                {
                    "order": line,
                    "status": "E_REAL_NAME",
                    "detail": f"{ticker} is a real name; this episode labels it "
                    f"{self._pseudo.to_label.get(ticker, '?')}",
                }
                for line, ticker in named
            ]
            self._anon_stats["real_name_orders_refused"] += len(named)
            self._anon_stats["real_names"] = sorted(
                set(self._anon_stats["real_names"]) | {t for _, t in named}
            )
            action_text = self._pseudo.unmask(action_text)
        parsed = parse_action(action_text, self.config, open_positions=self._roll_bases())
        results: list[Mapping[str, object]] = [
            *refusals,
            *(
                {"order": error.raw, "status": error.code, "detail": error.detail}
                for error in parsed.errors
            ),
        ]
        # Orders written alongside the quotes were written before their answers
        # existed.  Refusing them is the whole point of the two-turn shape; see
        # ``E_QUOTE_TURN``.
        results.extend(
            {
                "order": order.raw,
                "status": "E_QUOTE_TURN",
                "detail": "this turn asked for quotes; send orders on the next turn",
            }
            for order in parsed.orders
            if not isinstance(order, (QuoteOrder, HoldOrder))
        )
        results.extend(self._quote(quote, point) for quote in parsed.quotes)
        return tuple(results)

    def _open(
        self, order, point: GridPoint, *, replaces: Position | None = None
    ) -> Mapping[str, object]:
        resolved = self._resolve_and_size(order, point)
        if not isinstance(resolved, tuple):
            return resolved
        package, decision = resolved

        position, fill = self.execution.open(
            self.book, package, decision, as_of=point.timestamp, replaces=replaces
        )
        self._pending_fills.append(fill)
        return {
            "order": order.raw,
            "status": "OK",
            "position_id": position.position_id,
            "quantity": decision.quantity,
            "binding": decision.binding,
            # ``premium`` is echoed on the fill as well as on the quote, and is
            # read off the ``Fill`` rather than recomputed.  It is what makes
            # "quoted == filled" checkable *by the policy*: the Q answer and the
            # R row it produced carry the same three cells, so a divergence
            # between preview and execution shows up in the prompt instead of
            # only in a test nobody reruns.
            "premium": round(-fill.mid_value, 2),
            "debit": round(-fill.cash_delta, 2),
            "cost": round(fill.cost, 2),
            # What the "nearest" rule actually resolved to.  The order names a
            # tenor bucket and deltas; the ladder supplies the closest listed
            # expiry and strikes, and the gap between the two is invisible
            # unless it is reported here.  Taken off the package rather than
            # recomputed so the echo cannot disagree with what was filled.
            "dte": package.dte,
            "leg_deltas": [leg.quote.delta for leg in package.legs],
        }

    # -- phases 8-11 -----------------------------------------------------

    def _run_hedge(self, point: GridPoint) -> None:
        # Hedge shares transact at the fill point's spot for the same reason
        # option orders do: at AM the chain's ``underlying_price`` is still the
        # previous close, so hedging there would trade the overnight gap.
        fill_point = self._fill_point(point)
        # Held shares are priced too, not only the underlyings of live positions:
        # a share balance whose option leg has died still needs a spot, or the
        # resolver cannot unwind it and it becomes permanent naked stock.
        #
        # Proxy tickers are asked for by name rather than inferred, because on
        # the step that opens an index position the proxy is neither an
        # underlying nor yet a share balance -- it appears in neither set, and a
        # hedge priced at 0.0 reports ``no_price`` and silently never happens.
        wanted = {p.underlying for p in self.book.positions.values()} | set(self.book.shares)
        wanted |= {
            ticker
            for underlying in tuple(wanted)
            if (ticker := self.config.hedge.hedge_ticker_for(underlying)) is not None
        }
        spots = {underlying: self._spot(underlying, fill_point) for underlying in wanted}
        plan = self.hedge.hedge(
            self.book,
            spots={k: v for k, v in spots.items() if v > 0},
            as_of=fill_point.timestamp,
            spreads=self.spreads,
            # The *fill* point's session, not the decision point's.  At AM the
            # decision is taken at 09:30 but ``_fill_point`` may move the fill
            # to the close, and the spread that gets paid is the one at the
            # instant the shares actually trade.
            session=fill_point.session,
        )
        self._record_hedge(plan)
        for order in plan.orders:
            self._pending_fills.append(self.execution.hedge(self.book, order))

    def _record_hedge(self, plan: HedgePlan) -> None:
        """Count what the hedge grid did, including when it did nothing.

        Every field here answers a question that a zero-order run cannot
        otherwise answer: ``points`` says the grid ran at all, ``examined`` says
        positions reached the band rule, ``peak_exposure_ratio`` says how close
        they came to tripping it, and ``skipped`` says which ones never got a
        price.  Without these an unexercised hedge and a broken one are the same
        empty ``fills.jsonl``.
        """
        stats = self._hedge_stats
        stats["points"] += 1
        stats["examined"] += len(plan.examined)
        stats["orders"] += len(plan.orders)
        # Under Whalley-Wilmott the band is a different number every group and
        # every step, so a single configured ``delta_band`` no longer describes
        # what the run did.  The observed range is what tells a sweep whether
        # ``risk_aversion`` moved the band at all or whether the clamps ate the
        # whole effect -- a min and max both pinned to the clamps means the arm
        # was the fixed rule wearing a different name.
        for _, band_dollars, _, _ in plan.bands:
            stats["band_min"] = min(stats["band_min"], band_dollars)
            stats["band_max"] = max(stats["band_max"], band_dollars)
        stats["peak_exposure_ratio"] = max(
            stats["peak_exposure_ratio"], plan.peak_exposure_ratio
        )
        for reason in plan.skipped.values():
            stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1

    @property
    def hedge_stats(self) -> dict[str, Any]:
        """Provenance for the manifest; see :meth:`_record_hedge`.

        ``spreads`` is carried alongside because a Whalley-Wilmott arm whose
        spread table missed every lookup silently ran the flat fallback for
        every name, and the band rule in the config would still say
        ``whalley_wilmott``.  The per-name fallback counts are what make that
        detectable from the manifest instead of from a re-run.
        """
        stats = {**self._hedge_stats, "skipped": dict(self._hedge_stats["skipped"])}
        # ``inf`` survives ``json.dumps`` as the literal ``Infinity``, which is
        # not JSON and which every reader of the manifest then fails on.  The
        # sentinel only survives when no group was ever banded, so ``None`` says
        # that directly instead of encoding it as an unparseable number.
        if stats["band_min"] == float("inf"):
            stats["band_min"] = None
        stats["band_rule"] = self.config.hedge.band_rule
        if self.spreads is not None:
            stats["spreads"] = self.spreads.coverage
        return stats

    def _remark(self, point: GridPoint) -> None:
        """Step 9: re-mark at the same snapshot, so entry MTM is exactly zero."""
        self._mark(point)

    def _close_point(self, point: GridPoint, *, accrual: float | None = None) -> None:
        if self.ledger is not None and self.episode is not None:
            self.ledger.write_step(
                self.book,
                episode_id=self.episode.episode_id,
                step_ts=point.timestamp,
                session=point.session,
                is_decision_point=point.is_decision,
                fills=tuple(self._pending_fills),
                rf_accrual=self._accrual if accrual is None else accrual,
                terminated=self._terminated,
            )
        self._pending_fills = []

    def _check_ruin(self) -> str | None:
        """Section 11.2.  Absorbing, and checked before the policy may act.

        Checking after the action would let a book below the floor place one
        more trade, which is a state the run is meant to have terminated in.
        """
        floor = self.config.risk.ruin_floor_kappa * self.book.initial_cash
        if self.book.nav <= floor:
            return "ruin"
        return None

    # -- chain access ----------------------------------------------------

    def _slice(self, underlying: str, point: GridPoint) -> ChainSlice | None:
        key = (point.trade_date, point.session, underlying)
        if key not in self._slices:
            try:
                self._slices[key] = self.chain.slice_for(
                    underlying,
                    trade_date=point.trade_date,
                    session=point.session,
                    decision_time=point.timestamp,
                )
            except NoChainData:
                self._slices[key] = None
        return self._slices[key]

    def _fill_point(self, point: GridPoint) -> GridPoint:
        """The grid point whose chain an order placed at ``point`` fills against.

        Identity except on an AM point under ``GridSpec.am_fills_at_close``,
        where it is that same date's close.  The AM chain partition is a copy of
        the previous close while the AM *features* already carry today's open,
        so filling on the AM chain would let the policy trade the overnight gap
        at pre-gap prices.  See ``GridSpec.am_fills_at_close``.
        """
        if point.session != "AM" or not self.config.grid.am_fills_at_close:
            return point
        return replace(
            point,
            session="PM",
            timestamp=datetime.combine(
                point.trade_date,
                self.config.grid.pm_time_et,
                tzinfo=point.timestamp.tzinfo,
            ),
        )

    def _quotes(self, underlying: str, point: GridPoint):
        chain_slice = self._slice(underlying, point)
        return {q.contract_id: q for q in chain_slice.quotes} if chain_slice else None

    def _spot(self, underlying: str, point: GridPoint) -> float:
        """Underlying price at this grid point, or ``0.0`` if unobservable.

        Read off ``ChainSlice.underlying_price``, which the chain takes from the
        raw rows rather than from an admitted quote.  Taking it from
        ``quotes[0]`` made the spot conditional on some *option* being
        tradeable, and the AM slice admits nothing on any date -- no contract
        has printed yet at 09:30, so every AM quote's last print is from the
        previous session and fails the 900s age gate.  ``_settle_expiries``
        fires at the AM point of the expiry date, so that combination raised
        ``DataGap`` on every expiry that ever reached settlement.

        Falls back to the previous covered chain date, which is the one use
        ``OptionChain.previous_covered`` exists for: settling an expiry against
        yesterday's spot is a day of slippage, whereas *sourcing a decision
        quote* from yesterday is the stale-fill leak the chain module was
        written to prevent.  The lookup steps strictly back -- the grid is built
        from the chain's own coverage, so ``previous_covered(trade_date)`` is
        ``trade_date`` at every grid point and the fallback never ran.

        Callers that cannot proceed without a price raise ``DataGap``; the hedge
        resolver reports ``no_price`` and skips.
        """
        chain_slice = self._slice(underlying, point)
        if chain_slice is not None and chain_slice.underlying_price > 0:
            return chain_slice.underlying_price
        previous = self.chain.previous_covered(point.trade_date - timedelta(days=1))
        if previous is None:
            return 0.0
        try:
            fallback = self.chain.slice_for(
                underlying,
                trade_date=previous,
                session=point.session,
                decision_time=point.timestamp,
            )
        except NoChainData:
            return 0.0
        return fallback.underlying_price
