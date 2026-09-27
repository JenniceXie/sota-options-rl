"""Configuration for the options trading environment.

Every number that ``docs/env_contract.md`` marks as an open parameter lives
here, so that a change to a bound is a change to one config object rather than
a change to code.  Nothing downstream reads a literal.

The config is versioned and hashable: ``EnvConfig.fingerprint()`` goes into the
run manifest, and two runs with the same fingerprint are the same experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import time
from types import UnionType
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

CONFIG_VERSION = "env_config.v1"


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    """Which names the policy observes and which it may trade.

    ``observed`` may be a superset of ``tradeable``; with these defaults it is
    not.  The tenth slot carries the index exposure, and it is held by **SPY**
    rather than SPX.  Both were measured; the ETF wins on three counts and loses
    on none that bind.

    *Size.*  ``docs/env_contract.md`` section 2 wanted the index held out
    entirely, on the grounds that one cash-secured put would lock most of the
    book (~$650k of collateral against a $1M NAV).  The figure is right and the
    conclusion was not, because ``SizeResolver`` never reached the collateral
    gate: the ``max_loss`` limit refused any package whose max loss exceeded
    ``cap_for(family) * nav``, so the structure was rejected at size rather than
    filled and left to dominate.

    > **That rebuttal EXPIRED 2026-09-22** with the ceiling (see ``SIZE_RULES``).
    > The refusal it relied on no longer exists, and the original objection is
    > live again: under ``nav_fraction`` a deep index put is cheap in *premium*
    > and enormous in *max loss*, so ``f`` sizes it generously and only ``cash``
    > stops it -- at roughly ``NAV / (strike * 100 * 1.05)`` packages, which is
    > most of the book rather than zero packages.  The tradeability argument
    > below still holds because it is about the *smallest* package; the
    > concentration argument above does not.  Not re-litigated here: the index
    > slot stays SPY, and what changed is which limit is load-bearing.

    What actually decides tradeability is the *smallest* package.  A delta-specified width is ``spot * sigma * sqrt(T)``
    wide, so the index's low vol cancels most of its high spot -- SPX's minimum
    defined-risk package came to ~4.7x META's, not ~600x, and SPX was admitted on
    that measurement.  But "admitted" was marginal: at $1M it sized to 2 credit
    verticals and its iron butterfly was refused outright at 0 packages.

    Re-measured across the ten after the swap (``scripts/analysis/
    size_minimum_package.py``; median spot and ATM IV over the 495-date window,
    14 DTE, 100x multiplier).  Max loss of the smallest package, and packages
    approved at a $1M NAV:

    =====  =======  ======  ==========  ==========  ==========  ==========
    name     spot      iv    cv  loss    cv  @$1M    ib  loss    ib  @$1M
    =====  =======  ======  ==========  ==========  ==========  ==========
    PLTR      31.8   0.626        $106          93        $254          39
    MU        97.6   0.478        $252          39        $600          16
    GOOGL    164.4   0.296        $267          37        $631          15
    AMZN     186.1   0.302        $309          32        $730          13
    AAPL     207.6   0.257        $295          33        $695          14
    NVDA     146.8   0.471        $374          26        $888          11
    SPY      551.5   0.142        $436          22      $1,022           9
    MSFT     416.1   0.220        $508          19      $1,194           8
    TSLA     248.2   0.610        $811          12      $1,936           5
    META     521.1   0.319        $912          10      $2,152           4
    =====  =======  ======  ==========  ==========  ==========  ==========

    The ordering is the argument.  SPY has the *highest spot of the ten* and is
    still only seventh-largest by minimum package, because it also has much the
    lowest IV -- 0.142 against a 0.30-ish single-name median -- and a
    delta-anchored width scales with ``spot * sigma * sqrt(T)``, so the low vol
    does not merely cancel the high spot, it overturns it.  Concretely: SPY's
    minimum credit vertical is *cheaper* than META's and TSLA's.  The index is
    therefore not the constrained name in this universe; META and TSLA are, at 10
    and 12 packages against SPY's 22.  Nothing is pinned to the floor and no
    family is refused for any name.

    *Execution.*  SPY is the tighter market: median leg spread at 14 DTE and
    |delta| 0.30 is 0.57% against SPX's 1.16%.  (XSP, the Mini-SPX, was measured
    alongside and rejected at 3.74% -- a tenth the notional at six times the
    spread, and unhedgeable besides.)

    *Hedging.*  This is the decisive one.  SPX has no share to trade, so its
    delta could only ever be bounded by ``max_net_dollar_delta`` rather than
    actually offset, and the delta-balanced families had to be preferred on it
    as a workaround.  SPY hedges itself in its own shares like the nine single
    names, so the tenth name stops being a special case in ``HedgeResolver``.

    What is given up is real and worth naming.  SPY is American-style, so it can
    be assigned early where SPX could not; against that, SPX terminated against
    SET, a settlement print no one can trade, so the discontinuity moved rather
    than appeared.  Cboe publishes no SPY strategy indices, so the ``BXM``/
    ``PUT``/``WPUT``/``PPUT`` benchmark family in ``eval/arms.py`` no longer has
    a matching tradeable book -- those arms remain valid as external comparators
    because SPY tracks the index, but they are no longer same-instrument.

    SPX is *not* retained as an eleventh observed name.  Every symbol in this
    tuple is a constituent of the advance/decline breadth (the ``w`` field), and
    SPX and SPY are the same risk measured twice, so carrying both would let one
    index cast two of eleven votes.

    Cash-secured puts are the casualty of book size, not of the index: at a $1M
    NAV the 1% risk unit refuses one on **eight of the ten names**, admitting
    only MU (1 package) and PLTR (3).  SPY is among the eight, but so are AAPL,
    AMZN, GOOGL, META, MSFT, NVDA and TSLA -- the structure needs full notional
    on any name, so the refusal tracks share price, not index-ness.  It takes a
    ~$5.5M book before SPY's CSP clears.  That is ``⟨Q20⟩``, not ``⟨Q6⟩``.

    The nine single names are exactly those with a full ``*DI``/``*CW``/``*IO``
    triple in ``data/work/cboe_strategy_indices`` (2020-01-02 .. 2026-09-03, PLTR
    from 2020-10-08), so every one of them has a published buy-write and
    put-write index to be benchmarked against.
    """

    observed: tuple[str, ...] = (
        "AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA", "SPY",
    )
    tradeable: tuple[str, ...] = (
        "AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA", "SPY",
    )
    context_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("GOOGL", ("GOOGL", "GOOG")),
        ("SPY", ("SPY", "VOO", "IVV", "GSPC", "ES")),
    )

    def context_symbols(self, name: str) -> tuple[str, ...]:
        """Symbols in ``context_documents.related_symbols`` that mean ``name``.

        News is tagged by *issuer*, but the universe names an *instrument*, so a
        literal match under-counts two of the ten.  Measured over the 92-date
        corpus (2025-05-30 .. 2025-08-29, 106,520 documents;
        ``scripts/analysis/scan_universe_context_coverage.py``):

        *SPY.*  The corpus contains no ``SPX`` tag at all -- the S&P 500 index is
        tagged ``GSPC``.  SPY is an ETF, so it files nothing and generates no
        issuer news; a literal match finds 49 documents on 41 of 92 dates, the
        worst in the universe.  Adding the other S&P trackers and the index
        itself lifts it to 179 documents on **72 of 92 dates**.  QQQ, DIA and IWM
        are deliberately excluded -- they track NDX, the Dow and the Russell
        2000, not the S&P -- as is SPYD, a high-dividend screen rather than the
        index.

        *GOOGL.*  Alphabet's two share classes are both tagged, but on the *same*
        documents rather than on disjoint sets: the union is 580 documents
        against 576 for ``GOOGL`` alone.  Four documents, not a doubling.  The
        entry is kept because it costs nothing and the near-total overlap is a
        property of this provider that a future one need not share.

        No other name has an alias.  ``MUX``, ``MUFG``, ``MUSA``, ``MUR`` and
        ``MULN`` are different companies, not Micron, and there is no ``FB`` tag.
        MU's 73 documents on 41 dates are a real coverage gap, not an artifact.

        The table is reference data about how this provider tags documents, not
        an assertion about the universe, so entries for names that are not
        observed are simply unused rather than an error -- a narrowed universe
        must stay constructible against the default table.

        There is no untagged-macro pool to fall back on: every one of the 106,520
        documents carries at least one symbol, so index-level news reaches SPY
        only when the provider tags a proxy.  Widening that is an ingestion
        change, not a filter change.
        """
        for target, symbols in self.context_aliases:
            if target == name:
                return symbols
        return (name,)

    def __post_init__(self) -> None:
        if not self.observed:
            raise ConfigError("observed universe is empty")
        extra = set(self.tradeable) - set(self.observed)
        if extra:
            raise ConfigError(f"tradeable names not observed: {sorted(extra)}")
        if len(set(self.observed)) != len(self.observed):
            raise ConfigError("observed universe contains duplicates")
        targets = [target for target, _ in self.context_aliases]
        if len(set(targets)) != len(targets):
            raise ConfigError("context aliases contain duplicate targets")
        claimed: dict[str, str] = {}
        for target, symbols in self.context_aliases:
            if target not in symbols:
                raise ConfigError(f"context alias for {target} omits {target}")
            for symbol in symbols:
                if symbol in claimed:
                    raise ConfigError(
                        f"context symbol {symbol} claimed by "
                        f"{claimed[symbol]} and {target}"
                    )
                claimed[symbol] = target


@dataclass(frozen=True, slots=True)
class GridSpec:
    """The three grids of ``docs/env_contract.md`` section 1.2.

    The decision grid is configurable; the mark grid is not, so that changing
    when the policy may act cannot change the measured NAV path.

    ``decision_sessions`` defaults to ``("PM",)``.  That is not a preference:
    ``normalized/massive_option_quote_snapshots`` carries a single
    ``snapshot_time`` of 20:00 UTC, so the AM chain partition for date ``T`` is
    a *copy* of the close of ``T-1`` -- verified byte-for-byte on four
    consecutive date pairs of the rebuilt universe chain (15,775 / 15,820 /
    15,821 / 16,447 shared contracts, identical ``bid`` and ``ask``,
    ``max|dmid| = 0.0``).  See ``docs/state_space.md`` section 4.1, which
    reaches a similar conclusion from the feature side.

    Enabling AM decisions is therefore only sound together with
    ``am_fills_at_close``; see that field.
    """

    decision_sessions: tuple[str, ...] = ("PM",)
    mark_sessions: tuple[str, ...] = ("AM", "PM")
    hedge_sessions: tuple[str, ...] = ("PM",)
    #: Resolve orders placed on an AM point against that date's *PM* chain.
    #:
    #: Without this an AM decision is a look-ahead arbitrage rather than a
    #: trade.  ``underlying_market_features`` at AM on ``T`` carries ``T``'s
    #: open and ``spot_return_interval = 'previous_close_to_open'``, i.e. the
    #: realized overnight gap, while the AM chain is still priced off the close
    #: of ``T-1``.  A policy reading the one and filling against the other buys
    #: a move it has already been told about.  The gap is not small: mean
    #: ``|gap|`` over the 63 covered dates runs 0.43% (MSFT, AAPL) to 1.29%
    #: (TSLA), with a maximum of 10.9% (META), and option premia are a fraction
    #: of spot, so the leverage on that leak is large.
    #:
    #: Filling at the close instead makes the AM step an *early decision with a
    #: delayed fill*: the policy commits on the open's information and executes
    #: hours later at a price it did not know when it committed.  That is a cost
    #: to the policy, not a subsidy, which is the right direction for the error
    #: to run.  Marks are left alone -- an AM mark is stale but log returns
    #: telescope through it, so it cannot bias the cumulative return.
    #:
    #: Set ``False`` only if the chain gains a genuine 09:30 snapshot.
    am_fills_at_close: bool = True
    #: 09:30 / 16:00, matching ``env.clock.DecisionClock`` and, more to the
    #: point, matching the ``decision_time`` the datasets stamp on their own
    #: rows: ``market_open`` carries 13:30 UTC and ``market_close`` carries
    #: 20:00 UTC, in both ``option_chain_snapshots`` and
    #: ``underlying_market_features``.  An earlier draft used 09:45/15:45 and
    #: failed the point-in-time assertion on the first PM step of the window —
    #: every close-step record became available 15 minutes *after* that
    #: decision time, so there was no data the policy was permitted to read.
    #:
    #: The consequence is that a PM decision reads close information and fills
    #: against close quotes.  That is not look-ahead — nothing stamped after
    #: 16:00 is read — but it is a simultaneous fill, and the half-spread
    #: charged on entry is the only thing standing in for the fact that an
    #: order cannot be both informed by the closing print and executed at it.
    am_time_et: time = time(9, 30)
    pm_time_et: time = time(16, 0)
    timezone_name: str = "America/New_York"

    def __post_init__(self) -> None:
        known = {"AM", "PM"}
        for name, value in (
            ("decision_sessions", self.decision_sessions),
            ("mark_sessions", self.mark_sessions),
            ("hedge_sessions", self.hedge_sessions),
        ):
            if not value:
                raise ConfigError(f"{name} is empty")
            if set(value) - known:
                raise ConfigError(f"{name} contains unknown sessions: {value}")
        if set(self.decision_sessions) - set(self.mark_sessions):
            raise ConfigError("cannot decide on a session that is not marked")
        if set(self.hedge_sessions) - set(self.mark_sessions):
            raise ConfigError("cannot hedge on a session that is not marked")


@dataclass(frozen=True, slots=True)
class TenorSpec:
    """Admitted tenor buckets and the DTE window each one means.

    ``docs/env_contract.md`` section 6.2 argues for a single 7-day tenor.  The
    ingested chain does not support it: on 2025-06-09 the nearest listed expiry
    in ``massive_option_quote_snapshots`` is 11 DTE for every one of the five
    single names and for SPX, because ingestion samples a sparse expiry ladder
    rather than the full chain.  ``8_30`` is the shortest bucket that resolves.

    ``preferred_dte`` is the anchor the resolver targets inside the bucket.

    **All four buckets are admitted, and tenor is part of the action space.**
    It was pinned to ``8_30`` through the zero-shot and cost-disclosure arms,
    which made every package a 15-DTE package and left ``ts`` (the 90d-30d
    term-structure field) visible but untradeable.  Coverage was checked before
    widening: on twelve dates sampled across the window, every one of the ten
    names lists at least one admissible expiry in every bucket under the full
    ``_admit`` gate, so no bucket is decorative.  ``0_7`` is the thin one --
    median one expiry for the single names against six for SPY -- and is the
    bucket most likely to reopen the same-day round-trip loop, so it is tracked
    separately in the analysis rather than suppressed here.
    """

    admitted: tuple[str, ...] = ("0_7", "8_30", "31_90", "91_180")
    ranges: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "0_7": (0, 7),
            "8_30": (8, 30),
            "31_90": (31, 90),
            "91_180": (91, 180),
        }
    )
    preferred_dte: dict[str, int] = field(
        default_factory=lambda: {"0_7": 5, "8_30": 14, "31_90": 45, "91_180": 120}
    )
    max_dte: int = 182
    # Named, not ``admitted[0]``.  Widening ``admitted`` used to move the
    # default silently: prepending ``0_7`` would have re-pointed every caller
    # that still asks for a default at the thinnest and most churn-prone
    # bucket, with nothing in the config diff saying so.  Tenor is required on
    # the wire, so this is now only a fallback for non-policy callers.
    default: str = "8_30"

    def __post_init__(self) -> None:
        if not self.admitted:
            raise ConfigError("no tenor bucket admitted")
        if self.default not in self.admitted:
            raise ConfigError(f"default bucket {self.default!r} is not admitted")
        for bucket in self.admitted:
            if bucket not in self.ranges:
                raise ConfigError(f"admitted bucket {bucket!r} has no DTE range")
            if bucket not in self.preferred_dte:
                raise ConfigError(f"admitted bucket {bucket!r} has no preferred DTE")
            low, high = self.ranges[bucket]
            if not low <= self.preferred_dte[bucket] <= high:
                raise ConfigError(f"preferred DTE for {bucket!r} is outside its range")
            if high > self.max_dte:
                raise ConfigError(f"bucket {bucket!r} exceeds max_dte={self.max_dte}")

    @property
    def default_bucket(self) -> str:
        return self.default


@dataclass(frozen=True, slots=True)
class CoordinateBounds:
    """Per-family delta and width bounds for the parametric action space.

    ``docs/env_contract.md`` section 4.4 makes the standard values defaults
    rather than the whole space.  ``defaults`` is what an omitted coordinate
    resolves to; ``bounds`` is what an emitted coordinate is checked against.
    Both are in delta units on ``[0, 1]``.
    """

    defaults: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "outright": {"long_delta": 0.55},
            "debit_vertical": {"long_delta": 0.55, "width_delta": 0.10},
            "credit_vertical": {"short_delta": 0.35, "width_delta": 0.10},
            "long_straddle": {"call_delta": 0.50, "put_delta": 0.50},
            "long_strangle": {"put_delta": 0.25, "call_delta": 0.25},
            "iron_butterfly": {"short_delta": 0.50, "wing_delta": 0.25},
            "iron_condor": {"short_delta": 0.30, "width_delta": 0.10},
            "defined_risk_reversal": {"directional_delta": 0.35, "tail_wing_delta": 0.20},
            "butterfly": {"center_delta": 0.45, "lower_width_delta": 0.10, "upper_width_delta": 0.10},
        }
    )
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "long_delta": (0.20, 0.80),
            "short_delta": (0.10, 0.55),
            "width_delta": (0.05, 0.30),
            "wing_delta": (0.05, 0.40),
            "call_delta": (0.10, 0.55),
            "put_delta": (0.10, 0.55),
            "directional_delta": (0.15, 0.60),
            "tail_wing_delta": (0.05, 0.35),
            "center_delta": (0.25, 0.70),
            "lower_width_delta": (0.05, 0.30),
            "upper_width_delta": (0.05, 0.30),
        }
    )
    delta_step: float = 0.05

    def defaults_for(self, family: str) -> dict[str, float]:
        try:
            return dict(self.defaults[family])
        except KeyError as exc:
            raise ConfigError(f"no coordinate defaults for family {family!r}") from exc


@dataclass(frozen=True, slots=True)
class ResolverBounds:
    """What the chain refuses to return, and what it merely flags.

    The two groups are not interchangeable and the split is deliberate.

    **Gates** answer *can this contract be traded and priced at all*, and a
    failure removes the row: no Greeks to resolve a delta coordinate against, no
    two-sided market to derive a mid or a crossing cost from, a deliverable that
    is not 100 shares of the named underlying, a quote published after the
    decision, or an expiry that has already passed.  Those are questions about
    whether the instrument exists as described, and a wrong answer makes the
    fill fictional.

    **Warnings** (the ``warn_*`` fields) answer *how good is this market*, and a
    failure keeps the row and records a flag.  They were gates until 2026-09-13
    and the demotion was a correctness fix, not a loosening.  Removing a row for
    thinness does not stop the resolver trading; ``nearest_delta`` has no
    tolerance, so it silently lands on whatever survives however far away that
    is.  Measured over 1,002 (name, near expiry) cells, 20.1% had every call in
    the 45-55 delta band gated out, and the resolver answered those with
    contracts as far as 48 delta from target and reported success.  A package
    built on the wrong strikes is a worse outcome than one built on a thin
    strike, because the thin strike at least pays the spread it advertises.

    ``warn_open_interest_below`` was the worst of the three.  Open interest is
    published a session in arrears by construction -- ``open_interest_economic_
    date`` is the prior trading day -- so a newly listed at-the-money weekly
    carries yesterday's number, often zero.  On 2024-09-03, 96 at-the-money
    calls with two-sided ``primary`` quotes were dropped for it and 45 of them
    had traded 100+ contracts that same session.

    ``warn_quote_tier_outside`` uses the chain builder's own
    ``quote_quality_tier`` rather than a second set of thresholds reasoning
    about the same evidence.  Measured on 2025-06-04: of 54,259 rows, 26% are
    ``primary``, 6% ``fallback``, 68% ``diagnostic_only`` and 19 ``invalid``.
    The ``diagnostic_only`` bucket is dominated by ``missing_pricing_input`` and
    ``iv_width_above_fallback``; the first is still *gated*, because a row with
    no Greeks fails the Greeks gate on its own evidence rather than on its
    label.

    ``max_quote_age_seconds`` stays a gate, and is enforced against
    ``decision_time - quote_available_time``, **not** the dataset's
    ``quote_age_seconds`` column.  That column is measured relative to the
    source snapshot, so an AM row built from the prior close reads ~60 seconds
    while being 17.5 hours stale.  See ``chain.py``.
    """

    max_relative_spread_package: float = 0.20
    max_quote_age_seconds: float = 900.0
    min_dte: int = 1
    delta_field: str = "model_delta"
    delta_convention: str = "signed_black_scholes"
    require_two_sided_quote: bool = True
    require_standard_deliverable: bool = True
    expected_multiplier: int = 100
    #: Advisory only.  See the class docstring for why each stopped gating.
    warn_open_interest_below: int = 100
    warn_relative_spread_above: float = 0.25
    warn_quote_tier_outside: tuple[str, ...] = ("primary", "fallback")


#: Which limits enter the minimum that decides a size.  The count gates
#: (``max_positions``, ``max_positions_per_underlying``) are not in any of these
#: because they are not sizing limits: they bound how many *distinct* decisions
#: are live, which is a context-budget question, and they refuse rather than
#: shrink.
#:
#: **THE CEILING WAS REMOVED 2026-09-22**, by user ruling: *"drop the
#: max_loss_cap from the size bounds."*  That deletes two of the five rule names
#: rather than loosening them.  ``cap_for(family)`` was the entire numerator of
#: ``floor(c * NAV / max_loss)``, and a limit with no numerator is not a limit,
#: so the two rules whose limits were *only* the ceiling and the scenario --
#: ``max_loss`` and ``scenario_max_loss`` -- cease to exist.  A manifest naming
#: either still describes what was run, because a rule name is a runtime
#: selector and not a stored format, but it can no longer be re-run on this code.
#:
#: What the removal buys, and it is the reason for it: the ceiling was the
#: binding limit on **every** package the oracle selected, so ``nav_fraction``
#: -- the shipped default and the toggle-OFF arm -- was ``max_loss`` wearing a
#: premium label and sweeping ``f`` measured nothing.  ``f`` now binds.
#:
#: What it costs, stated here rather than discovered later: per-package ultimate
#: risk was a flat ``c * NAV`` = 2% of NAV under the ceiling, uniform across
#: families by construction.  It is now ``f * (max_loss / gross_premium) * NAV``,
#: and that ratio is 1.0 for an outright but larger for anything whose worst case
#: is a strike width rather than a premium.  Uniform ultimate risk was a property
#: of the ceiling, not of the book, and it is gone with it.  ``docs/env_contract``
#: 6A.1 carries the measurement.
#:
#: ``full`` is the remaining ablation.  Measured across 77 ``algo_v2_core`` arms
#: and 13,770 approved fills, the family/max-loss limit bound 99.5% of them and
#: ``delta`` 0.5%; ``name``, ``total`` and ``cash`` bound **zero** times.  That
#: measurement is now history rather than guidance -- the limit that bound 99.5%
#: of those fills is the one that was just deleted -- but ``name`` and ``total``
#: survive in ``full`` and are the only remaining limits expressed in max-loss
#: terms, so ``full`` is now the rule to reach for when ultimate risk has to be
#: bounded at all.
#:
#: ``scenario`` never had the ceiling as its *unit*: the scenario measures a
#: one-day loss and max loss measures the ultimate one, and a short-gamma package
#: with large positive theta scores near zero on the first while being unbounded
#: on the second.  It carries ``cash`` because with no ceiling there is no
#: arithmetic guarantee that the book can fund what it approves.
#:
#: ``nav_fraction`` is the rule that does not look at risk at all: every position
#: is the same fraction of NAV in gross premium, so the size depends on what the
#: package *costs* and not on what it can lose or how it moves.  It is the null
#: hypothesis the scenario term has to beat.  Nothing in it reads a greek, so it
#: is also the only rule well defined on a package whose Greeks the chain could
#: not resolve.  It carries ``cash`` for the same reason ``scenario`` does:
#: gross premium does not bound max loss, so without ``cash`` the book could
#: approve a fill it cannot settle.
SIZE_RULES: tuple[str, ...] = (
    "scenario",
    "full",
    "nav_fraction",
)

#: The limits each rule takes a minimum over, in report order.
#:
#: **The two toggle arms are nested, not disjoint** (user ruling, 2026-09-22:
#: *"bound the position size by upper bound of 10% NAV when opening it.  when the
#: toggle is off, only apply the 10% NAV upper bound."*).  ``scenario`` is
#: ``nav_fraction`` with exactly one limit added, so turning the toggle on adds a
#: term rather than swapping the rule out.  That is what makes the two arms a
#: comparison: their difference isolates the scenario term, where two disjoint
#: rules at an arbitrary ``f`` would have measured the two *scales* as much as
#: the two *shapes* -- the caveat ``SizeBounds.nav_fraction`` used to state
#: against itself and no longer has to.
#:
#: **The nesting survived the ceiling removal, which is the thing to check.**
#: ``scenario`` is still ``nav_fraction`` plus exactly one limit, so the
#: toggle is still a single term and ``N0 -> S0`` still isolates the scenario.
#: Dropping ``max_loss`` took the same limit out of both arms, so the
#: *difference* between them is unchanged -- it is the common part that moved.
#:
#: ``cash`` is what now bounds ``_scenario_limit``'s ``+inf`` cliff, together
#: with ``nav_fraction``.  The scenario returns ``+inf`` whenever one-day risk is
#: non-positive, which a long-gamma package near delta-neutral genuinely can be
#: -- 8 of 1,661 menu packages in the 2026-09-21 sweep -- and a rule that can
#: return ``+inf`` must never be the only limit.  Before 2026-09-22 the ceiling
#: was the guard; now ``nav_fraction`` is, and it is a *premium* guard rather
#: than an ultimate-risk one.  It bounds those packages, but at
#: ``f * max_loss / gross_premium`` of NAV rather than at ``c``.
#:
#: ``nav_fraction`` -- the OFF arm -- is the **shipped default** as of
#: 2026-09-22, so the sweep baseline is one of the two arms rather than a third
#: shape.  See ``SizeBounds.size_rule``.
#:
#: ``full`` is not an arm.  It is the ablation, and since 2026-09-22 it is also
#: the only rule that still bounds ultimate risk: ``name`` and ``total`` divide
#: the remaining per-underlying and book-wide max-loss budgets by this package's
#: max loss, so they are ceilings in all but name -- book-level rather than
#: per-package, and therefore not a drop-in replacement for what was removed.
#:
#: ``full`` **gained** ``nav_fraction`` on 2026-09-22 and it is not an addition.
#: ``full`` is every limit there is -- that is what the name claims and what
#: makes it usable as an ablation, because ``binding`` under ``full`` has to be
#: comparable to ``binding`` under any arm.  It held that property by carrying
#: ``max_loss``, a superset of ``scenario_max_loss``; when the ceiling went it
#: would otherwise have become the one rule *missing* a live limit, and a
#: cross-rule reading of "what was the constraint" would have compared labels
#: rather than behaviour.
SIZE_RULE_LIMITS: Mapping[str, tuple[str, ...]] = {
    "scenario": ("scenario", "nav_fraction", "cash"),
    "full": ("scenario", "nav_fraction", "name", "total", "delta", "cash"),
    "nav_fraction": ("nav_fraction", "cash"),
}


@dataclass(frozen=True, slots=True)
class SizeBounds:
    """Everything ``SizeResolver`` needs (``docs/env_contract.md`` section 6A.4)."""

    #: Which limits are live.  See ``SIZE_RULES``.
    #:
    #: ``nav_fraction`` by user ruling, 2026-09-22: *"remove scenario_max_loss
    #: from default, use nav_fraction as the default.  run the oracles, with the
    #: default as the baseline, and only change one toggle in each arm."*
    #:
    #: The default is the toggle-**OFF** arm, which is what makes the sweep a
    #: single-toggle design: every other arm is this rule with exactly one field
    #: changed, so the baseline is a member of the comparison rather than a shape
    #: sitting outside it.  ``scenario_max_loss`` -- the previous default -- was
    #: neither arm, and was deleted outright later the same day when the ceiling
    #: it was built on went.
    size_rule: str = "nav_fraction"

    #: The sizing unit: a one-day, one-sigma adverse move is budgeted to cost
    #: this fraction of NAV per position.  It replaces ``base_risk_unit``, which
    #: budgeted ``max_loss`` instead and was the binding limit on 91% of fills
    #: across four measured arms -- so every position in every arm was the same
    #: size and the rule was a constant with five decorations.
    #:
    #: Budgeting max loss is not risk parity, because max loss is reachable for
    #: some structures and fictional for others.  Measured over 687 PM
    #: position-steps at an identical 1%-of-NAV max-loss budget, an outright
    #: carried a median |$delta| of 10,307 against a debit vertical's 1,123 --
    #: 9x the exposure for the same budget, because a long call's 100%-loss tail
    #: essentially never realizes over 15-43 DTE while a vertical's does.  That
    #: is a standing subsidy to the simplest structure, and the policy took it:
    #: 664 of those 687 steps were outrights.
    target_scenario_risk: float = 0.005

    #: The sizing unit under ``size_rule="nav_fraction"`` and a live ceiling
    #: under ``size_rule="scenario"``: each position commits at most this
    #: fraction of NAV in **gross premium**, ``floor(f * NAV / gross_premium)``.
    #: The ``full`` ablation does not read it -- see ``SIZE_RULE_LIMITS``.
    #:
    #: Gross and not net, because ``mid_cost`` is signed and negative for the
    #: four credit families, so ``f * NAV / mid_cost`` returns a negative
    #: quantity on ``cv``, ``ip``, ``ic`` and ``ib``.  Gross premium is the one
    #: quantity that is positive, finite and defined for all nine families
    #: without consulting a greek.
    #:
    #: **Since the ceiling was removed on 2026-09-22 this is normally the
    #: binding limit**, which is the point of the removal -- it previously was
    #: not, on any package the oracle selected.  It is a bound on *cost*, so
    #: ultimate risk per position is ``f * (max_loss / gross_premium) * NAV``
    #: and varies by family.  Nothing bounds that ratio any more.
    #:
    #: 0.10 per the user's ruling of 2026-09-22 ("equal sized position, at ten
    #: percent of nav per position"), superseding the 0.01 placeholder.
    #:
    #: **At this value the book is deliberately over-subscribed and ``cash`` is
    #: expected to bind.**  Twelve positions at 10% of NAV in gross premium is
    #: 120% of NAV, so a full book cannot be funded and the realized position
    #: count settles below ``max_positions``.  That is the intended design, not
    #: an oversight: the rule targets a ten-position book, and which limit
    #: actually bound is recorded per fill so that a comparison against the
    #: scenario rule can separate orders where the sizing rule bound from orders
    #: where capital did.  The guard in ``_validate`` does not fire because
    #: ``nav_fraction`` retains the ``cash`` limit (``SIZE_RULE_LIMITS``).
    #:
    #: The caveat this field used to state against itself -- *"comparing this
    #: rule against a scenario rule at an arbitrary ``f`` measures the two
    #: scales as much as the two shapes"* -- is retired as of the 2026-09-22
    #: nesting ruling.  ``scenario`` is now ``nav_fraction`` plus one limit, so
    #: the same ``f`` is deployed in both arms by construction and their
    #: difference is the scenario term alone.
    nav_fraction: float = 0.10
    #: Relative shock applied to the package's own implied vol in the scenario,
    #: charged as adverse in both spot directions.  Direction-agnostic on
    #: purpose: signing it would assert a spot/vol correlation the resolver has
    #: no business holding a view on.
    vol_shock_relative: float = 0.10
    #: ``scenario_sigmas`` (``k``) was **removed on 2026-09-22**, not defaulted.
    #: It multiplied the spot move, so it reached delta linearly and gamma
    #: quadratically and vega and theta not at all -- a real second dimension,
    #: which is why it was configurable.  But it was only ever run at 1.0, and
    #: 1.0 is the value at which ``Theta_day = -1/2*Gamma*(sigma_day*S)^2`` makes
    #: the gamma and theta terms cancel for a delta-hedged package.  Measured
    #: after the 2026-09-20 day-count fix, the uncancelled residual
    #: ``|Gterm+Ttheta| / |Gterm|`` is 0.009 for a long straddle and 0.016 for a
    #: long strangle.  Leaving the knob in made that identity a matter of
    #: configuration; removing it makes it a matter of construction.  Runs whose
    #: manifest carries a ``scenario_sigmas`` entry predate this and were sized
    #: under the old expression.
    trading_days_per_year: float = 252.0
    #: ``cap_family``/``default_cap_family`` were **removed 2026-09-22** by user
    #: ruling -- *"drop the max_loss_cap from the size bounds"* -- along with the
    #: ``max_loss`` limit they scaled and the two rules that consisted of it.
    #: They are not deprecated-but-accepted: a config that still passes them
    #: raises ``TypeError`` from the dataclass, which is the loud failure, and is
    #: wanted, because silently ignoring a risk ceiling somebody thought they had
    #: set is the bad version of this change.  See ``SIZE_RULES``.
    #:
    #: 0.05 and 0.20, down from 0.06 and 0.30.  At the older values neither could
    #: fire: 3 positions/name and 12 positions at a 1%-of-NAV unit reach 3% and
    #: 12%, so binding required a 40-60% drawdown first.  Measured across four
    #: arms and 144 successful opens, neither was ever the binding limit once.
    #: They were then sized against the 2% ceiling -- 3 x 2% = 6% > 5% and
    #: 12 x 2% = 24% > 20% -- so a loaded name and a full book both bind.  **That
    #: sizing argument died with the ceiling on 2026-09-22** and these two numbers
    #: have not been re-derived against ``f``; they are live only under ``full``,
    #: where they are now the sole remaining bound on ultimate risk.
    max_risk_per_underlying: float = 0.05
    max_total_open_risk: float = 0.20
    max_net_dollar_delta: float = 0.50
    max_positions: int = 12
    max_positions_per_underlying: int = 3
    collateral_buffer: float = 0.05
    conviction_multipliers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Execution costs (``docs/env_contract.md`` section 7, 12.4).

    ``half_spread_multiplier`` is the sensitivity axis of
    ``docs/evaluation_protocol.md`` section 10 and is swept over
    ``(0.0, 0.25, 0.5, 1.0)``; the default of 1.0 is the honest one, because
    marks are taken at mid and a mark-to-mid book that never pays the spread is
    the reward hack that section guards against.
    """

    half_spread_multiplier: float = 1.0
    option_fee_per_contract: float = 0.65
    assignment_fee: float = 5.00
    stock_commission_per_share: float = 0.005
    stock_half_spread_bps: float = 1.0
    borrow_rate_annual: float = 0.005


@dataclass(frozen=True, slots=True)
class MarkingSpec:
    """Bounds for valuing a position the chain cannot price.

    Separate from ``ResolverBounds`` on purpose.  A quote good enough to value
    something already owned is not good enough to open a position against, and
    collapsing the two is how a backtest starts filling at prices nobody could
    have traded.  Nothing here may reach an execution path.

    ``max_mark_quote_age_seconds`` is deliberately looser than the resolver's
    900 s.  The out-of-chain source returns the last NBBO at or before the
    instant, so on an illiquid contract it can be hours old -- and an hours-old
    print is still a far better estimate of what the position is worth than the
    entry price, which is the alternative.  The bound exists so that "last
    traded some time in the previous week" does not silently become a mark.
    """

    max_mark_quote_age_seconds: float = 6.0 * 3600.0
    #: Binomial steps for recovering Greeks from an out-of-chain mid.  Matches
    #: the chain builder's ``american_tree_steps`` so that a repriced leg and a
    #: chain-priced leg are on one convention and can be summed.
    american_tree_steps: int = 128


@dataclass(frozen=True, slots=True)
class RiskControls:
    """Stop-loss and the ruin floor (``docs/env_contract.md`` sections 8.7, 11.2)."""

    stop_loss_enabled: bool = True
    stop_loss_fraction: float = 0.60
    ruin_floor_kappa: float = 0.20
    force_close_dte: int = 1
    max_stale_mark_share: float = 0.25


#: The families whose thesis is volatility rather than direction.
#:
#: This is the answer to ``env_contract.md`` section 9.1 (``⟨Q23⟩``), which
#: left open which families carry a hedge policy.  The split is on what the
#: package is *for*.  A straddle, strangle, butterfly, iron butterfly or iron
#: condor is a bet on realized versus implied volatility; whatever delta it
#: carries is a by-product of where the strikes landed, and leaving it
#: unhedged means the P&L is contaminated by a directional move the trade was
#: never expressing.  That is the Coval-Shumway construction the contract's
#: section 4.3.3 already points at for ``ls``, applied to its siblings for the
#: same reason.
#:
#: The four omitted families -- outright, debit vertical, credit vertical and
#: defined risk reversal -- are directional by construction.  Delta *is* the
#: thesis, so hedging them to a band would null the position the policy asked
#: for and make the arm untestable.
#:
#: Note the limit: this keys on family, not on the orientation the policy
#: chose.  A butterfly opened bearish is a direction bet and will still be
#: hedged.  Orientation-aware hedging is ``⟨Q24⟩`` and is not settled here.
VOLATILITY_FAMILIES: tuple[str, ...] = (
    "butterfly",
    "long_straddle",
    "long_strangle",
    "iron_butterfly",
    "iron_condor",
)

#: The complement, named so that a reader does not have to derive it.
DIRECTIONAL_FAMILIES: tuple[str, ...] = (
    "outright",
    "debit_vertical",
    "credit_vertical",
    "defined_risk_reversal",
)


@dataclass(frozen=True, slots=True)
class HedgeSpec:
    """Delta hedging (``docs/env_contract.md`` section 9).

    ``proxy_instruments`` names the ticker that carries an underlying's hedge
    when the underlying itself has no share to trade.  SPX is the case that
    forces it: it is tradeable (see ``UniverseSpec``) and cash-settled, so
    without a proxy its delta is bounded only by ``max_net_dollar_delta`` and
    never actually offset.

    SPY is the right proxy on measurement, not merely by convention.  Over the
    246-date window ``normalized/option_trades`` carries 248,994,504 SPY prints
    -- more than SPX's 197,789,446 and denser than any single name -- each one
    stamped with an ``underlying_price``, so the hedge instrument is priceable
    at every grid point the index is.  Basis risk is the ETF's, and it is small
    against a 5% band: SPY holds the index outright, so the tracking difference
    over a hedge's life is the 9.45 bp fee and the dividend-accrual sawtooth,
    both bp-scale.

    The ratio is 1.0 and is *not* SPY's 1/10 price ratio.  Hedging is stated in
    dollars of delta throughout, and the share count falls out as dollars over
    the proxy's own spot, so the tenfold price difference cancels without ever
    being written down.  A non-unit entry here would mean a genuine beta, which
    SPY against SPX does not have.

    The short side is where the proxy stops being free.  A short SPY balance
    owes the dividend, which is real cash the book does not model (section 8.3
    charges borrow, not distributions), and SPY is an American-style equity
    line while SPX options are European -- so hedging an index with the ETF
    swaps a settlement discontinuity the resolver cannot model for a cash item
    it does not charge.  That is the honest price of being able to hedge at all.
    """

    enabled: bool = False
    portfolio_level: bool = True
    #: Half-width of the no-trade region, as a fraction of NAV (section 9.2).
    #:
    #: 0.005 rather than the 0.05 this shipped with, because 0.05 never bound.
    #: Measured on the 2024-09-03..09 volatility-hedged arm: across 13 examined
    #: (step, instrument) groups the largest net exposure was -$9,199 against a
    #: $50,000 band -- 20% of it -- so the resolver ran five times, declined five
    #: times, and the arm's manifest was indistinguishable from one where
    #: hedging had been switched off.  A band that no book in the universe can
    #: leave is not a risk control, it is an unexercised branch.
    #:
    #: At 0.005 the same 13 groups fire twice (NVDA at -$9,199 and -$8,945),
    #: correcting $4,199 and $3,945 back to the edge.  That is the whole point of
    #: the number: it is the widest band that the *observed* delta of a
    #: delta-neutral-by-construction book actually crosses.  Straddles are opened
    #: near flat, so their residual delta is small in absolute terms and a band
    #: wide enough to ignore it ignores everything.
    #:
    #: The cost side is not yet measured.  A tighter band trades more often and
    #: pays the share half-spread each time, and whether 0.005 is *optimal* is a
    #: sweep nobody has run; ``MIN_HEDGE_FRACTION`` bounds the churn but does not
    #: price it.  What is settled is that 0.05 measured nothing.
    delta_band: float = 0.005
    hedge_to_band_edge: bool = True
    allow_fractional_shares: bool = True

    #: ``fixed`` uses ``delta_band * nav`` for every name and every session.
    #: ``whalley_wilmott`` replaces it with the utility-indifference band
    #:
    #:     H_shares = band_multiple * (1.5 * k * S * Gamma^2 * nav / risk_aversion)^(1/3)
    #:
    #: evaluated per hedge group, where ``Gamma`` is that group's share-gamma
    #: (delta per $1 move) and ``k`` is the relative *half*-spread of the
    #: underlying at that session.  The band is then ``H_shares * S`` dollars,
    #: so it stays in the dollar-delta units the rest of the resolver speaks.
    #:
    #: The point of the rule is that the band is no longer one number: it widens
    #: where gamma is high (delta runs away faster, so a tight band churns) and
    #: where the spread is wide (each correction costs more).  The fixed rule
    #: has neither dependence, which is why it over-hedges MU at the open and
    #: under-hedges SPY all day.
    #:
    #: The ``e^{-r(T-t)}`` discount factor of the published formula is omitted.
    #: At r = 5% and a 180-day maximum tenor it is at most 2.5% in ``H^3``, i.e.
    #: 0.8% in the band, and carrying it would require a time-to-expiry per
    #: hedge group that the book does not aggregate.  Stated rather than faked.
    band_rule: str = "fixed"

    #: Relative risk aversion, i.e. the published ``gamma`` times NAV.  Stated
    #: relative so the band does not change meaning when the account grows.
    #:
    #: Larger means more risk averse means a *narrower* band and more hedging.
    #: It enters as a cube root, so this is a coarse knob by construction: a
    #: factor of 8 here moves the band by 2.  That is a feature for a sweep --
    #: ``(1, 8, 64, 512)`` spans an order of magnitude in band width in four
    #: arms -- and the reason the default is a round number rather than a fitted
    #: one.  Nothing about 1.0 is optimal; ``band_multiple`` and the clamps are
    #: what keep it sane, and which value is best is the sweep, not a default.
    risk_aversion: float = 1.0

    #: The "wider than the continuous formula gives" factor.
    #:
    #: Whalley-Wilmott is an asymptotic result for *continuous* observation as
    #: the cost goes to zero.  This environment observes twice a day, so between
    #: observations delta drifts unwatched and a band sized for continuous
    #: monitoring is systematically too narrow -- the exposure spends time
    #: outside it that no observation ever sees.  Widening is the cheap honest
    #: correction; the expensive correct one is a discrete-time band nobody has
    #: solved for this grid.
    band_multiple: float = 1.0

    #: Floor and ceiling on the band as a fraction of NAV.
    #:
    #: The floor is load-bearing, not defensive: ``H`` goes to zero with gamma,
    #: and a group whose gamma has decayed to nothing would get a zero-width
    #: band and be re-hedged to the dollar on every single grid point, paying
    #: the spread forever to correct noise.  ``MIN_HEDGE_FRACTION`` does not
    #: save it, because that is a fraction *of the band*, so it shrinks too.
    #:
    #: The ceiling bounds the loss the band rule can admit, so that a high-gamma
    #: group cannot quietly opt out of the delta limit altogether.
    min_band_fraction: float = 0.001
    max_band_fraction: float = 0.05

    #: Relative half-spread used when the spread table has no row for a name.
    #:
    #: Defaults to ``CostModel.stock_half_spread_bps`` expressed as a fraction,
    #: so a missing row degrades to the flat cost the environment already
    #: assumed rather than to zero -- ``k = 0`` collapses the band and would
    #: turn a data gap into unlimited hedging.
    fallback_half_spread: float = 1.0e-4
    #: Which strategies get hedged, resolving ``env_contract.md`` section 9.1.
    #: Empty by default so that a config which never says what to hedge hedges
    #: nothing, rather than guessing on the operator's behalf.  The intended
    #: value is ``VOLATILITY_FAMILIES``; see its docstring for the split.
    hedged_families: tuple[str, ...] = ()
    proxy_instruments: dict[str, str] = field(default_factory=lambda: {"SPX": "SPY"})

    def hedge_ticker_for(self, underlying: str) -> str | None:
        """The ticker whose shares hedge ``underlying``, or ``None`` if it has none.

        Names with their own share hedge themselves; the map is consulted only
        for the ones that do not, so a stray entry for a single name cannot
        silently redirect its hedge to something else.
        """
        from .resolvers.hedge import NON_SHARE_UNDERLYINGS

        if underlying not in NON_SHARE_UNDERLYINGS:
            return underlying
        return self.proxy_instruments.get(underlying)


#: Band rules ``HedgeSpec.band_rule`` admits.  ``fixed`` is the shipped
#: behaviour and stays the default, because every trajectory recorded so far was
#: produced under it and switching the default would silently make them
#: non-comparable rather than loudly.
BAND_RULES: tuple[str, ...] = ("fixed", "whalley_wilmott")

#: Wires the quote round can run on.  ``text`` is what every quote arm before
#: 2026-09-24 ran under; ``tool`` is the ruled default from that date.  Listed
#: as a closed set because a typo would otherwise read as "not tool", which is
#: the one failure that looks exactly like a healthy run: the prompt keeps its
#: ``Q`` grammar, the provider is sent no schema, and the arm quietly reverts to
#: the superseded channel with nothing in the manifest to say so.
QUOTE_CHANNELS: tuple[str, ...] = ("text", "tool")


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    """Behaviours that did not exist in the shipped environment, one switch each.

    Separate from the specs that hold the *numbers* on purpose: a flag answers
    "is this on", its spec answers "with what parameters".  Both reach the
    manifest through ``EnvConfig.as_dict()``, so an arm is identified by
    ``fingerprint()`` whichever of the two a sweep varies.

    Everything defaults off.  A config that says nothing gets the environment
    the existing trajectories were run in.

    The flags are deliberately independent rather than nested, so that the
    oracle can be run over the product.  ``measured_spread_costs`` without
    ``whalley_wilmott`` charges the real spread under the flat band, which is
    the control that says how much of any effect is the band rule and how much
    is merely having priced the underlying honestly -- without it the two are
    confounded and a WW arm that beats the baseline proves nothing about WW.
    """

    #: Charge the per-(name, session) measured half-spread at the share fill
    #: instead of the flat ``CostModel.stock_half_spread_bps``.  Off by default:
    #: it changes realized PnL, not just the hedge schedule.
    measured_spread_costs: bool = False

    #: Drop the textual context: no ``N`` rows, and no ``cat``/``dir``/``hzn``/
    #: ``clause`` legend in the system block.  This is the ablation arm --
    #: "remove the textual context, only use the numeric states, with the rest
    #: of the env unchanged".
    #:
    #: Both halves are one switch on purpose.  Suppressing only the rows would
    #: leave a schema line and ~11 lines of legend describing a block that never
    #: arrives: the numeric arm would still pay those prompt tokens and would
    #: still be told that news exists and is being withheld from it, which is a
    #: third condition rather than the control.  Suppressing only the legend
    #: would ship undocumented rows.
    #:
    #: This also drops ``news_catalyst_rows`` from ``required_datasets``, so the
    #: ablation arm does not report a coverage gap on a dataset it never reads.
    suppress_textual_context: bool = False

    #: Tell the policy, in the system block, to reason only from what the step
    #: gives it and not from anything it remembers about these dates.  This is
    #: the "suppress at source" remedy for teacher-trace leakage.
    #:
    #: Off by default, and it has to be, because turning it on changes the
    #: prompt every arm shares: a trajectory generated with this line is not
    #: comparable to one generated without it, so it is an arm, not a fix to be
    #: applied retroactively.
    #:
    #: **This is admissible under the no-leakage rule for prompts and the
    #: distinction is worth stating.** ``system_block`` already separates a
    #: *rule of the environment* -- true on day one, derived from no data,
    #: knowable before any of the scored window happened -- from a *measurement*
    #: taken on the window, which is withheld. "Do not use what you remember
    #: about these dates" is a rule. It quotes no price, no date, no outcome and
    #: no measured constant, so it cannot itself carry the hindsight it asks the
    #: model to set aside.
    #:
    #: **What it cannot do, so the arm is not over-claimed.** It is an
    #: instruction, not a capability bound: a model that knows how November 2024
    #: ended still knows it, and compliance is unverifiable from the outside.
    #: The only remedy that makes leakage impossible rather than rarer is a
    #: teacher whose training cutoff precedes the window. Treat a drop in the
    #: audited leak rate under this flag as evidence about *this* model's
    #: behaviour, never as a guarantee about the corpus.
    suppress_future_knowledge: bool = False

    #: De-identify the observation: tickers become ``U01..U10`` (redrawn every
    #: episode), the calendar date becomes ``m<dte>`` -- days to the standard
    #: third-Friday monthly expiry -- and non-universe proper nouns in the news
    #: clause become ``<ent>``.  The book, the ledger, the resolvers and every
    #: dataset keep the real ticker and the real date; substitution happens only
    #: where text crosses to and from the policy, and the order is unmasked
    #: before it is parsed.
    #:
    #: Off by default for the same reason as the flag above -- it rewrites the
    #: prompt every arm shares, so it is an arm and not a retroactive fix.
    #:
    #: **Why this is the strongest of the three leakage remedies.** Filtering
    #: (remedy A) removes contaminated steps after the fact and costs sample;
    #: the suppression line (remedy B) asks a model that still knows how the
    #: window ended to set that knowledge aside, and compliance is unverifiable.
    #: This one removes the *cue*: a step that names no company and no date
    #: cannot be matched against a memory of that company on that date. It does
    #: not require the model to cooperate.
    #:
    #: **What it still cannot do, so the arm is not over-claimed.** The guard in
    #: ``pseudonyms.residual`` counts proper nouns. Sector language survives --
    #: "memory chip demand", "the election", "the robotaxi maker" each narrow
    #: the window, and the layer reports such a row as clean. The measured
    #: "100% of news rows clean" figure is therefore a statement about proper
    #: nouns only and must never be quoted as proof that a step cannot be dated.
    anonymize: bool = False


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """The whole environment contract as data."""

    version: str = CONFIG_VERSION
    universe: UniverseSpec = field(default_factory=UniverseSpec)
    grid: GridSpec = field(default_factory=GridSpec)
    tenor: TenorSpec = field(default_factory=TenorSpec)
    coordinates: CoordinateBounds = field(default_factory=CoordinateBounds)
    resolver: ResolverBounds = field(default_factory=ResolverBounds)
    size: SizeBounds = field(default_factory=SizeBounds)
    cost: CostModel = field(default_factory=CostModel)
    risk: RiskControls = field(default_factory=RiskControls)
    hedge: HedgeSpec = field(default_factory=HedgeSpec)
    marking: MarkingSpec = field(default_factory=MarkingSpec)
    flags: FeatureFlags = field(default_factory=FeatureFlags)

    initial_cash: float = 1_000_000.0
    admitted_families: tuple[str, ...] = (
        "outright",
        "debit_vertical",
        "credit_vertical",
        "defined_risk_reversal",
        "butterfly",
        "long_straddle",
        "long_strangle",
        "iron_butterfly",
        "iron_condor",
    )
    allow_naked_shorts: bool = False
    risk_free_rate: float | None = None
    state_space_id: str = "state_space.v1"
    max_context_tokens: int = 32_768
    #: Orders one completion may carry before the tail is refused.  Lives here
    #: rather than as a ``parse_action`` default so the number the grammar
    #: prints and the number the parser enforces are one value; they were two,
    #: and the prompt did not print either.
    max_orders_per_step: int = 8
    #: Candidates the ``Q`` verb may price **per underlying** per step.  Ruled
    #: per-name rather than per-portfolio on 2026-09-24: the point of the quote
    #: round is to compare structures *within* a name, and a portfolio-wide cap
    #: of 8 would have let one name's shortlist crowd out every other name's.
    #: Counted separately from ``max_orders_per_step`` because quoting moves no
    #: cash -- the trading limit is a risk control, this is a budget control.
    #: Raised 4 -> 6 on 2026-09-24, after the measurement that the 4 came
    #: from.  The projection that forced 8 -> 4 assumed the cap *is* the usage;
    #: it is not.  Measured on job 46911489 (astra, September, one episode):
    #: the policy quoted on 40% of steps, asked for 4-8 per step spread over
    #: 3-5 names, reached 4 on a single name exactly once, and tripped
    #: ``E_TOO_MANY_QUOTES`` zero times -- peak prompt 20,610 of 32,768.  So
    #: the retention cost scales with what the policy asks for, not with this
    #: number, and headroom is real rather than projected.
    max_quotes_per_name: int = 6
    #: Whether the ``Q`` verb is offered and answered.  One switch for both
    #: halves on purpose: the grammar must advertise the verb exactly when the
    #: runner will spend a turn answering it.  Printing ``Q`` in a prompt no
    #: runner honours would train the policy to emit a line that silently
    #: becomes a hold, and honouring a verb the prompt never named would waste
    #: the turn.  Kept as a knob rather than hard-wired because "does pricing
    #: before ordering change the trades" is a question the sweep has to be
    #: able to ask.
    quotes_enabled: bool = True
    #: How the quote round is carried on the wire: ``"text"`` prints the ``Q``
    #: grammar and reads ``Q`` lines out of the completion, ``"tool"`` declares
    #: ``quote_package`` as a provider-side tool and reads ``toolUse`` blocks.
    #: Ruled ``"tool"`` on 2026-09-24 -- *"I want the quote turn structured as a
    #: tool call, with 6 strategy proposals allowed per name step"* -- and it is
    #: the default because a forgotten flag should run what was ruled.
    #:
    #: It is a config field rather than a policy argument because it changes the
    #: prompt: under ``"tool"`` the ``Q`` grammar lines come *out* of the system
    #: block, since a grammar that teaches a verb the provider now validates
    #: separately invites the model to emit both.  That makes it part of the
    #: arm, hashed into ``fingerprint()``, and not a transport detail.
    #:
    #: Only :class:`BedrockPolicy` speaks ``"tool"`` today; ``DeepSeekPolicy``
    #: refuses it at ``reset`` rather than ignoring it, so an OpenRouter arm
    #: fails at startup instead of running a prompt with no way to ask a price.
    quote_channel: str = "tool"
    #: Which out-of-chain mark source the run was given, by *version* -- the
    #: value :data:`MARK_QUOTE_SOURCE_VERSION` carries, or ``None`` for a run
    #: with no mark source at all.
    #:
    #: Here because it was measurably missing.  ``quote_tool_s09`` ran with
    #: ``massive_mark_quotes.v1`` and 267 requests, ``quote_astra_s09`` with no
    #: mark source whatever, and the two runs agreed on every leaf of
    #: ``as_dict()`` except ``max_quotes_per_name`` and ``quote_channel``.  A
    #: matched fingerprint therefore asserted a matched environment that did not
    #: exist, and the mark source decides what a held position is worth on a day
    #: the chain cannot price it -- which is a term in the reward, not a
    #: diagnostic.
    #:
    #: The *version*, deliberately, and never the cache path: every arm in a
    #: batch is handed its own cache file to avoid interleaved appends on
    #: Lustre, so hashing the path would make ten identically-configured draws
    #: ten different arms and the fingerprint would stop answering the only
    #: question it is asked.
    #:
    #: Adding this field changes every fingerprint ever recorded.  That is the
    #: honest outcome: the old hashes were computed over a description of the
    #: environment that omitted this, so they cannot be reproduced by code that
    #: includes it.  ``STATUS.md`` carries the old -> new mapping for the arms
    #: that matter.
    mark_quote_source: str | None = None

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ConfigError("initial_cash must be positive")
        if not self.admitted_families:
            raise ConfigError("no strategy family admitted")
        if self.quote_channel not in QUOTE_CHANNELS:
            raise ConfigError(
                f"quote_channel must be one of {QUOTE_CHANNELS}, got "
                f"{self.quote_channel!r}"
            )
        for family in self.admitted_families:
            self.coordinates.defaults_for(family)
        # Checked here rather than in the resolver because a sweep that mistypes
        # a band parameter should fail before it burns an episode, and because
        # several of these produce a *plausible* band rather than an error when
        # wrong -- a negative risk_aversion gives a complex cube root, a zero one
        # divides, and either way the failure would surface as a hedge schedule
        # nobody could explain.
        hedge = self.hedge
        if hedge.band_rule not in BAND_RULES:
            raise ConfigError(f"band_rule must be one of {BAND_RULES}, got {hedge.band_rule!r}")
        if hedge.risk_aversion <= 0:
            raise ConfigError("hedge.risk_aversion must be positive")
        if hedge.band_multiple <= 0:
            raise ConfigError("hedge.band_multiple must be positive")
        if hedge.fallback_half_spread < 0:
            raise ConfigError("hedge.fallback_half_spread must not be negative")
        if not 0 < hedge.min_band_fraction <= hedge.max_band_fraction:
            raise ConfigError(
                "require 0 < hedge.min_band_fraction <= hedge.max_band_fraction, got "
                f"{hedge.min_band_fraction} and {hedge.max_band_fraction}"
            )
        size = self.size
        if size.size_rule not in SIZE_RULES:
            raise ConfigError(f"size_rule must be one of {SIZE_RULES}, got {size.size_rule!r}")
        if size.target_scenario_risk <= 0:
            raise ConfigError("size.target_scenario_risk must be positive")
        if size.nav_fraction <= 0:
            raise ConfigError("size.nav_fraction must be positive")
        if size.vol_shock_relative < 0:
            raise ConfigError("size.vol_shock_relative must not be negative")
        # The solvency guard that used to stand here is **gone with the ceiling**
        # (2026-09-22), and deleted rather than left dormant.  It fired only for
        # a rule with no ``cash`` limit, and it discharged its obligation by
        # multiplying ``max_positions`` by the per-package max-loss cap.  Both
        # halves are now vacuous: every surviving rule in ``SIZE_RULE_LIMITS``
        # carries ``cash``, and there is no longer a per-package bound on
        # ``max_loss`` to multiply.  A guard that cannot fire and could not
        # compute its own bound if it did is worse than no guard, because it
        # reads like a proof that solvency is still checked here.
        #
        # Solvency is now carried entirely by the ``cash`` limit at size time,
        # which is exact rather than a bound: it divides real buying power by
        # this package's real collateral plus debit.  What was lost is the
        # *a-priori* guarantee -- the book can no longer be shown solvent from
        # the config alone, only refused one order at a time.
        if "cash" not in SIZE_RULE_LIMITS[size.size_rule]:
            raise ConfigError(
                f"size_rule {size.size_rule!r} drops the cash limit. Since the "
                "max-loss ceiling was removed (2026-09-22) nothing else bounds "
                "what a single package may commit, so such a rule cannot be "
                "admitted -- see SIZE_RULES"
            )
        if self.allow_naked_shorts:
            raise ConfigError(
                "naked shorts are excluded by docs/env_contract.md Q10; admitting them "
                "requires a margin engine that section 8.4 explicitly declines to build"
            )

    def with_(self, **changes: Any) -> "EnvConfig":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=_encode))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnvConfig":
        """Rebuild a config from what ``as_dict`` wrote into a manifest.

        The inverse existed nowhere until 2026-09-24, and its absence was not
        cosmetic.  The system block -- which is where the size rule, the hedge
        disclosure and the action grammar are *stated to the policy* -- is built
        by ``state_space.system_block(config)`` and is **not persisted in the
        run directory**.  So a recorded trajectory could not be turned back into
        the conversation that produced it: the instructions the policy was
        conditioned on were reconstructible only by guessing the flags.  Any
        supervised corpus built from ``decisions.jsonl`` alone would train the
        student on answers without the rulebook that made them correct.

        Decoding is driven by the annotations rather than by a hand-written
        table, so a new field on any of the nested specs is handled the day it
        is added.  A key the dataclass does not declare is an **error**, not a
        skip: ``as_dict`` is ``asdict``, so an unknown key means the manifest
        was written by a different version of this file, and quietly ignoring
        it would rebuild a config that silently differs from the one that ran.

        A *missing* key, by contrast, takes the field default -- and is caught
        downstream instead, by comparing :meth:`fingerprint` against the
        ``env_fingerprint`` the manifest recorded.  That one comparison catches
        missing, extra and altered values together, which is why this method
        does not try to catch them separately.
        """
        return _decode_dataclass(cls, payload, path="EnvConfig")

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode(value: Any) -> Any:
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__} into the env config")


def _decode_dataclass(cls: type, payload: Mapping[str, Any], *, path: str) -> Any:
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path}: expected an object, got {type(payload).__name__}")
    hints = get_type_hints(cls)
    declared = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - declared)
    if unknown:
        raise ConfigError(
            f"{path}: {', '.join(unknown)} is not a field of {cls.__name__}. "
            "The manifest was written by a different version of spec.py; "
            "rebuilding the config while dropping it would produce an object "
            "that fingerprints differently from the run it claims to describe."
        )
    kwargs = {
        name: _decode(hints[name], payload[name], path=f"{path}.{name}")
        for name in declared
        if name in payload
    }
    return cls(**kwargs)


def _decode(annotation: Any, value: Any, *, path: str) -> Any:
    if is_dataclass(annotation):
        return _decode_dataclass(annotation, value, path=path)
    if annotation is time:
        return time.fromisoformat(value)

    origin = get_origin(annotation)
    if origin is UnionType or origin is Union:
        options = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        # ``float | None`` and friends: exactly one real branch in this config,
        # and a union of two real types would be ambiguous to decode, so it is
        # refused here rather than resolved by a guess.
        if len(options) != 1:
            raise ConfigError(f"{path}: cannot decode the union {annotation!r}")
        return _decode(options[0], value, path=path)
    if origin in (tuple, Tuple):
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _decode(args[0], v, path=f"{path}[{i}]") for i, v in enumerate(value)
            )
        if len(args) != len(value):
            raise ConfigError(
                f"{path}: expected {len(args)} elements, got {len(value)}"
            )
        return tuple(
            _decode(a, v, path=f"{path}[{i}]")
            for i, (a, v) in enumerate(zip(args, value))
        )
    if origin in (list, List):
        (arg,) = get_args(annotation)
        return [_decode(arg, v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        _, val_type = get_args(annotation)
        return {k: _decode(val_type, v, path=f"{path}[{k!r}]") for k, v in value.items()}

    # ``json.load`` gives ``int`` where the annotation says ``float``; that is a
    # JSON artefact, not a config difference, and widening it keeps the
    # round-trip fingerprint stable.
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value
