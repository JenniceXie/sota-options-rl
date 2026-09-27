"""Arm registry for the evaluation report.

Every row of Table 1 is declared here, including arms that are not yet built.
Declared-but-unbuilt arms render as blank rows so the table shows what is
missing rather than silently omitting it.

Two independent groupings apply, and conflating them is the easiest way to
produce a comparison that looks fair and is not:

* **Book** (``BOOKS``) is the *set of underlyings*. The index book and the
  single-name book are separate evaluations with separate capital, separate
  benchmarks and, critically, very different statistical power.
* **Band** (``BANDS``) is *how the row was produced*: our policy, our
  replication of a published methodology, or a published level series plugged
  in directly.

A row may be differenced against the policy only if it agrees on both, and on
``basis``. ``pairable`` records the first two; ``basis`` is reported so the
third is visible rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from portfolio_monkey.env.spec import UniverseSpec

#: The 10-name study universe (protocol 7.1), read from the environment config
#: rather than restated. An earlier version of this file kept its own literal
#: tuple, and when the index slot moved from SPX to SPY the registry went on
#: declaring arms over a name the policy could no longer trade -- silently, as
#: every arm here is a *declaration* and declarations do not fail. Deriving it
#: means the two can no longer disagree.
UNIVERSE_NAMES = UniverseSpec().observed

#: The name holding the index exposure. It is the one member of the universe
#: with no ``*DI``/``*CW``/``*IO`` triple, so every place that splits the
#: universe splits on this rather than on position.
INDEX_NAME = "SPY"

SINGLE_STOCKS = tuple(name for name in UNIVERSE_NAMES if name != INDEX_NAME)

#: Display names; the nine single-name entries are taken from the Cboe index
#: directory. ``SPY`` is labelled as the ETF it is, not as "S&P 500", because
#: the benchmark rows in this file *are* labelled "S&P 500" and the two must
#: stay distinguishable in Table 1's first column.
UNDERLYING_LABEL = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "MSFT": "Microsoft",
    "MU": "Micron",
    "NVDA": "NVIDIA",
    "PLTR": "Palantir",
    "TSLA": "Tesla",
    "SPY": "SPDR S&P 500 ETF",
}

BOOKS = {
    "index": "options on the index slot (SPY), benchmarked against Cboe SPX",
    "names": "single-stock options on nine underlyings",
    "all": "the combined ten-name portfolio",
}

BANDS = {
    "A": "Policy arms",
    "B": "Rule replications (our universe, our cost model)",
    "C": "Series arms, published levels",
    "D": "Passive references",
}

#: What a published level actually accrues. This governs which comparisons are
#: like-for-like: an option-overlay level and a total-return level differ by the
#: whole equity risk premium, so their `cumulative log return` values are not
#: two measurements of the same thing.
#:
#: ``total return``     equity leg held, dividends inside the level
#: ``collateralized``   no equity leg; Treasury collateral plus option P&L
#: ``TR + collateral``  equity leg *and* a Treasury account of comparable size,
#:                      so the capital base is roughly twice the equity notional
#: ``option overlay``   option P&L only: no equity leg, no collateral accrual
#: ``price return``     price index, dividends excluded
#: ``income accrual``   distributable income only, no mark-to-market leg, so the
#:                      level is monotone and its risk statistics are degenerate
#: ``unverified``       construction not yet confirmed against Cboe methodology
BASES = (
    "total return",
    "collateralized",
    "TR + collateral",
    "option overlay",
    "price return",
    "income accrual",
    "unverified",
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Declaration of a Table 1 row."""

    name: str
    book: str
    band: str
    universe: str
    kind: str  # policy | rule | series | composite
    underlying: str
    strategy: str
    ticker: str | None = None
    basis: str = "unverified"
    pairable: bool = False
    status: str = "pending"  # pending | built

    @property
    def label(self) -> str:
        """``TICKER (underlying, strategy)`` for the first table column."""

        return f"{self.name} ({self.underlying}, {self.strategy})"


# -- Band A: policy arms ---------------------------------------------------


def _policy(stem: str, strategy: str) -> tuple[ArmSpec, ...]:
    """One policy row per book.

    The three books are three different NAV series, not three views of one:
    a log return is additive over time but not over names, so the combined
    book's cumulative log return is *not* the sum of the two sleeves'.
    Declaring them separately keeps that from being papered over.
    """

    return tuple(
        ArmSpec(
            name=f"{stem}.{book}",
            book=book,
            band="A",
            universe="ours",
            kind="policy",
            underlying=underlying,
            strategy=strategy,
            basis="total return",
            pairable=True,
        )
        for book, underlying in (
            ("index", "SPY"),
            ("names", "9 single names"),
            ("all", "10 names"),
        )
    )


POLICY_ARMS = (
    # The untrained baseline. Without it neither of the trained arms can be
    # read: SFT is only evidence that the teacher labels taught something if
    # the same model, same grammar and same resolvers do worse untouched.
    _policy("policy_zeroshot", "base model, no SFT and no RL")
    + _policy("policy_sft", "SFT checkpoint, no RL")
    + _policy("policy_grpo", "main arm, after GRPO")
    + _policy("bench_random", "random control, matched action rate and size")
)

# -- Band B: rule replications --------------------------------------------

#: Each replication is assigned to the book its methodology can actually run
#: in. The cross-sectional sorts need a cross-section and so cannot run on the
#: index book at all; the Cboe SPX methodologies have no published single-name
#: analogue, so replicating them on the names book would yield a rule with no
#: external series to validate the replication against.
#:
#: The four index-book replications run the *methodology* on **SPY** options,
#: which is what the policy trades. They are the bridge across the instrument
#: mismatch documented at ``EXTERNAL_SPX``: each one is pairable against both
#: the policy (same instrument, same cost model) and its published SPX sibling
#: (same methodology), so the size of the SPY-vs-SPX gap is measurable from the
#: table rather than argued about.
RULE_ARMS = (
    ArmSpec(
        "rule_covered_call", "index", "B", "ours", "rule",
        "SPY", "BuyWrite, BXM methodology",
        basis="total return", pairable=True,
    ),
    ArmSpec(
        "rule_putwrite", "index", "B", "ours", "rule",
        "SPY", "PutWrite, PUT methodology",
        basis="collateralized", pairable=True,
    ),
    ArmSpec(
        "rule_iron_condor", "index", "B", "ours", "rule",
        "SPY", "Iron condor, CNDR methodology",
        basis="collateralized", pairable=True,
    ),
    ArmSpec(
        "rule_collar", "index", "B", "ours", "rule",
        "SPY", "95-110 collar, CLL methodology",
        basis="total return", pairable=True,
    ),
    ArmSpec(
        "rule_deltahedged_cw", "names", "B", "ours", "rule",
        "9 single names", "Delta-hedged call writing, *CW methodology",
        basis="option overlay", pairable=True,
    ),
    ArmSpec(
        "rule_short_straddle_dh", "names", "B", "ours", "rule",
        "9 single names", "Delta-hedged short straddle, Coval & Shumway (2001)",
        basis="option overlay", pairable=True,
    ),
    ArmSpec(
        "rule_goyal_saretto", "names", "B", "ours", "rule",
        "9 single names", "Vol-spread sort, Goyal & Saretto (2009)",
        basis="option overlay", pairable=True,
    ),
    ArmSpec(
        "rule_cao_han", "names", "B", "ours", "rule",
        "9 single names", "Idiosyncratic-vol sort, Cao & Han (2013)",
        basis="option overlay", pairable=True,
    ),
)

# -- Band C: published level series ---------------------------------------


def _series(
    ticker: str,
    book: str,
    universe: str,
    underlying: str,
    strategy: str,
    basis: str,
    *,
    pairable: bool,
    band: str = "C",
) -> ArmSpec:
    return ArmSpec(
        name=ticker,
        book=book,
        band=band,
        universe=universe,
        kind="series",
        underlying=underlying,
        strategy=strategy,
        ticker=ticker,
        basis=basis,
        pairable=pairable,
        status="built",
    )


#: Cboe publishes the Single Stock Defined Income Series as **three** indices per
#: name, all based 100 on 2020-01-02 (PLTR 2020-10-08), and all three are in the
#: level file. They are three different objects, not three vintages of one:
#:
#: ``*DI``  long stock + short weekly OTM call sized to a 20% annualized yield,
#:          delta-hedged daily, plus a money market account. Total return.
#: ``*CW``  the same short call and delta hedge **without** the long stock:
#:          option-only premium collection.
#: ``*IO``  cumulative weekly stock return in weeks that finish up but below the
#:          call strike. A decomposition of where ``*DI``'s upside comes from,
#:          not a portfolio anyone can hold.
#:
#: Declaring only two of the three, as an earlier version of this file did,
#: silently dropped nine rows from Table 1.
CW_TICKERS = tuple(f"{name}CW" for name in SINGLE_STOCKS)
DI_TICKERS = tuple(f"{name}DI" for name in SINGLE_STOCKS)
IO_TICKERS = tuple(f"{name}IO" for name in SINGLE_STOCKS)

SERIES_MATCHED = tuple(
    _series(
        f"{name}CW",
        "names",
        "1 of ours",
        UNDERLYING_LABEL[name],
        "Delta-hedged call writing",
        "option overlay",
        pairable=True,
    )
    for name in SINGLE_STOCKS
) + tuple(
    _series(
        f"{name}DI",
        "names",
        "1 of ours",
        UNDERLYING_LABEL[name],
        "Covered call + daily delta hedge",
        "total return",
        pairable=True,
    )
    for name in SINGLE_STOCKS
) + tuple(
    # Reported for completeness, never paired. ``*IO`` tracks only the income the
    # Defined Income strategy distributes; it carries no mark-to-market leg, so
    # the level is monotone non-decreasing -- ``AAPLIO`` takes one negative step
    # in 1,676 trading days and ``MUIO`` none, against 739 for ``AAPLDI``. Its
    # volatility, drawdown, Sharpe and Calmar are therefore artefacts of that
    # construction rather than measurements of risk, which is exactly why it is
    # marked unpairable. It earns a row because omitting it makes the table look
    # as though Cboe publishes two indices per name when it publishes three, and
    # because the accrual rate is the cleanest read on how much premium the
    # short call actually collects.
    _series(
        f"{name}IO",
        "names",
        "1 of ours",
        UNDERLYING_LABEL[name],
        "Income only, distributable accrual (not tradeable)",
        "income accrual",
        pairable=False,
    )
    for name in SINGLE_STOCKS
)

COMPOSITE_ARMS = (
    ArmSpec(
        "series_cw_ew",
        "names",
        "C",
        "9 of ours",
        "composite",
        "9 single names",
        "Equal-weight composite of the nine *CW indices",
        basis="option overlay",
        pairable=True,
        status="built",
    ),
    ArmSpec(
        "series_di_ew",
        "names",
        "C",
        "9 of ours",
        "composite",
        "9 single names",
        "Equal-weight composite of the nine *DI indices",
        basis="total return",
        pairable=True,
        status="built",
    ),
    # There is deliberately no ``series_io_ew``. The composite builder answers
    # "what would holding all nine equally have returned", which is a question
    # ``*IO`` cannot be asked: it is a conditional slice of a stock return, not
    # something held. The asymmetry against *CW and *DI is the point.
)

#: ``(ticker, strategy, basis)``.
#:
#: ``RXM`` and ``CMBO`` were carried as ``unverified`` until their construction
#: was read off the Cboe methodology PDFs, because "covered" in Cboe naming does
#: not reliably mean stock-covered and the answer decides what may be
#: differenced against what:
#:
#: * ``RXM`` (cdn.cboe.com/api/global/us_indices/governance/RXM_Methodology.pdf)
#:   buys a 25-delta call, writes a 25-delta put and "holds a Treasury bill
#:   account invested in one-month Treasury bills". Its return formula carries
#:   no SPX term and no dividend term, so there is no equity leg.
#: * ``CMBO`` (.../CMBO_Methodology.pdf) writes a 2% OTM call and an ATM put and
#:   *does* "establish a long position indexed to the S&P 500 Index", with
#:   dividends inside the level -- but it also holds a T-bill account of the put
#:   strike. Its capital base is therefore roughly twice the equity notional,
#:   which is why its beta lands near a 1x buy-write's despite carrying a delta
#:   of about 1.15. It is neither ``total return`` nor ``collateralized``, and
#:   giving it its own basis is what stops it being differenced against ``BXM``.
EXTERNAL_SPX = (
    ("BXM", "BuyWrite / covered call", "total return"),
    ("PUT", "PutWrite, cash-secured", "collateralized"),
    ("PUTY", "PutWrite, 2% OTM", "collateralized"),
    ("WPUT", "PutWrite, one-week", "collateralized"),
    ("CNDR", "Iron condor", "collateralized"),
    ("BFLY", "Iron butterfly", "collateralized"),
    ("CLL", "95-110 collar", "total return"),
    ("CLLZ", "Zero-cost put spread collar", "total return"),
    ("PPUT", "5% put protection", "total return"),
    ("RXM", "Risk reversal, long 25d call / short 25d put on T-bills", "collateralized"),
    (
        "CMBO",
        "Covered combo, 1x SPX + short 2% OTM call + short ATM put, T-bill funded",
        "TR + collateral",
    ),
    ("SVRPO", "Market-neutral volatility risk premia", "collateralized"),
)

#: **The index book trades SPY and is benchmarked against SPX.** SPX is no
#: longer in the universe, so these twelve are a *proxy* benchmark, and the
#: mismatch is accepted rather than assumed away. It was chosen over the two
#: alternatives: synthesising SPY-scale replications of each methodology and
#: calling those the benchmark, which would compare our replication against our
#: replication and validate nothing external; or dropping the index book, which
#: would leave the tenth name evaluated against no published strategy at all.
#: (The replications in ``RULE_ARMS`` are still built -- but as a *third* band,
#: not as the benchmark.)
#:
#: What the mismatch costs, measured over the 246-date window
#: (2024-09-03 .. 2025-08-29) by joining ``features/underlying_market_features``
#: to its ``.pre_spy_swap`` backup, which holds the same dates marked on SPX:
#:
#: * **Scale is free.** SPY is ~1/10 of SPX in notional, which is the whole
#:   reason the universe trades it, but an index level is scale-invariant: the
#:   benchmark enters as a return series, so the ratio never appears.
#: * **Daily returns agree.** Close-to-close log returns, 245 steps:
#:   ``rho = 0.9992``, tracking error 0.74%/yr, drift -5 bp/yr with SPY below
#:   SPX -- the right sign and order of magnitude for the ETF's fee. A
#:   daily-grid comparison is sound.
#: * **Half-day returns do not.** The same pair on the AM/PM decision step gives
#:   ``rho = 0.988`` and 2.4%/yr tracking error, with a worst step of 1.0%. The
#:   two series are marked at different instants from different sources and the
#:   discrepancy only cancels over the full day. Do not difference a policy arm
#:   against one of these on any grid finer than daily.
#: * **Exercise style does not cancel, and is not measured.** SPY options are
#:   American on a physically-settled ETF that pays a quarterly dividend; SPX is
#:   European and cash-settled. Every short-call methodology below (``BXM``,
#:   ``CLL``, ``CMBO``) is therefore exposed on SPY to early assignment around
#:   the ex-dividend date, which the published level cannot exhibit. This is a
#:   known unquantified bias in the benchmark, recorded so the comparison is
#:   read with it rather than without it. ``rule_covered_call`` and
#:   ``rule_collar`` are the arms that will eventually size it.
SERIES_SPX = tuple(
    _series(ticker, "index", "SPX proxy", "S&P 500", strategy, basis, pairable=True)
    for ticker, strategy, basis in EXTERNAL_SPX
)

# -- Band D: passive references -------------------------------------------

PASSIVE_ARMS = (
    _series(
        "SPX",
        "index",
        "SPX proxy",
        "S&P 500",
        "Price index, no options",
        "price return",
        pairable=False,
        band="D",
    ),
    # Sourced from Yahoo's ^SP500TR, not the Cboe CDN, which does not publish a
    # total-return SPX. This is the correct passive comparator for every
    # `total return` row in the index book; `SPX` is short the dividend yield,
    # which is the same order of magnitude as the premium these strategies
    # collect. See `scripts/analysis/cboe_strategy_index_returns.py`.
    #
    # It is also the right passive line for a SPY book despite the instrument
    # gap, and for the same reason the gap is tolerable at all: `SPXTR` accrues
    # the dividends a SPY holder actually receives, where `SPX` does not. The
    # gap between these two lines is 1.32%/yr over the 246-date window and
    # 2.09%/yr over the full 1988-2026 level file; the -5 bp/yr SPY-vs-SPX
    # drift measured at `EXTERNAL_SPX` is more than twenty times smaller. The
    # instrument mismatch is the second-order error here, and confusing `SPX`
    # for `SPXTR` is the first-order one.
    _series(
        "SPXTR",
        "index",
        "SPX proxy",
        "S&P 500",
        "Total return index, no options",
        "total return",
        pairable=True,
        band="D",
    ),
    ArmSpec(
        "rule_ew_buyhold",
        "names",
        "D",
        "ours",
        "rule",
        "9 single names",
        "Equal-weight buy-and-hold, our ledger",
        basis="total return",
        pairable=True,
    ),
    ArmSpec(
        "rf_tbill",
        "all",
        "D",
        "cash",
        "series",
        "Cash",
        "Realized 3M bill / SOFR path",
        basis="total return",
        pairable=False,
    ),
)

REGISTRY: tuple[ArmSpec, ...] = (
    POLICY_ARMS + RULE_ARMS + SERIES_MATCHED + COMPOSITE_ARMS + SERIES_SPX + PASSIVE_ARMS
)


def by_book(book: str) -> tuple[ArmSpec, ...]:
    return tuple(spec for spec in REGISTRY if spec.book == book)


def by_band(band: str) -> tuple[ArmSpec, ...]:
    return tuple(spec for spec in REGISTRY if spec.band == band)


def lookup(name: str) -> ArmSpec | None:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    return None
