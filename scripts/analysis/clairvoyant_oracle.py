"""The most this environment can pay, to a policy that already knows the future.

``project_algo_existence_proof.md`` establishes a *lower* bound: some rule beats
``hold``.  It says nothing about the ceiling, and the two questions want opposite
next steps.  If the best achievable is +0.20 then a policy at +0.15 is nearly
done and the remaining work is variance reduction; if it is +8.0 then +0.15 is
noise and the action space, not the policy, is what to argue about.  GRPO needs
that number before it is worth running, because an advantage estimated against
an unknown ceiling cannot say whether a group is good or merely less bad.

**What "oracle" means here.**  Not a better signal.  The oracle is given the
realized path and asked the purely combinatorial question the environment
already answers: of the packages this chain will actually resolve, at the fills
this cost model will actually charge, which non-overlapping set is worth most?
Every candidate it considers is one the wire grammar can emit and the resolver
can fill.  So the bound is *achievable*, not notional -- it is the value of
perfect foresight over the real menu, which is exactly the quantity a learned
policy is trying to approach and can never exceed.

**Why it must not be run on a broken mark path.**  An oracle maximizes over
model error as eagerly as over signal, so any valuation defect becomes the
answer.  On the pre-2026-09-15 chain it would have reported the AAPL 220/222.5/225
put butterfly at +$1,188,243 -- a structure whose maximum possible profit was
$74,167 -- and called that the ceiling.  That cuts both ways and is the second
reason this script exists: an oracle is the sharpest corruption detector
available, because it searches directly for the largest number the pricing path
can be made to emit.  ``--audit-bounds`` therefore checks every candidate
against ``value_bounds`` and refuses to report a total if any of them clears it.

**What the bound is loose about**, stated here so no reader has to infer it:

* *Dollars are additive across concurrent positions; log return is not.*  The
  selection maximizes summed dollar PnL against a fixed reference NAV, then the
  log return is reported on the total.  Compounding within the window would
  only help, so this understates -- it stays an upper bound.
* *One exit per position, on the decision grid.*  The oracle may not leg out of
  a package or re-enter the same structure intraday.  That matches the wire
  grammar, which has no partial close.
* *Sizing is the environment's own -- literally, since 2026-09-21.*  Every
  position is sized by calling ``SizeResolver``, the same object the policy
  gets, so the oracle cannot win by leverage.  This is the one place the bound
  is deliberately *not* maximal: a bound that could size freely would be about
  the risk budget rather than about the actions.

  Until 2026-09-21 this paragraph was false in a way that mattered.  The script
  carried its own ``quantity_for()``, a max-loss ceiling
  ``floor(cap_for(family)*nav / max_loss)``, which was the environment's rule
  when it was written and stopped being so when sizing moved to the
  Taylor-expansion scenario budget.  Every ceiling reported before that date is
  sized by the max-loss rule alone, so it is **not comparable** to one reported
  after, and the gap is not small.  Measured on 2024-09-03 .. 2024-09-13 PM,
  same trips (1,704 in all three runs), attainable profit:

      max-loss only (the old rule)   $1,411,006
      scenario_max_loss (default)    $  692,957    2.04x lower
      scenario (no ceiling)          $  804,448

  So every pre-2026-09-21 oracle number is roughly **twice** what the same
  window bounds under the sizing the policy actually gets.  That reproduction
  recipe is **EXPIRED 2026-09-22**: the max-loss ceiling was removed from the
  environment entirely, so ``scenario_max_loss`` is not a rule any more and
  there is no setting of the surviving knobs that re-creates it.  The numbers
  above stay on the record as history; do not plot them against anything
  produced after that date.

  Sizing is against a **flat book**, which is a deliberate looseness and the
  reason the selection stays a flow problem.  A limit that reads book state
  would make a candidate's size depend on which other candidates were chosen,
  turning the selection into a knapsack with coupled weights.  That looseness
  is no longer free.  It was free under the old default, whose two limits --
  the scenario budget and the max-loss ceiling -- neither touched the book.
  Every surviving rule carries ``cash``, which does, and since the ceiling went
  ``cash`` is the only thing bounding what one package may commit.  So the
  flat-book relaxation is now a genuine overstatement under *every* rule:
  ``cash`` (and, under ``full``, ``name``, ``total`` and ``delta``) sees full
  headroom here and would not in a real run.

* *Under ``--hedge`` both ends of the bracket are hedged, by two different
  hedgers on purpose.*  Until 2026-09-23 the search was unhedged and the hedge
  was measured afterwards; the headline was then an unhedged number, which is
  not the number the policy is scored on.

  The **upper end** scores every arc through the real ``HedgeResolver``,
  per position, at a NAV frozen at ``--nav`` (``hedged_best_exits``).  Both
  concessions are what make it exact: per-position hedging nets with nothing, so
  a candidate's hedged profit is a property of the candidate alone, and a frozen
  band is a constant, so an arc cost is fixed before selection and
  ``select_flow``'s optimum is a real bound over the relaxation.

  The **attainable end** is greedy under every rule, replayed through the real
  hedger as it actually runs -- portfolio-level, on the running NAV, share
  balance carried (``replay_hedged``) -- which is rung H2.

  **Neither end checks the other.**  Portfolio-level hedging nets across a name
  and is therefore *cheaper* than per-position hedging, so ``net_profit`` can
  legitimately land above ``total_profit``.  Separately, hedge PnL is not
  sign-definite, so on a mean-reverting path a hedged package can gamma-scalp
  past its own best single exit.  Only ``hedge_transaction_cost`` is signed.

  With ``--hedge none`` nothing above applies: every candidate goes through the
  plain ``best_exit`` and the report is the unhedged one it always was.
* *Only ``O`` heads are enumerated, and that costs the bound nothing.*
  ``execution.open`` charges a roll's replacement exactly what it charges a
  plain open -- ``replaces`` touches only the three lineage fields -- so an
  ``X`` is economically a ``C`` plus an ``O`` and the trip decomposition
  already spans it.  The one place the verb is not neutral is the per-step
  order budget, where it saves a line; see ``_lines``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from heapq import heappop, heappush
from math import log, sqrt
from pathlib import Path
from statistics import fmean, stdev

from portfolio_monkey.env.actions import OpenOrder, parse_action
from portfolio_monkey.env.book import Book, Leg, Position
from portfolio_monkey.env.chain import ChainSlice, NoChainData, OptionChain
from portfolio_monkey.env.datasets import data_root
from portfolio_monkey.env.environment import build_grid
from portfolio_monkey.env.execution import ExecutionModel
from portfolio_monkey.env.payoff import PayoffLeg, value_bounds
from portfolio_monkey.env.resolvers.contract import ContractResolver, ResolvedPackage
from portfolio_monkey.env.resolvers.hedge import HedgeResolver
from portfolio_monkey.env.resolvers.market import MarketResolver
from portfolio_monkey.env.resolvers.size import SizeDecision, SizeResolver
from portfolio_monkey.env.spec import (
    BAND_RULES,
    SIZE_RULE_LIMITS,
    SIZE_RULES,
    CostModel,
    EnvConfig,
    FeatureFlags,
    GridSpec,
    HedgeSpec,
    ResolverBounds,
    SizeBounds,
)
from portfolio_monkey.env.spreads import SpreadTable, SpreadTableError

# Imported rather than re-implemented, for the reason ``enumerate_orders`` runs
# wire strings through the parser: ``--hedge all`` has to mean the same set of
# families here as it does in the run this bound is compared against, and a
# second copy of the mapping is a copy that can drift.
from portfolio_monkey.jobs.run_policy_episodes import _hedged_families

#: The liquidity gates ``--lift-quote-gates`` can switch off, in report order.
#: Deliberately does not include the deliverable or multiplier checks: those
#: say what the contract *is*, not how hard it is to trade, and lifting them
#: would corrupt prices rather than widen the menu.
LIFTABLE_QUOTE_GATES: tuple[str, ...] = ("spread", "age", "one_sided")

#: Every (family, orientation) the grammar admits.  Taken from the code tables
#: rather than retyped, so a family added to ``FAMILY_CODES`` enters the oracle's
#: menu automatically and cannot be silently left out of the ceiling.
HEADS: tuple[str, ...] = (
    "ol b", "ol r",
    "dv b", "dv r",
    "cv b", "cv r",
    "dg b", "dg r",
    "bf b", "bf r",
    "ls n", "lg n", "ib n", "ic n",
)

# What a roll may *not* change, and so what a close and an open must share for
# the two of them to be one ``X`` rather than two order lines.
RollKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class OracleLeg:
    """A leg flattened to the numbers the forward pass needs.

    Deliberately not a ``Leg``: the forward pass touches this object millions of
    times and re-deriving ``multiplier`` or ``entry_half_spread`` through the
    book's accessors is the difference between a job that finishes and one that
    does not.
    """

    contract_id: str
    right: str
    strike: float
    ratio: int
    multiplier: int
    entry_mid: float
    entry_half_spread: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """One package the policy could have opened at one decision point."""

    underlying: str
    head: str
    tenor: str
    open_index: int
    expiry: date
    legs: tuple[OracleLeg, ...]
    #: Signed, positive is a net debit paid.  Equals the package's value at
    #: entry, which is why the round trip at mid is ``exit_mark - entry_cost``.
    entry_cost: float
    entry_half_spread: float
    max_loss: float
    #: ``head`` split back into the two fields ``Position`` carries, taken off
    #: the ``OpenOrder`` rather than re-parsed from the head string.  Only the
    #: hedged replay reads them -- ``HedgeResolver`` filters on
    #: ``position.family`` -- but they are carried on every candidate so that
    #: the unhedged and hedged passes cannot be looking at different menus.
    family: str = ""
    orientation: str = ""
    strategy_handle: str = ""

    @property
    def contracts(self) -> int:
        return sum(abs(leg.ratio) for leg in self.legs)

    def payoff_legs(self) -> list[PayoffLeg]:
        return [
            PayoffLeg(leg.right, leg.strike, leg.ratio, leg.entry_mid, leg.multiplier)
            for leg in self.legs
        ]


@dataclass(frozen=True, slots=True)
class Trip:
    """A candidate together with the exit that was best for it."""

    candidate: Candidate
    close_index: int
    quantity: int
    #: **The objective the schedulers maximise.**  After entry spread, exit
    #: spread and both legs of per-contract fees -- and, when the run hedges this
    #: family, after the package's own per-position share leg as well.  So under
    #: ``--hedge`` this is not an option number and must not be compared with one
    #: from an unhedged run.
    profit: float
    #: The same trip's option leg alone, always, hedged run or not.  Carried
    #: separately because ``replay_hedged``'s tie-out reconstructs option PnL
    #: from its own cash accounting: checked against ``profit`` under a hedged
    #: objective it would be comparing an option number with an option-plus-share
    #: one and would fire on every run.
    option_profit: float
    exit_mark: float
    #: The package's half-spread at the chosen exit, per package and before
    #: ``half_spread_multiplier``.  Stored rather than recomputed because the
    #: hedged replay has to charge the *same* exit cost the profit was scored
    #: with, or the two passes stop being the same schedule.
    exit_half_spread: float = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, type=_as_date)
    parser.add_argument("--end", required=True, type=_as_date)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--decision-sessions", default="PM")
    parser.add_argument("--nav", type=float, default=1_000_000.0)
    parser.add_argument(
        "--risk-free-annual",
        type=float,
        default=0.0,
        help=(
            "the Sharpe and Sortino hurdle. Defaults to 0, which is a choice and "
            "not a neutral one: over a 3-month window at a 4%% cash rate it "
            "flatters the ratios by about 1%% of NAV of foregone interest. It is "
            "recorded in the report so two arms are never compared across "
            "different hurdles"
        ),
    )
    parser.add_argument(
        "--half-spread-multiplier",
        type=float,
        default=1.0,
        help="the fill convention, as in the algorithmic sweep. 1.0 is honest",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help=(
            "concurrent position cap. Defaults to SizeSpec.max_positions (12). "
            "This is the binding resource: the risk budget at 12 x 1%% of NAV is "
            "12%% against a 30%% cap, so slots run out long before dollars do"
        ),
    )
    # -- sizing toggles ----------------------------------------------------
    #
    # Defaults come from ``SizeBounds()`` instances rather than class attributes:
    # the dataclass is ``slots=True``, so the class attribute is a
    # ``member_descriptor`` and not the default value, and it would reach
    # ``SizeBounds(...)`` intact.
    parser.add_argument(
        "--size-rule",
        choices=SIZE_RULES,
        default=SizeBounds().size_rule,
        help=(
            "which limits the size is a minimum over. 'nav_fraction' is the "
            "default and the toggle-OFF arm; 'scenario' is the same rule plus "
            "the one-day Taylor-expansion risk budget, i.e. the toggle-ON arm. "
            "'full' is the ablation, and since the max-loss ceiling was removed "
            "on 2026-09-22 it is the only rule that still bounds ultimate risk"
        ),
    )
    parser.add_argument(
        "--target-scenario-risk",
        type=float,
        default=SizeBounds().target_scenario_risk,
        help="budgeted cost of a one-day one-sigma adverse move, as a fraction of NAV",
    )
    parser.add_argument(
        "--nav-fraction",
        type=float,
        default=SizeBounds().nav_fraction,
        help=(
            "gross premium one position may commit, as a fraction of NAV. The "
            "primary sizing unit under the default 'nav_fraction' rule and a "
            "live ceiling under 'scenario' and 'full'. Live under every rule "
            "since the max-loss ceiling was removed (2026-09-22): it is what "
            "bounds ultimate risk now, at f x (max_loss / gross_premium) x NAV "
            "per position, and that ratio is unbounded above"
        ),
    )
    parser.add_argument(
        "--vol-shock",
        type=float,
        default=SizeBounds().vol_shock_relative,
        help="relative shock to the package's own IV; the only term reaching vega",
    )
    # ``--max-loss-cap`` was removed 2026-09-22 with the ceiling itself.  It is
    # deliberately not accepted-and-ignored: argparse will reject it, so a stale
    # sbatch or arms row fails loudly instead of running at a risk setting the
    # caller believes is in force and is not.
    parser.add_argument(
        "--audit-bounds",
        action="store_true",
        default=True,
        help=(
            "check every priced exit against the package's structural value "
            "range and refuse to report if any clears it. On by default: an "
            "oracle run on a broken mark path reports the defect as the ceiling"
        ),
    )
    parser.add_argument("--no-audit-bounds", dest="audit_bounds", action="store_false")
    parser.add_argument(
        "--chain-cache-dates",
        type=int,
        default=0,
        help="0 sizes the cache to the window; see the sweep's flag of the same name",
    )
    parser.add_argument(
        "--lift-quote-gates",
        default="none",
        help=(
            "comma-separated subset of {spread, age, one_sided} to lift, or "
            "'all' / 'none' (default). These are the liquidity gates that keep "
            "a contract out of the menu, so lifting them measures the "
            "environment rather than the filters. Take them ONE AT A TIME: "
            "lifting all three at once moves the ceiling by more than the menu "
            "widens, and the cause is not attributable from the total. Does "
            "NOT lift the deliverable/multiplier check, which is a correctness "
            "gate, not a liquidity one"
        ),
    )
    # -- the hedged replay (rung H2) ---------------------------------------
    #
    # Deliberately *not* wired into the search.  These flags change what is
    # measured about the selected schedule, never which schedule is selected;
    # a run with and without them picks the same trips, which is what makes
    # the hedge cost attributable.
    parser.add_argument(
        "--hedge",
        default="none",
        help=(
            "replay the selected schedule through the real HedgeResolver. "
            "'none' (default), 'all', 'volatility', or a comma-separated family "
            "list -- the same vocabulary as run_policy_episodes. The result is "
            "achievable, not a bound: the schedule was chosen unhedged"
        ),
    )
    parser.add_argument(
        "--delta-band", type=float, default=HedgeSpec().delta_band,
        help="half-width of the no-trade region as a fraction of NAV",
    )
    parser.add_argument(
        "--band-rule", choices=BAND_RULES, default=HedgeSpec().band_rule,
        help="'fixed' or 'whalley_wilmott'; the latter needs --spread-table",
    )
    parser.add_argument("--risk-aversion", type=float, default=HedgeSpec().risk_aversion)
    parser.add_argument("--band-multiple", type=float, default=HedgeSpec().band_multiple)
    parser.add_argument(
        "--min-band-fraction", type=float, default=HedgeSpec().min_band_fraction
    )
    parser.add_argument(
        "--max-band-fraction", type=float, default=HedgeSpec().max_band_fraction
    )
    parser.add_argument(
        "--fallback-half-spread", type=float, default=HedgeSpec().fallback_half_spread
    )
    parser.add_argument(
        "--spread-table", type=Path, default=None,
        help="per-(date, ticker) session half-spreads; see build_underlying_spreads.py",
    )
    parser.add_argument(
        "--measured-spread-costs", action="store_true",
        help="charge the share fill the table's spread instead of the flat bps",
    )
    parser.add_argument(
        "--dump-candidates",
        type=Path,
        default=None,
        help="also write every candidate's best-exit outcome as JSONL, one row "
             "per (step, underlying, family, tenor, orientation). This is the "
             "supervised label source for the Band D baselines: the report's "
             "by_family totals are portfolio aggregates and cannot say which "
             "family was best for one name on one day. Rows are written BEFORE "
             "the profit>0 filter, so a name whose every family loses is "
             "representable as 'no trade' rather than as an absent row.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def enumerate_orders(
    underlying: str, tenors: Sequence[str], config: EnvConfig
) -> Iterator[tuple[str, OpenOrder]]:
    """The full menu at one name: every head at every tenor.

    Built by running real wire strings through ``parse_action`` rather than by
    constructing ``OpenOrder`` directly.  That is not fastidiousness: the
    parser is what fills in default coordinates, snaps them to the admitted
    grid and rejects topologies the registry does not admit, so an order built
    around it would be one the policy could not actually have emitted and the
    bound would stop being achievable.  It also means a family whose defaults
    change cannot leave a stale copy of them in this file.

    Coordinates are left at those defaults rather than swept.  They are a
    continuum, so sweeping them would make the menu unbounded and the answer
    would stop being about the *action space* -- which is what the policy
    chooses over -- and start being about strike optimization.  The bound is
    therefore over heads and tenors, and is loose by whatever the coordinates
    are worth.
    """
    for head in HEADS:
        for tenor in tenors:
            wire = f"O {underlying} {head} {tenor}"
            parsed = parse_action(wire, config)
            for order in parsed.orders:
                if isinstance(order, OpenOrder):
                    yield head, order


def to_candidate(
    package: ResolvedPackage,
    *,
    underlying: str,
    head: str,
    tenor: str,
    open_index: int,
    order: OpenOrder | None = None,
) -> Candidate:
    return Candidate(
        underlying=underlying,
        head=head,
        tenor=tenor,
        open_index=open_index,
        family=order.family if order is not None else "",
        orientation=order.orientation if order is not None else "",
        strategy_handle=order.strategy_handle if order is not None else "",
        expiry=package.expiry,
        legs=tuple(
            OracleLeg(
                contract_id=leg.contract_id,
                right=leg.quote.right,
                strike=leg.quote.strike,
                ratio=leg.ratio,
                multiplier=leg.quote.multiplier,
                entry_mid=leg.quote.mid,
                entry_half_spread=leg.quote.half_spread,
            )
            for leg in package.legs
        ),
        entry_cost=package.mid_cost,
        entry_half_spread=package.half_spread_cost,
        max_loss=package.max_loss,
    )


def price_book(chain_slice: ChainSlice | None) -> dict[str, tuple[float, float]]:
    """``contract_id -> (mid, half_spread)`` for everything the slice can value.

    ``markable`` is folded in because a package the book can hold is one the
    book must be able to value; see ``MARK_ONLY_REASONS``.  ``quotes`` is
    applied second so a tradeable row wins any collision, though by construction
    there are none.
    """
    if chain_slice is None:
        return {}
    prices: dict[str, tuple[float, float]] = {
        q.contract_id: (q.mid, q.half_spread) for q in chain_slice.markable
    }
    prices.update({q.contract_id: (q.mid, q.half_spread) for q in chain_slice.quotes})
    return prices


def quantity_for(
    package: ResolvedPackage,
    *,
    sizer: SizeResolver,
    book: Book,
    as_of: datetime,
) -> int:
    """The environment's own sizing, at a flat book -- by calling it, not copying it.

    ``sizer`` is a real ``SizeResolver`` and ``book`` is empty, so every limit
    that reads book state sees full headroom.  That is the flat-book relaxation
    the module docstring describes, and it is what keeps the selection a flow
    problem: a size that depended on which other candidates were chosen would
    make this a knapsack with coupled weights.  The relaxation used to be exact
    under the default, whose limits were the scenario budget and the max-loss
    ceiling, neither of which reads the book.  Since the ceiling was removed
    (2026-09-22) every admitted rule carries ``cash``, so the relaxation is
    approximate under all of them and the bound is loose by whatever ``cash``
    would have taken off a full book.

    The count gates never fire here for the same reason -- an empty book is
    never at ``max_positions`` -- which is correct: the concurrent-position cap
    is enforced in the selection, where it can see the whole schedule.

    Returns 0 on refusal, which the caller treats as "not a candidate".  The
    common refusal is a package whose smallest possible size, one lot, already
    breaks a limit; that is a real answer, not an error.
    """
    decision = sizer.resolve(package, book, as_of=as_of)
    if not isinstance(decision, SizeDecision):
        return 0
    return decision.quantity


def parse_sessions(raw: str) -> tuple[str, ...]:
    """The decision sessions named on the command line.

    ``|`` is accepted as well as ``,`` because Slurm's ``--export`` splits its
    own argument on commas: ``--export=ALL,PM_ORACLE_DECISION_SESSIONS=AM,PM``
    delivers ``AM`` and turns ``PM`` into a separate, meaningless export.  The
    sbatch has always documented ``|`` as the way round that, but this only ever
    split on ``,``, so ``AM|PM`` reached ``GridSpec`` as one token and raised --
    which meant the AM+PM grid, the one the environment actually runs, could not
    be requested from a batch job at all.  It failed loudly rather than quietly
    running PM, which is the only reason it was merely a blocker.

    A function rather than an expression inside ``main`` so that the rule can be
    tested; it was wrong for as long as it was untestable.
    """
    return tuple(
        s.strip().upper() for s in raw.replace("|", ",").split(",") if s.strip()
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sessions = parse_sessions(args.decision_sessions)
    # ``max_positions`` goes into ``SizeBounds`` and not only into the selection
    # cap, so that ``--max-positions`` is subject to the same validation the
    # policy runner is.  Since 2026-09-22 that check is no longer the a-priori
    # ``max_positions * cap * (1 + buffer)`` solvency bound -- there is no cap to
    # multiply -- but the refusal of any rule without a ``cash`` limit, which is
    # what now keeps the book able to fund its own fills.
    size = SizeBounds(
        size_rule=args.size_rule,
        max_positions=args.max_positions or SizeBounds().max_positions,
        target_scenario_risk=args.target_scenario_risk,
        vol_shock_relative=args.vol_shock,
        nav_fraction=args.nav_fraction,
    )
    try:
        hedged_families = _hedged_families(args.hedge)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.band_rule == "whalley_wilmott" and args.spread_table is None:
        # The same refusal the policy runner makes.  Without a table the band
        # falls back to a flat half-spread, which makes the WW arm the fixed
        # arm wearing a different name -- and the report would say otherwise.
        print(
            "--band-rule whalley_wilmott requires --spread-table: with no table "
            "the band is computed off a flat fallback spread and the arm is the "
            "fixed rule under another name",
            file=sys.stderr,
        )
        return 2
    if args.measured_spread_costs and args.spread_table is None:
        print(
            "--measured-spread-costs requires --spread-table", file=sys.stderr
        )
        return 2
    spreads: SpreadTable | None = None
    if args.spread_table is not None:
        try:
            spreads = SpreadTable.from_csv(
                args.spread_table, fallback=args.fallback_half_spread
            )
        except SpreadTableError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    # ``--lift-quote-gates`` lifts the gates that reject a contract for being
    # *illiquid*, and only those.  A wide package, an old quote and a one-sided
    # quote are all statements about how hard the fill would be -- and the fill
    # is already charged for separately, at the half-spread, so gating on them
    # too both filters and charges for the same fact.
    #
    # That argument is sound for ``spread`` and ``one_sided``, where the wider
    # half-spread does price the illiquidity back in.  It is *not* sound for
    # ``age``: a stale quote is not an expensive price, it is a price from
    # another time, and the half-spread charges nothing for that.  ``age`` is
    # kept liftable because the standing rule is that freshness is a label and
    # never a filter, but a ceiling measured with it lifted is a ceiling over
    # prices that were not available at the decision.  Lift it to size the
    # effect, not to quote the result.
    #
    # ``require_standard_deliverable`` and ``expected_multiplier`` stay on.
    # They are not liquidity gates: they say the row's contract delivers 100
    # ordinary shares.  Every dollar figure downstream multiplies by that
    # multiplier, so admitting a row whose multiplier is wrong or unknown does
    # not widen the menu, it corrupts the prices on it -- see the NVDA split
    # note at ``chain.py:526``.
    raw_gates = {g.strip().lower() for g in args.lift_quote_gates.split(",") if g.strip()}
    if raw_gates == {"all"}:
        raw_gates = set(LIFTABLE_QUOTE_GATES)
    elif raw_gates <= {"none", ""}:
        raw_gates = set()
    if not raw_gates <= set(LIFTABLE_QUOTE_GATES):
        unknown = sorted(raw_gates - set(LIFTABLE_QUOTE_GATES))
        print(
            f"--lift-quote-gates: unknown gate(s) {', '.join(unknown)};"
            f" choose from {', '.join(LIFTABLE_QUOTE_GATES)}, or 'all'/'none'",
            file=sys.stderr,
        )
        return 2
    lifted = tuple(g for g in LIFTABLE_QUOTE_GATES if g in raw_gates)

    bounds = ResolverBounds()
    if "spread" in lifted:
        bounds = replace(bounds, max_relative_spread_package=float("inf"))
    if "age" in lifted:
        bounds = replace(bounds, max_quote_age_seconds=float("inf"))
    if "one_sided" in lifted:
        bounds = replace(bounds, require_two_sided_quote=False)

    config = EnvConfig(
        grid=GridSpec(decision_sessions=sessions),
        cost=CostModel(half_spread_multiplier=args.half_spread_multiplier),
        resolver=bounds,
        size=size,
        # ``enabled`` follows the family list rather than being a second switch,
        # for the reason ``build_config`` gives: two flags that can disagree do
        # so silently, and a report claiming ``hedged: true`` on a run that
        # hedged nothing is worse than one that never hedged.
        hedge=HedgeSpec(
            enabled=bool(hedged_families),
            hedged_families=hedged_families,
            delta_band=args.delta_band,
            band_rule=args.band_rule,
            risk_aversion=args.risk_aversion,
            band_multiple=args.band_multiple,
            min_band_fraction=args.min_band_fraction,
            max_band_fraction=args.max_band_fraction,
            fallback_half_spread=args.fallback_half_spread,
        ),
        flags=FeatureFlags(measured_spread_costs=args.measured_spread_costs),
    )
    cap = args.max_positions or config.size.max_positions
    sizer = SizeResolver(config)
    # One book, never mutated: every candidate is sized as if it were the first
    # position opened.  Shared rather than rebuilt per candidate because the
    # forward pass touches this path once per (step, name, head, tenor).
    flat_book = Book(cash=args.nav, initial_cash=args.nav)
    fee = config.cost.option_fee_per_contract
    hs_mult = config.cost.half_spread_multiplier

    root = data_root(args.data_root)
    probe = OptionChain(config.resolver, root=root)
    dates = [d for d in probe.coverage if args.start <= d <= args.end]
    if len(dates) < 2:
        print(f"only {len(dates)} chain date(s) in {root}", file=sys.stderr)
        return 1
    chain = OptionChain(
        config.resolver, root=root, cache_size=args.chain_cache_dates or 2 * len(dates) + 4
    )
    # Decision points only, for closes as much as for opens: a close is an
    # action, so an oracle allowed to exit at a non-decision mark would be
    # bounding a game the policy is not playing.  With ``--decision-sessions PM``
    # this also halves the forward pass, and costs nothing -- the AM slice
    # admits zero quotes for every name on every date in the window.
    grid = tuple(p for p in build_grid(dates, config) if p.is_decision)
    resolver = ContractResolver(config)
    names = config.universe.tradeable
    tenors = config.tenor.admitted

    if not args.quiet:
        print(f"window   {dates[0]} .. {dates[-1]}  ({len(dates)} dates, {len(grid)} steps)")
        print(f"menu     {len(names)} names x {len(HEADS)} heads x {len(tenors)} tenors"
              f" = {len(names) * len(HEADS) * len(tenors)} intents/step")
        print(f"cap      {cap} concurrent positions, sized by {config.size.size_rule}"
              f" (f={config.size.nav_fraction:.1%} of NAV in premium each)")

    # -- pass 1: price every contract at every step ------------------------
    #
    # Built once and held, because the forward pass reads each step once per
    # live candidate and re-slicing the chain there would make the job
    # quadratic in chain I/O rather than in arithmetic.
    prices: list[dict[str, tuple[float, float]]] = []
    spots: list[dict[str, float]] = []
    for index, point in enumerate(grid):
        step_prices: dict[str, tuple[float, float]] = {}
        step_spots: dict[str, float] = {}
        for name in names:
            # ``NoChainData`` only.  A bare ``except`` here returns a confident
            # zero for any bug in the call, which is what it did on the first
            # run of this script: the grid point's field is ``timestamp``, the
            # AttributeError was swallowed 2,460 times, and the oracle reported
            # a ceiling of exactly +0 without a single error on stderr.
            try:
                sl = chain.slice_for(
                    name,
                    trade_date=point.trade_date,
                    session=point.session,
                    decision_time=point.timestamp,
                )
            except NoChainData:
                continue
            step_prices.update(price_book(sl))
            step_spots[name] = sl.underlying_price
        prices.append(step_prices)
        spots.append(step_spots)
        if not args.quiet and index % 25 == 0:
            print(f"  priced step {index}/{len(grid)}  {point.trade_date} {point.session}"
                  f"  {len(step_prices)} contracts", flush=True)

    # -- pass 2: enumerate the menu and find each candidate's best exit ----
    #
    # Two routes out of here, decided per candidate by whether this run hedges
    # its family.  An unhedged family carries no share leg at all, so its best
    # exit is a forward walk over prices alone and nothing is gained by putting
    # it through the hedged machinery.  A hedged family's exit has to be chosen
    # on hedged profit, which is path-dependent and therefore step-major -- see
    # ``hedged_best_exits``.  The split is on family, not on convenience: the two
    # routes must agree on everything except the share leg, which is why they
    # share ``package_mark``, ``out_of_bounds`` and ``option_leg_profit``.
    trips: list[Trip] = []
    #: Every best-exit outcome, losers included, when --dump-candidates is on.
    #: ``None`` otherwise so a normal run pays neither the memory nor the risk
    #: of a second list drifting out of step with ``trips``.
    dumped: list[Trip] | None = [] if args.dump_candidates else None
    hedged_opens: dict[int, list[tuple[Candidate, int]]] = {}
    resolved = failed = 0
    violations: list[dict] = []
    for index, point in enumerate(grid):
        if index == len(grid) - 1:
            break  # nothing opened at the last step can be closed
        for name in names:
            try:
                sl = chain.slice_for(
                    name,
                    trade_date=point.trade_date,
                    session=point.session,
                    decision_time=point.timestamp,
                )
            except NoChainData:
                continue
            for head, order in enumerate_orders(name, tenors, config):
                package = resolver.resolve(order, sl)
                if not isinstance(package, ResolvedPackage):
                    failed += 1
                    continue
                resolved += 1
                candidate = to_candidate(
                    package, underlying=name, head=head, tenor=order.tenor_bucket,
                    open_index=index, order=order,
                )
                quantity = quantity_for(
                    package, sizer=sizer, book=flat_book, as_of=point.timestamp
                )
                if quantity <= 0:
                    continue
                if (
                    config.hedge.enabled
                    and candidate.family in config.hedge.hedged_families
                    and config.hedge.hedge_ticker_for(candidate.underlying) is not None
                ):
                    hedged_opens.setdefault(index, []).append((candidate, quantity))
                    continue
                trip = best_exit(
                    candidate,
                    quantity=quantity,
                    grid=grid,
                    prices=prices,
                    fee=fee,
                    hs_mult=hs_mult,
                    audit=args.audit_bounds,
                    violations=violations,
                )
                if trip is not None:
                    if dumped is not None:
                        # Recorded *before* the sign filter, deliberately: the
                        # supervised label is "the best family, or none if none
                        # of them clears zero after cost", and a list that has
                        # already dropped the losers cannot answer the second
                        # half of that.
                        dumped.append(trip)
                    if trip.profit > 0.0:
                        trips.append(trip)
        if not args.quiet and index % 25 == 0:
            print(f"  walked step {index}/{len(grid)}  {resolved} resolved,"
                  f" {len(trips)} profitable trips so far", flush=True)

    walk_stats: dict = {}
    if hedged_opens:
        queued = sum(len(v) for v in hedged_opens.values())
        if not args.quiet:
            print(f"\nwalking {queued} hedged candidates per-position at frozen NAV"
                  f" {args.nav:,.0f}", flush=True)
        # ``portfolio_level=False`` is a no-op on the one-position books the walk
        # hedges -- ``_per_position`` on a single position returns that position
        # -- and it is set anyway, because the claim this end of the bracket
        # makes is that each arc is hedged *alone*, and a resolver configured to
        # net across a name would make that claim by accident rather than by
        # construction.
        per_position = replace(
            config, hedge=replace(config.hedge, portfolio_level=False)
        )
        hedged_trips, walk_stats = hedged_best_exits(
            hedged_opens,
            grid=grid,
            prices=prices,
            spots=spots,
            config=per_position,
            chain=chain,
            nav=args.nav,
            spreads=spreads,
            fee=fee,
            hs_mult=hs_mult,
            audit=args.audit_bounds,
            violations=violations,
            quiet=args.quiet,
        )
        if dumped is not None:
            dumped.extend(hedged_trips)
        trips.extend(t for t in hedged_trips if t.profit > 0.0)
        # ``select_greedy`` sorts by profit and Python's sort is stable, so the
        # order trips arrive in decides ties. Appending the hedged walk's output
        # after the plain walk's would otherwise make the schedule depend on
        # which families the run happens to hedge, for reasons that have nothing
        # to do with profit. Restored to enumeration order.
        trips.sort(
            key=lambda t: (
                t.candidate.open_index,
                t.candidate.underlying,
                t.candidate.head,
                t.candidate.tenor,
            )
        )

    # The answer is a bracket, not a number, and saying so is the honest form.
    #
    # The upper end is the min-cost flow: exact, but over a relaxation, because a
    # flow cannot see the name on a position and so cannot honour the per-name
    # cap.  Its optimum over a larger action set is therefore an upper bound on
    # the environment's own optimum.
    #
    # The lower end is greedy under *every* rule including the per-name cap.  It
    # is a schedule the environment would genuinely have permitted, so the true
    # ceiling cannot be below it.
    per_name = config.size.max_positions_per_underlying
    per_step = config.max_orders_per_step
    relaxed = select_flow(trips, cap=cap, steps=len(grid))
    upper = sum(t.profit for t in relaxed)
    selected = select_greedy(
        trips,
        cap=cap,
        steps=len(grid),
        per_underlying=per_name,
        per_step=per_step,
    )
    total = sum(t.profit for t in selected)
    if upper < total - 1e-6:
        # The flow optimises over a superset of what greedy searches, so this is
        # not a tie that went the wrong way — it means the flow is broken.
        print(
            f"min-cost flow returned {upper:+,.2f} on a relaxation of the problem"
            f" greedy solved for {total:+,.2f} — the flow is wrong, not conservative",
            file=sys.stderr,
        )
        return 1
    crowded = per_underlying_excess(relaxed, limit=per_name)
    busy_relaxed = orders_per_step_excess(relaxed, limit=per_step)
    busy_selected = orders_per_step_excess(selected, limit=per_step)
    if busy_selected:
        # Greedy enforces this, so a breach here is a bug in the enforcement and
        # the lower end of the bracket is not attainable. Refusing beats
        # printing a number that has to be taken on trust.
        print(
            f"the attainable schedule needs more than {per_step} orders at"
            f" {len(busy_selected)} steps — greedy did not enforce its own cap",
            file=sys.stderr,
        )
        return 1

    # -- pass 3 (optional): replay the attainable schedule under the hedger --
    hedged: dict | None = None
    if config.hedge.enabled:
        if not args.quiet:
            print(f"\nreplaying {len(selected)} trips under --hedge {args.hedge}"
                  f"  band {config.hedge.band_rule} @ {config.hedge.delta_band:.2%}")
        hedged = replay_hedged(
            selected,
            grid=grid,
            prices=prices,
            spots=spots,
            config=config,
            chain=chain,
            nav=args.nav,
            spreads=spreads,
            quiet=args.quiet,
        )
        # The tie-out. The replay reconstructs the schedule's option PnL from
        # its own cash accounting; if that does not reproduce the number the
        # search scored, the two passes are not running the same schedule and
        # the hedge figure is a difference between two different things.
        drift = hedged["option_profit"] - hedged["option_profit_expected"]
        if abs(drift) > 1.0:
            print(
                f"the hedged replay priced the same schedule at"
                f" {hedged['option_profit']:+,.2f} against the search's"
                f" {hedged['option_profit_expected']:+,.2f} — the replay is not"
                f" running the selected schedule",
                file=sys.stderr,
            )
            return 1

    # The series is a series, not a hedge setting, so it comes out of the block
    # before that block is spread into the report.
    curve = (hedged or {}).pop("nav_curve", [])
    opening_nav = (hedged or {}).pop("initial_nav", args.nav)
    # Same argument: a per-trip record is a record, not a hedge setting.  It is
    # also the only place the *replayed* hedge becomes per-trade, so it leaves
    # the block here and re-enters below as ``performance.replayed``.
    per_trip = (hedged or {}).pop("per_trip", [])
    attribution = (hedged or {}).pop("attribution", {})
    _, levels = daily_closes(curve, initial_nav=opening_nav)
    # Computed on the *attainable* schedule only. The relaxed schedule breaches
    # the per-name cap, so its drawdown and Sharpe would describe a policy the
    # environment would have refused to run -- and unlike a profit total, a risk
    # ratio on an infeasible path is not a bound on anything.
    performance = {
        "schedule": "attainable_greedy_hedged_replay",
        **performance_metrics(
            levels,
            trading_days_per_year=config.size.trading_days_per_year,
            risk_free_annual=args.risk_free_annual,
        ),
        **trip_metrics(selected, grid=grid),
        # WR/PLR twice, on purpose, under two names that cannot be confused.
        # The keys spread in above are the SEARCH's per-position hedge, where no
        # split exists to argue about.  ``replayed`` is the attainable end's
        # portfolio-level hedge, attributed pro rata to |dollar_delta|. They
        # measure different hedgers and are not expected to agree; merging them
        # into one pair would force a choice this report does not have to make.
        # Absent (not null) when there was no replay -- under ``--hedge none``
        # there is no share leg, so a hedged win rate is not a missing number,
        # it is a question that does not arise.
        #
        # Gated on ``hedged``, NOT on ``per_trip`` being non-empty.  The two come
        # apart on a hedged run that selected nothing, and they say different
        # things: "no replay happened" has to stay distinguishable from "the
        # replay ran and had no trades", and only the second deserves nulls.
        **(
            {"replayed": replay_trip_metrics(per_trip, attribution=attribution)}
            if hedged is not None
            else {}
        ),
    }

    report = {
        "window": [dates[0].isoformat(), dates[-1].isoformat()],
        "steps": len(grid),
        "sessions": list(sessions),
        "nav": args.nav,
        "half_spread_multiplier": hs_mult,
        "max_positions": cap,
        "max_positions_per_underlying": per_name,
        # Without this block a ceiling is not interpretable. Two runs of this
        # script can differ by 30%+ on sizing alone, and before 2026-09-21 the
        # script sized by max-loss only while claiming to use the environment's
        # rule -- so a bare number carries no way to tell which rule produced it.
        "sizing": {
            "rule": config.size.size_rule,
            "limits": list(SIZE_RULE_LIMITS[config.size.size_rule]),
            "target_scenario_risk": config.size.target_scenario_risk,
            "vol_shock_relative": config.size.vol_shock_relative,
            # ``max_loss_cap`` was recorded here until 2026-09-22 and is gone
            # with the ceiling.  Its absence is the version marker: a manifest
            # carrying it was written under a rule that bounded ultimate risk at
            # a flat fraction of NAV, and is not commensurable with one below.
            #
            # Recorded since 2026-09-22, when ``nav_fraction`` became the default
            # rule and this field stopped being decoration: under that rule it is
            # the sizing unit, so a run that omitted it would be a ceiling with
            # its scale unrecorded.
            "nav_fraction": config.size.nav_fraction,
            "sized_against": "flat_book",
        },
        # The same argument as ``sizing``, for the three specs the script used to
        # leave at their ``EnvConfig`` defaults without saying so. A run that
        # differs from another only in ``max_relative_spread_package`` produced
        # two files that were byte-identical in their configuration, so the
        # arm was unrecoverable from the artifact.
        "resolver": {
            "lifted_quote_gates": list(lifted),
            "max_relative_spread_package": config.resolver.max_relative_spread_package,
            "max_quote_age_seconds": config.resolver.max_quote_age_seconds,
            "require_two_sided_quote": config.resolver.require_two_sided_quote,
            # Never lifted by --lift-quote-gates, deliberately; see above.
            "require_standard_deliverable": config.resolver.require_standard_deliverable,
            "expected_multiplier": config.resolver.expected_multiplier,
            "min_dte": config.resolver.min_dte,
            "delta_field": config.resolver.delta_field,
            "delta_convention": config.resolver.delta_convention,
        },
        "marking": {
            "max_mark_quote_age_seconds": config.marking.max_mark_quote_age_seconds,
            "american_tree_steps": config.marking.american_tree_steps,
        },
        "costs": {
            "half_spread_multiplier": config.cost.half_spread_multiplier,
            "option_fee_per_contract": config.cost.option_fee_per_contract,
            "stock_commission_per_share": config.cost.stock_commission_per_share,
            "stock_half_spread_bps": config.cost.stock_half_spread_bps,
            "assignment_fee": config.cost.assignment_fee,
            "borrow_rate_annual": config.cost.borrow_rate_annual,
        },
        # Recorded because it is a looseness, not a setting. ``config.risk`` is
        # never read by this script and ``breached_positions`` is called only
        # from ``OptionsEnv``, so no stop-loss fires, nothing is force-closed at
        # ``force_close_dte`` and the ruin floor never applies. The policy arms
        # this ceiling is compared against *do* get all three. Stating it beats
        # leaving a reader to infer it from an absence.
        "risk_controls_applied": False,
        # Stated in the payload, not just the docstring. ``selection_hedged``
        # was permanently false until 2026-09-23; it now follows ``--hedge``,
        # and when it is true every profit in this report -- ``total_profit``,
        # ``upper_bound_profit``, both log returns, ``by_family`` and friends --
        # is an option-plus-share number and is **not comparable** to the same
        # field in a report where it is false.
        "hedged": config.hedge.enabled,
        "selection_hedged": config.hedge.enabled,
        "hedge": {
            "families": list(hedged_families),
            "delta_band": config.hedge.delta_band,
            "band_rule": config.hedge.band_rule,
            "portfolio_level": config.hedge.portfolio_level,
            "spread_table": str(args.spread_table) if args.spread_table else None,
            "measured_spread_costs": config.flags.measured_spread_costs,
            # Since 2026-09-21 ``Position.with_mark`` scales the marked dollar
            # greeks by the package count. Every hedged run before that date
            # under-hedged by exactly ``quantity`` and is not comparable to this
            # one; recorded here so a reader of an old report can tell.
            "per_position_greeks": True,
            # The scoring hedger, which is deliberately *not* the replay's: per
            # position, at a NAV frozen at ``--nav``, because that is what makes
            # an arc cost a constant and the flow's optimum a bound. Nets
            # nothing, so it is the more expensive of the two hedgers and
            # ``net_profit`` below may exceed ``total_profit``.
            "selection_hedger": {
                "portfolio_level": False,
                "reference_nav": args.nav if config.hedge.enabled else None,
                **walk_stats,
            },
            **(hedged or {}),
        },
        "intents_resolved": resolved,
        "intents_failed": failed,
        "profitable_trips_available": len(trips),
        "trips_taken": len(selected),
        # An exit the oracle was *offered* and refused because the package
        # cannot be worth what the quotes said. Kept in the report because a
        # sudden jump here is the signature of a valuation regression, and the
        # oracle is the most sensitive detector of one available: it searches
        # directly for the largest number the pricing path will emit.
        "exits_refused_out_of_bounds": len(violations),
        "worst_refused": max(
            (v for v in violations), key=lambda v: v["excess"], default=None
        ),
        # Where the relaxed schedule broke the one rule the flow cannot see.
        # Published rather than assumed away: it is the whole reason the upper
        # end of the bracket is an upper bound and not the answer.
        "per_underlying_breaches": crowded,
        # The same disclosure for the per-step order budget. It is reported for
        # the attainable schedule too, where it must be empty: greedy enforces
        # it, and a non-empty list here means the lower end of the bracket is
        # not a schedule the environment would have accepted.
        "max_orders_per_step": per_step,
        "orders_per_step_breaches": busy_relaxed,
        # How many of the attainable schedule's step-boundaries are rolls, i.e.
        # a close and an open of the same name and head in one step. These are
        # the lines the ``X`` verb saves, and without them greedy would have to
        # forgo trips for an encoding reason.
        "rolls_credited": sum(
            rolled for _, rolled in order_lines_per_step(selected).values()
        ),
        "relaxed_trips_taken": len(relaxed),
        "upper_bound_profit": upper,
        "total_profit": total,
        "log_return": log((args.nav + total) / args.nav) if args.nav + total > 0 else None,
        "upper_bound_log_return": (
            log((args.nav + upper) / args.nav) if args.nav + upper > 0 else None
        ),
        # Risk-adjusted performance of the attainable schedule, on the hedged
        # equity path. Empty when ``--hedge none``: without the replay there is
        # no path, and the option-only curve is not the one to report ratios on.
        "performance": performance,
        "nav_curve": curve,
        "by_family": _by(selected, lambda t: t.candidate.head),
        "by_tenor": _by(selected, lambda t: t.candidate.tenor),
        "by_underlying": _by(selected, lambda t: t.candidate.underlying),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if dumped is not None:
        written = write_candidate_dump(
            dumped, args.dump_candidates, grid=grid, hedge=str(args.hedge)
        )
        if not args.quiet:
            print(f"\ncandidates  {written} rows -> {args.dump_candidates}")
    if not args.quiet:
        print(f"\nceiling  {total:+,.0f} .. {upper:+,.0f} on {args.nav:,.0f}")
        print(f"         log return {report['log_return']:+.4f}"
              f" .. {report['upper_bound_log_return']:+.4f}")
        print(f"         attainable: {len(selected)} trips under every rule,"
              f" of {len(trips)} profitable available")
        print(f"         relaxed:    {len(relaxed)} trips ignoring the"
              f" {per_name}/underlying cap")
        print(f"         {len(violations)} exits refused as outside the package's range")
        if crowded:
            worst = max(crowded, key=lambda c: c["held"])
            print(f"         the relaxation breaches {per_name}/underlying at"
                  f" {len(crowded)} steps (worst: {worst['underlying']} holding"
                  f" {worst['held']} at step {worst['step']})")
        if busy_relaxed:
            worst = max(busy_relaxed, key=lambda c: c["orders"])
            print(f"         the relaxation breaches {per_step} orders/step at"
                  f" {len(busy_relaxed)} steps (worst: {worst['orders']} at"
                  f" step {worst['step']})")
        print(f"         {report['rolls_credited']} of the attainable schedule's"
              f" order lines are saved by rolling")
        if hedged is not None:
            net = hedged["net_profit"]
            gross = hedged["option_profit"]
            cost = hedged["hedge_transaction_cost"]
            print(f"\nhedged   {net:+,.0f} net under the real portfolio-level"
                  f" hedger, against {gross:+,.0f} option-only"
                  f" on the same {len(selected)} trips")
            # Not a discrepancy. The schedule was chosen on per-position hedged
            # profit, which nets nothing and so overpays for the hedge; the
            # replay nets across each name and gets some of that back.
            if net > total:
                print(f"         {net - total:+,.0f} above the schedule's own"
                      f" score, which is netting the per-position hedge gave back")
            print(f"         hedge PnL {hedged['hedge_profit']:+,.0f}, of which"
                  f" {-cost:+,.0f} is transaction cost")
            # Stated against gross trip profit because that is the quantity the
            # hedge is being charged against: "the hedge costs 8% of what the
            # trades made" is the interpretable form, not a dollar figure whose
            # scale depends on --nav.
            if gross > 0:
                print(f"         transaction cost is {cost / gross:.1%} of gross"
                      f" trip profit")
            print(f"         {hedged['hedge_orders']} share orders over"
                  f" {hedged['hedge_points']} steps,"
                  f" {hedged['hedge_shares_traded']:,.0f} shares")
            if hedged["hedge_skipped"]:
                print(f"         skipped: {hedged['hedge_skipped']}")
    return 0


def write_candidate_dump(
    trips: Sequence[Trip], path: Path, *, grid: Sequence, hedge: str
) -> int:
    """Write one JSONL row per candidate best-exit outcome.

    This is the ex-post supervised label source for the Band D baselines, and it
    is a different object from anything else this script emits.  The report's
    ``by_family`` block is a *portfolio* aggregate over the ``selected``
    schedule -- it answers "what did the chosen 89 trips earn, split by family",
    which is a fact about the scheduler under its position limits, not about
    which family was best for one name on one day.  A supervised label needs the
    latter, so it has to come from the enumeration and not from the selection.

    ``profit`` is carried through unchanged: after entry spread, exit spread and
    both legs of per-contract fees, and after the package's own share leg when
    the run hedges that family.  So a dump taken under ``--hedge volatility``
    and one taken unhedged are **not** comparable and must not be pooled; the
    hedge setting is recorded per row so that mistake is at least visible.

    Losers are included.  The label a caller builds from this is "the family
    with the highest profit, or no-trade when the best of them is still
    negative", and dropping the negatives would silently turn every hopeless day
    into a buy signal for whichever family lost least.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for trip in trips:
            c = trip.candidate
            point = grid[c.open_index]
            fh.write(json.dumps({
                "trade_date": str(point.trade_date),
                "session": point.session,
                "open_index": c.open_index,
                "close_index": trip.close_index,
                "underlying": c.underlying,
                "family": c.family,
                "orientation": c.orientation,
                "tenor": c.tenor,
                "head": c.head,
                "hedge": hedge,
                "profit": trip.profit,
                "option_profit": trip.option_profit,
                "quantity": trip.quantity,
                "contracts": c.contracts,
                "entry_cost": c.entry_cost,
                "max_loss": c.max_loss,
                "expiry": str(c.expiry),
            }) + "\n")
            n += 1
    return n


def package_mark(
    candidate: Candidate, quotes: Mapping[str, tuple[float, float]]
) -> tuple[float, float] | None:
    """``(mid value, half-spread)`` of the whole package, or ``None``.

    ``None`` when *any* leg is unpriceable at this step.  Deliberately
    all-or-nothing, unlike ``MarketResolver._mark_position``, which falls back
    leg by leg: a mark may be partly carried, but an *exit* the oracle banks has
    to be one the environment would actually have filled, and a package with one
    carried leg is not fillable at the sum of its parts.
    """
    mark = 0.0
    half_spread = 0.0
    for leg in candidate.legs:
        quote = quotes.get(leg.contract_id)
        if quote is None:
            return None
        mid, leg_half = quote
        mark += leg.ratio * mid * leg.multiplier
        half_spread += abs(leg.ratio) * leg_half * leg.multiplier
    return mark, half_spread


def out_of_bounds(
    candidate: Candidate,
    *,
    mark: float,
    half_spread: float,
    bounds: tuple[float, float],
    close_index: int,
    violations: list[dict],
) -> bool:
    """Is this exit at a price the package provably cannot be worth?

    The tolerance is the package's own quoted half-width, for the reason given
    in ``market._tolerance``: a sum of per-leg midpoints may sit a tick outside a
    bound on what the package can be *worth* without anything being wrong.
    Measured at $25 median excess against a flat cent.

    Recorded and skipped, never fatal.  These are quotes that disagree with each
    other, which is a property of the chain rather than of the mark path, and an
    exit at a price the package cannot be worth is one the oracle must not be
    allowed to bank.  Refusing the whole run would be the wrong response: the
    first version did, and the population it refused on had a median excess of
    one tick.
    """
    low, high = bounds
    if low - half_spread - 0.01 <= mark <= high + half_spread + 0.01:
        return False
    violations.append(
        {
            "underlying": candidate.underlying,
            "head": candidate.head,
            "open_step": candidate.open_index,
            "close_step": close_index,
            "mark": mark,
            "bounds": [low, high],
            "spread": half_spread,
            "excess": (low - mark) if mark < low else (mark - high),
        }
    )
    return True


def option_leg_profit(
    candidate: Candidate,
    *,
    quantity: int,
    mark: float,
    exit_half_spread: float,
    entry_charge: float,
    fee: float,
    hs_mult: float,
) -> float:
    """Round-trip PnL on the option package alone, after both spreads and fees."""
    return (
        (mark - candidate.entry_cost) * quantity
        - entry_charge
        - (exit_half_spread * hs_mult + fee * candidate.contracts) * quantity
    )


def entry_charge_for(
    candidate: Candidate, *, quantity: int, fee: float, hs_mult: float
) -> float:
    return (
        candidate.entry_half_spread * hs_mult + fee * candidate.contracts
    ) * quantity


def best_exit(
    candidate: Candidate,
    *,
    quantity: int,
    grid: Sequence,
    prices: Sequence[Mapping[str, tuple[float, float]]],
    fee: float,
    hs_mult: float,
    audit: bool,
    violations: list[dict],
) -> Trip | None:
    """The most this package could have been closed for, after costs.

    Walks forward to expiry.  A step at which any leg is unpriceable is skipped
    rather than filled at a carried price: the whole point of the exercise is
    that every number in it is one the environment would actually have paid, and
    a mixed-provenance mark is the failure this script refuses to launder.

    **No share leg.**  This is the right answer for a candidate the run does not
    hedge, and only for that: a hedged family's exit has to be chosen on hedged
    profit, or the flow's optimum is over the wrong quantity.  See
    ``hedged_best_exits``.
    """
    bounds = value_bounds(candidate.payoff_legs()) if audit else (0.0, 0.0)
    entry_charge = entry_charge_for(
        candidate, quantity=quantity, fee=fee, hs_mult=hs_mult
    )
    best: Trip | None = None
    for index in range(candidate.open_index + 1, len(grid)):
        if grid[index].trade_date > candidate.expiry:
            break
        quote = package_mark(candidate, prices[index])
        if quote is None:
            continue
        mark, exit_hs = quote
        if audit and out_of_bounds(
            candidate,
            mark=mark,
            half_spread=exit_hs,
            bounds=bounds,
            close_index=index,
            violations=violations,
        ):
            continue
        profit = option_leg_profit(
            candidate,
            quantity=quantity,
            mark=mark,
            exit_half_spread=exit_hs,
            entry_charge=entry_charge,
            fee=fee,
            hs_mult=hs_mult,
        )
        if best is None or profit > best.profit:
            best = Trip(
                candidate=candidate,
                close_index=index,
                quantity=quantity,
                profit=profit,
                option_profit=profit,
                exit_mark=mark,
                exit_half_spread=exit_hs,
            )
    return best


def position_for(
    candidate: Candidate, *, position_id: str, quantity: int, opened_at: datetime
) -> Position:
    """The ``Position`` the environment would be holding after this open.

    One constructor for both the hedged *search* and the hedged *replay*.  They
    answer different questions and use different hedgers, but if they disagreed
    about what the book is holding -- a sign on ``entry_cost``, a missing
    ``family`` -- the bound and the attainable number would be bounding
    different books and the bracket would mean nothing.

    ``entry_cost`` flips sign on the way in: the candidate states a positive
    debit, the book states a signed cash flow.  ``mark`` is left per package,
    which is what makes ``unrealized_pnl`` exactly zero at open.
    """
    return Position(
        position_id=position_id,
        underlying=candidate.underlying,
        family=candidate.family,
        orientation=candidate.orientation,
        strategy_handle=candidate.strategy_handle,
        legs=tuple(
            Leg(
                contract_id=leg.contract_id,
                right=leg.right,
                strike=leg.strike,
                expiry=candidate.expiry,
                ratio=leg.ratio,
                multiplier=leg.multiplier,
                entry_price=leg.entry_mid,
                entry_half_spread=leg.entry_half_spread,
            )
            for leg in candidate.legs
        ),
        quantity=quantity,
        opened_at=opened_at,
        entry_cost=-candidate.entry_cost * quantity,
        collateral=0.0,
        max_loss_per_package=candidate.max_loss,
        mark=candidate.entry_cost,
        entry_mark=candidate.entry_cost,
    )


@dataclass(slots=True)
class _Walk:
    """One candidate's own hedged history, live only while it is open.

    ``book`` is not an account and its NAV is never read.  Cash starts at zero
    and only ever receives share ``cash_delta``, so ``book.cash`` *is* this
    package's share ledger; ``book.positions`` is overwritten every step with the
    freshly marked copy of its one position, because ``HedgeResolver`` reads
    ``dollar_delta`` off the position and nothing else.
    """

    candidate: Candidate
    quantity: int
    entry_charge: float
    position_id: str
    ticker: str
    bounds: tuple[float, float]
    book: Book
    best: Trip | None = None


def hedged_best_exits(
    opens: Mapping[int, Sequence[tuple[Candidate, int]]],
    *,
    grid: Sequence,
    prices: Sequence[Mapping[str, tuple[float, float]]],
    spots: Sequence[Mapping[str, float]],
    config: EnvConfig,
    chain: OptionChain,
    nav: float,
    spreads: SpreadTable | None,
    fee: float,
    hs_mult: float,
    audit: bool,
    violations: list[dict],
    quiet: bool = False,
) -> tuple[list[Trip], dict]:
    """Each candidate's best exit scored on **hedged** profit, at frozen NAV.

    This is the upper end of the bracket, and it exists because scoring an arc
    hedged while its exit was chosen unhedged does not bound anything: the flow
    would be optimising over exits picked to maximise a quantity it is not
    maximising, and some other exit of the same candidate could have had higher
    hedged profit without the flow ever seeing it.

    **Two properties make it an exact relaxation, and both are deliberate
    misstatements of the account.**

    *Per-position hedging.*  Each candidate carries its own share ledger and
    nets with nothing, so its hedged profit is a property of the candidate
    alone.  Real portfolio-level hedging nets across a name and is therefore
    *cheaper*; this end of the bracket pays more for its hedge than the
    attainable end does, which is why ``replay_hedged`` may legitimately come in
    above it.  The two ends bound different things and neither is a check on the
    other.

    *Frozen NAV.*  ``band = delta_band * nav`` on a running NAV makes one trip's
    hedge depend on every other trip's marks, and then an arc cost is not a
    constant and ``select_flow`` is solving a different problem from the one it
    reports.  ``nav`` is pinned to the opening equity for every candidate on
    every step -- see ``HedgeResolver.hedge``'s ``nav`` argument.

    **One walk per candidate yields every exit.**  The share ledger up to step
    ``s`` does not depend on when the position is *planned* to close, so the
    forward hedge is walked once and each step is asked "if this closed here",
    rather than replaying the candidate once per candidate exit.

    **Step-major, because marking is not per-trip and hedging is.**  A mark is
    position-local -- ``_mark_position`` reads the position and the step's chain
    slice and nothing else -- so every live candidate is marked in a single
    ``MarketResolver.mark`` call on one shared book, and the marked positions are
    then handed out to the per-candidate books.  The hedge is the opposite: it is
    exactly the part that has to stay per-trip, because separability is the
    property being bought.

    Returns ``(trips, diagnostics)``.  A candidate with no priceable exit
    contributes no trip, the same refusal ``best_exit`` makes.
    """
    spec = config.hedge
    hedger = HedgeResolver(config)
    market = MarketResolver(config, chain)
    execution = ExecutionModel(config)
    # Marks only: its cash is never spent, its NAV is never read, and no share
    # ever touches it.  It exists so that ``mark`` is called once per step
    # instead of once per candidate per step.
    marking = Book(cash=0.0, initial_cash=0.0)
    # Read-only probe for "what would liquidating this balance raise?".
    # ``HedgeResolver`` never mutates the book, so the same instance is reused
    # and the orders it returns are priced but not filled.
    probe = Book(cash=0.0, initial_cash=0.0)
    live: dict[str, _Walk] = {}
    trips: list[Trip] = []
    orders = stale = unliquidatable = 0
    shares_traded = hedge_cost = 0.0

    def retire(position_id: str) -> None:
        walk = live.pop(position_id)
        marking.positions.pop(position_id, None)
        if walk.best is not None:
            trips.append(walk.best)

    for index, point in enumerate(grid):
        for position_id in [
            p for p, w in live.items() if point.trade_date > w.candidate.expiry
        ]:
            retire(position_id)

        for candidate, quantity in opens.get(index, ()):
            ticker = spec.hedge_ticker_for(candidate.underlying)
            if ticker is None:
                # Unreachable through ``main``, which routes these to
                # ``best_exit``; asserted rather than silently hedged to zero so
                # a future caller cannot get an unhedged arc into a hedged flow.
                raise RuntimeError(
                    f"{candidate.underlying} has no share instrument, so its"
                    f" arc cannot be hedged -- it belongs in best_exit"
                )
            position_id = marking.next_position_id()
            marking.open(
                position_for(
                    candidate,
                    position_id=position_id,
                    quantity=quantity,
                    opened_at=point.timestamp,
                )
            )
            live[position_id] = _Walk(
                candidate=candidate,
                quantity=quantity,
                entry_charge=entry_charge_for(
                    candidate, quantity=quantity, fee=fee, hs_mult=hs_mult
                ),
                position_id=position_id,
                ticker=ticker,
                bounds=value_bounds(candidate.payoff_legs()) if audit else (0.0, 0.0),
                book=Book(cash=0.0, initial_cash=0.0),
            )

        if not live:
            continue

        step_prices = prices[index]
        step_spots = spots[index]

        # -- "if it closed here".  Before the hedge, because within a step
        #    ``replay_hedged`` closes first and hedges afterwards, so the balance
        #    a close unwinds is the one carried in from the step before.
        for walk in live.values():
            if index <= walk.candidate.open_index:
                continue
            quote = package_mark(walk.candidate, step_prices)
            if quote is None:
                continue
            mark, exit_hs = quote
            if audit and out_of_bounds(
                walk.candidate,
                mark=mark,
                half_spread=exit_hs,
                bounds=walk.bounds,
                close_index=index,
                violations=violations,
            ):
                continue
            held = walk.book.shares.get(walk.ticker, 0.0)
            unwind = 0.0
            if held:
                spot = step_spots.get(walk.ticker, 0.0)
                if spot <= 0:
                    # No price for the share leg means the balance cannot be
                    # retired, and an exit that leaves permanent naked stock
                    # behind is not an exit.  Skipped, not valued at a carried
                    # spot: that is the mixed-provenance mark this script
                    # refuses everywhere else.
                    unliquidatable += 1
                    continue
                probe.shares = {walk.ticker: held}
                plan = hedger.hedge(
                    probe,
                    spots={walk.ticker: spot},
                    as_of=point.timestamp,
                    spreads=spreads,
                    session=point.session,
                    nav=nav,
                )
                if not plan.orders:
                    unliquidatable += 1
                    continue
                unwind = sum(o.cash_delta for o in plan.orders)
            option = option_leg_profit(
                walk.candidate,
                quantity=walk.quantity,
                mark=mark,
                exit_half_spread=exit_hs,
                entry_charge=walk.entry_charge,
                fee=fee,
                hs_mult=hs_mult,
            )
            # ``book.cash`` is every share ``cash_delta`` so far -- slippage and
            # commission included, because ``_order`` folds both into it -- and
            # ``unwind`` retires what is left.  Together they are the share
            # leg's whole contribution, realized.
            profit = option + walk.book.cash + unwind
            if walk.best is None or profit > walk.best.profit:
                walk.best = Trip(
                    candidate=walk.candidate,
                    close_index=index,
                    quantity=walk.quantity,
                    profit=profit,
                    option_profit=option,
                    exit_mark=mark,
                    exit_half_spread=exit_hs,
                )

        # -- carry the hedge forward
        report = market.mark(
            marking,
            trade_date=point.trade_date,
            session=point.session,
            decision_time=point.timestamp,
        )
        marking.apply_marks(report.marks)
        stale += sum(1 for m in report.marks.values() if m.quality == "stale")

        for walk in live.values():
            spot = step_spots.get(walk.ticker, 0.0)
            if spot <= 0:
                continue
            # The marked copy, not the one opened above: ``apply_marks``
            # replaces the entry in ``marking.positions`` with a new frozen
            # ``Position`` carrying this step's greeks, and the hedger reads
            # nothing else.
            walk.book.positions = {walk.position_id: marking.positions[walk.position_id]}
            plan = hedger.hedge(
                walk.book,
                spots={walk.ticker: spot},
                as_of=point.timestamp,
                spreads=spreads,
                session=point.session,
                nav=nav,
            )
            for order in plan.orders:
                execution.hedge(walk.book, order)
                orders += 1
                shares_traded += abs(order.quantity)
                hedge_cost += order.commission + abs(order.quantity) * abs(
                    order.fill_price - order.reference_price
                )

        if not quiet and index % 25 == 0:
            print(
                f"  hedged-walk step {index}/{len(grid)}  {len(live)} open,"
                f" {orders} share orders, {len(trips)} walks finished",
                flush=True,
            )

    for position_id in list(live):
        retire(position_id)

    return trips, {
        "walks": len(trips),
        "hedge_orders": orders,
        "hedge_shares_traded": shares_traded,
        "hedge_transaction_cost": hedge_cost,
        "stale_marks": stale,
        # Exits refused because the share leg had no price to be retired at.
        # A large number here means the bound is being computed over a thinner
        # exit menu than the unhedged one, which loosens it in the wrong
        # direction and has to be visible rather than inferred.
        "exits_without_a_share_price": unliquidatable,
    }


def replay_hedged(
    selected: Sequence[Trip],
    *,
    grid: Sequence,
    prices: Sequence[Mapping[str, tuple[float, float]]],
    spots: Sequence[Mapping[str, float]],
    config: EnvConfig,
    chain: OptionChain,
    nav: float,
    spreads: SpreadTable | None = None,
    quiet: bool = False,
) -> dict:
    """Rung H2: push the chosen schedule through the real hedger and a real book.

    The schedule is fixed -- every open, every close, every size is the one the
    unhedged selection picked.  What is *not* fixed is the share leg: at every
    step the book is marked by the real ``MarketResolver`` and offered to the
    real ``HedgeResolver``, and whatever it asks for is filled by the real
    ``ExecutionModel``.  Nothing here reimplements a rule; if the hedge is
    mis-specified, this reports the mis-specification, which is the point.

    Three couplings the flow could not see are therefore honoured rather than
    relaxed: the band is ``delta_band * nav`` on the *running* NAV, positions on
    a name net into one delta request, and the share balance carries forward so
    the cost of a correction depends on the whole hedge history.

    **The tie-out is the corruption check.**  Option PnL is accumulated
    independently of the hedge and asserted equal to ``sum(t.option_profit)``.
    If the replay's cash accounting drifts from the search's -- a different exit
    spread, a sign flip on a credit structure, a position left open -- the two
    numbers separate and the caller refuses to report.  Without it a hedge
    number could absorb an accounting bug and look merely expensive.

    ``option_profit`` and not ``profit``: under a hedged objective ``profit``
    already carries the *search's* per-position share leg, which this replay
    deliberately does not reproduce, so comparing against it would fire on every
    hedged run and the check would be discarded rather than believed.

    Ordering within a step is close, then open, then mark, then hedge.  That
    differs from the environment, which marks at the top of the step and hedges
    at the bottom, but only in where a *fresh* open gets its greeks: here from
    the step's chain slice, there from ``ResolvedPackage``.  Under
    ``--decision-sessions PM`` they are the same slice, so the two coincide.
    """
    book = Book(cash=nav, initial_cash=nav)
    execution = ExecutionModel(config)
    market = MarketResolver(config, chain)
    hedger = HedgeResolver(config)
    fee = config.cost.option_fee_per_contract
    hs_mult = config.cost.half_spread_multiplier

    opens: dict[int, list[Trip]] = {}
    closes: dict[int, list[Trip]] = {}
    for trip in selected:
        opens.setdefault(trip.candidate.open_index, []).append(trip)
        closes.setdefault(trip.close_index, []).append(trip)

    live: dict[int, tuple[str, float]] = {}  # id(trip) -> (position_id, entry_charge)
    option_pnl = 0.0
    hedge_cost = 0.0
    hedge_orders = 0
    hedge_shares = 0.0
    hedge_points = 0
    examined = 0
    no_price: dict[str, int] = {}
    unmarkable = 0
    curve: list[dict] = []

    # -- per-trip attribution of the PORTFOLIO-LEVEL hedge ------------------
    #
    # ``replay_trip_metrics`` documents the convention and what it costs.
    # Everything here is bookkeeping for one identity, which the conservation
    # check at the bottom enforces:
    #
    #     hedge PnL = sum_s h_s * (S_{s+1} - S_s)  -  sum_orders (comm + |q|*|F-ref|)
    #
    # i.e. *holding* PnL minus *transaction* cost.  It is exact, not an
    # approximation, because the hedger's ``reference_price`` is the same spot
    # the book marks shares at: buying ``dh`` shares at fill ``F`` moves cash by
    # ``-dh*F`` and share value by ``+dh*S``, so the valuation-neutral part
    # cancels and only the spread and commission survive.
    n_trips = len(selected)
    order_of: dict[int, int] = {id(t): i for i, t in enumerate(selected)}
    pos_to_trip: dict[str, int] = {}
    #: Static per trip: the ticker whose shares hedge it, which is the underlying
    #: for the nine single names and the proxy for an index.  ``None`` for a trip
    #: in an unhedged family, which therefore never accrues a share leg at all --
    #: under ``--hedge volatility`` that is every directional trip.
    trip_ticker_of: list[str | None] = [
        config.hedge.hedge_ticker_for(t.candidate.underlying)
        if t.candidate.family in config.hedge.hedged_families
        else None
        for t in selected
    ]
    trip_option: list[float] = [0.0] * n_trips
    trip_hedge: list[float] = [0.0] * n_trips
    #: Notional share count each trip owns, from splitting the ticker's single
    #: fungible balance.  Rebuilt after every hedge; sums to ``book.shares``.
    trip_shares: list[float] = [0.0] * n_trips
    #: Last split used per ticker, carried so an ``_unwind_orphans`` order --
    #: which has ``attributed_to == ()`` because no hedgeable position is left
    #: standing behind the balance -- is charged to the trips whose closing
    #: created the orphan, rather than being dropped on the floor.
    last_weights: dict[str, dict[int, float]] = {}
    last_spot: dict[str, float] = {}
    #: Orders whose ticker has no live and no remembered owner.  Reported, not
    #: silently absorbed: it is the one leak the identity below cannot see.
    unattributed_cost = 0.0
    orders_single_owner = 0
    orders_multi_owner = 0

    def hedge_weights() -> dict[str, dict[int, float]]:
        """``ticker -> {trip index: fraction}``, pro rata by ``|dollar_delta|``.

        The same rule ``HedgeResolver._per_position`` uses to divide a share
        balance, applied to the same object.  Reusing it is the point: this is
        not a new convention invented for a metric, it is the convention the
        hedger already states for exactly this question.

        A group whose deltas all mark to zero splits evenly.  ``_per_position``
        hands such a group ``0.0`` shares each, which is the same statement --
        nobody has a claim -- but a metric needs weights that sum to one, and an
        even split is the only choice that does not privilege a position.
        """
        groups: dict[str, list[tuple[int, float]]] = {}
        for pid, position in book.positions.items():
            idx = pos_to_trip.get(pid)
            if idx is None or position.family not in config.hedge.hedged_families:
                continue
            ticker = config.hedge.hedge_ticker_for(position.underlying)
            if ticker is None:
                continue
            groups.setdefault(ticker, []).append((idx, abs(position.dollar_delta)))
        out: dict[str, dict[int, float]] = {}
        for ticker, members in groups.items():
            total = sum(w for _, w in members)
            if total > 0.0:
                out[ticker] = {i: w / total for i, w in members}
            else:
                out[ticker] = {i: 1.0 / len(members) for i, _ in members}
        return out

    def record(index: int, point) -> None:
        """NAV at the *end* of a step: after the closes, opens, mark and hedge.

        This is the only place the hedged equity path is observable.  The
        aggregate return fields below are differences between two endpoints and
        so are blind to everything in between; a drawdown or a volatility cannot
        be recovered from them afterwards, which is why the series is recorded
        here rather than reconstructed.

        Written on *every* grid point, including the ones the loop
        short-circuits for an empty book.  A step with no positions still has a
        NAV -- all cash -- and omitting it would not leave a visible gap: it
        would silently shorten the return series and join two non-adjacent days
        into one return, which a volatility estimate would read as a calm
        window rather than a missing one.
        """
        curve.append(
            {
                "step": index,
                "trade_date": point.trade_date.isoformat(),
                "session": str(point.session),
                "nav": book.nav,
                "cash": book.cash,
                "position_value": book.position_value,
                "share_value": book.share_value,
                "open_positions": len(book.positions),
            }
        )

    for index, point in enumerate(grid):
        # Holding PnL for the interval that just ENDED, accrued before this
        # step's closes.  A trip that closes at ``index`` still owned its share
        # of the balance across ``index-1 -> index`` and is owed that move; doing
        # this after the close loop would silently give the last interval of
        # every trip's life to whoever was still open.
        for idx, held_shares in enumerate(trip_shares):
            if held_shares == 0.0:
                continue
            ticker = trip_ticker_of[idx]
            was = last_spot.get(ticker)
            now = spots[index].get(ticker)
            if was is None or now is None or now <= 0.0:
                continue
            trip_hedge[idx] += held_shares * (now - was)

        for trip in closes.get(index, ()):
            exit_charge = (
                trip.exit_half_spread * hs_mult + fee * trip.candidate.contracts
            ) * trip.quantity
            position_id, entry_charge = live.pop(id(trip))
            # Taken off ``Book.close``'s return value, not off ``trip.profit``.
            # That is the whole point of the tie-out: accumulating the search's
            # own number here would make the comparison below true by
            # construction and would check nothing.  This path runs through
            # ``Position.entry_cost``, whose sign convention is inverted from
            # the candidate's, so a credit structure booked with the wrong sign
            # separates the two.  ``close`` charges the exit but knows nothing
            # about entry, which is why the entry charge is subtracted here.
            realized = (
                book.close(
                    position_id,
                    proceeds=trip.exit_mark * trip.quantity,
                    cost=exit_charge,
                )
                - entry_charge
            )
            option_pnl += realized
            # Per trip off the SAME expression the aggregate uses, so the two can
            # never disagree about a trip and agree about the total.
            trip_option[order_of[id(trip)]] += realized
        for trip in opens.get(index, ()):
            candidate = trip.candidate
            entry_charge = entry_charge_for(
                candidate, quantity=trip.quantity, fee=fee, hs_mult=hs_mult
            )
            position_id = book.next_position_id()
            book.open(
                position_for(
                    candidate,
                    position_id=position_id,
                    quantity=trip.quantity,
                    opened_at=point.timestamp,
                )
            )
            book.cash -= candidate.entry_cost * trip.quantity + entry_charge
            live[id(trip)] = (position_id, entry_charge)
            pos_to_trip[position_id] = order_of[id(trip)]

        if not book.positions and not book.shares:
            # No shares means nothing to hold and nothing to split, so the
            # ledger is flat by construction rather than by omission.
            trip_shares = [0.0] * n_trips
            last_spot.clear()
            record(index, point)
            continue

        report = market.mark(
            book,
            trade_date=point.trade_date,
            session=point.session,
            decision_time=point.timestamp,
        )
        book.apply_marks(report.marks)
        unmarkable += sum(1 for m in report.marks.values() if m.quality == "stale")

        # Shares are marked from pass 1's spots rather than the mark report's,
        # so a balance whose option leg has already closed still has a price and
        # can be unwound.  An unpriced balance is not hedgeable and becomes
        # permanent naked stock.
        step_spots = dict(spots[index])
        for ticker in book.shares:
            if ticker in step_spots:
                book.set_share_mark(ticker, step_spots[ticker])

        wanted = {p.underlying for p in book.positions.values()} | set(book.shares)
        wanted |= {
            ticker
            for underlying in tuple(wanted)
            if (ticker := config.hedge.hedge_ticker_for(underlying)) is not None
        }
        plan = hedger.hedge(
            book,
            spots={k: v for k, v in step_spots.items() if k in wanted and v > 0},
            as_of=point.timestamp,
            spreads=spreads,
            session=point.session,
        )
        hedge_points += 1
        examined += len(plan.examined)
        # ``skipped`` is keyed by position id and valued by reason, so the
        # rollup counts the values.  A non-empty tally here is not cosmetic: a
        # position that never reached the band rule was not hedged, and the
        # number would otherwise read as "the band never fired".
        for reason in plan.skipped.values():
            no_price[reason] = no_price.get(reason, 0) + 1
        # Taken BEFORE the orders execute.  ``dollar_delta`` is a property of the
        # option leg and the fills do not touch it, but the weights must be the
        # ones the hedger itself grouped on, and reading them afterwards would
        # invite a later change to the fill path to silently desynchronise them.
        weights_now = hedge_weights()

        for order in plan.orders:
            execution.hedge(book, order)
            hedge_orders += 1
            hedge_shares += abs(order.quantity)
            cost = order.commission + abs(order.quantity) * abs(
                order.fill_price - order.reference_price
            )
            hedge_cost += cost
            # ``attributed_to`` is the hedger's own statement of ownership, and
            # under ``portfolio_level`` it names every position in the netted
            # group -- which is exactly why a split is needed.  Its keys are the
            # keys of ``weights_now[ticker]``, so the group is re-derived rather
            # than re-listed.  An orphan unwind names nobody and falls back to
            # the remembered split.
            shares_of = weights_now.get(order.ticker) or last_weights.get(order.ticker)
            if not shares_of:
                unattributed_cost += cost
                continue
            if len(shares_of) == 1:
                orders_single_owner += 1
            else:
                orders_multi_owner += 1
            for idx, fraction in shares_of.items():
                trip_hedge[idx] -= cost * fraction

        # Re-split the post-hedge balance and remember the spot the book just
        # marked it at.  Rebuilt from zero rather than patched, so a trip whose
        # ticker has gone flat cannot keep a stale share count.
        trip_shares = [0.0] * n_trips
        for ticker, balance in book.shares.items():
            shares_of = weights_now.get(ticker) or last_weights.get(ticker)
            if not shares_of:
                continue
            last_weights[ticker] = shares_of
            for idx, fraction in shares_of.items():
                trip_shares[idx] = balance * fraction
        for ticker in weights_now:
            last_weights[ticker] = weights_now[ticker]
        # ``update``, not replace.  A ticker with no spot this step also had
        # ``set_share_mark`` skipped, so the book is still carrying the OLD mark
        # and its value did not move.  Keeping the old entry makes the next
        # accrual span the gap exactly as the book's own mark does; dropping it
        # would skip that interval on both sides of the missing step and leak a
        # real price move out of the attribution.
        last_spot.update({t: s for t, s in step_spots.items() if s > 0.0})

        record(index, point)
        if not quiet and index % 25 == 0:
            print(
                f"  hedged step {index}/{len(grid)}  {len(book.positions)} open,"
                f" {hedge_orders} share orders, {hedge_cost:,.0f} paid",
                flush=True,
            )

    if live:
        raise RuntimeError(f"{len(live)} trips never closed in the replay")

    # Every position is closed, so anything left is cash plus the residual share
    # balance.  Share PnL and share costs are not separable from NAV without
    # re-deriving the basis, so the hedge's total effect is stated as a residual
    # and its *transaction* component -- the only signed part -- is stated
    # separately from it.
    net = book.nav - nav
    # ``option_profit``, never ``profit``.  Under a hedged objective ``profit``
    # carries the search's own per-position share leg, which this replay does not
    # reproduce and is not trying to: it hedges portfolio-level on a running NAV.
    # Checked against ``profit`` the tie-out would fire on every hedged run and
    # would stop being a check on anything.
    expected = sum(t.option_profit for t in selected)
    if len(curve) != len(grid):
        # Not a by-construction tautology: the two ``record`` call sites are on
        # either side of a ``continue``, so any future branch that leaves the
        # step body by a third path drops a point, and a dropped point does not
        # show up as a gap -- it shows up as a return taken across two
        # non-adjacent days.  Refusing here beats publishing a volatility
        # computed on a series that is quietly missing its quiet days.
        raise RuntimeError(
            f"the equity curve has {len(curve)} points for {len(grid)} grid steps"
        )
    if curve and abs((curve[-1]["nav"] - nav) - net) > 1e-6:
        raise RuntimeError(
            f"the curve ends at {curve[-1]['nav'] - nav:+,.2f} against a reported"
            f" net of {net:+,.2f} -- the series is not this replay's"
        )

    # **The attribution's own corruption check.**  The per-trip share legs plus
    # the cost nobody owned must reconstruct the aggregate hedge PnL, which is
    # computed a completely different way -- as the residual ``net - option_pnl``
    # off the book's NAV.  They agree only if the holding/transaction identity
    # holds at every step, which is the one thing that could quietly rot: a
    # missed accrual, a double-counted interval, or weights that failed to sum
    # to one all show up here and nowhere else.  Without it a per-trip hedge
    # number would still look like a hedge number.
    hedge_total = net - option_pnl
    attributed = sum(trip_hedge) - unattributed_cost
    attribution_residual = hedge_total - attributed
    tolerance = 1e-6 * max(1.0, abs(hedge_total))
    if abs(attribution_residual) > tolerance:
        raise RuntimeError(
            f"per-trip hedge attribution sums to {attributed:+,.2f} against an"
            f" aggregate hedge PnL of {hedge_total:+,.2f}"
            f" (residual {attribution_residual:+,.4f} > {tolerance:,.6f});"
            " the holding/transaction identity broke"
        )

    return {
        "option_profit": option_pnl,
        "option_profit_expected": expected,
        "net_profit": net,
        "hedge_profit": net - option_pnl,
        "hedge_transaction_cost": hedge_cost,
        "hedge_orders": hedge_orders,
        "hedge_shares_traded": hedge_shares,
        "hedge_points": hedge_points,
        "hedge_groups_examined": examined,
        "hedge_skipped": no_price,
        "residual_share_value": book.share_value,
        "stale_marks": unmarkable,
        # Per trip, aligned index-for-index with ``selected``.  Popped out by the
        # caller the same way ``nav_curve`` is: it is a record, not a setting.
        "per_trip": [
            {
                "option_profit": trip_option[i],
                "hedge_profit": trip_hedge[i],
                "profit": trip_option[i] + trip_hedge[i],
            }
            for i in range(n_trips)
        ],
        # How much the convention is actually doing.  An order with one owner is
        # attributed EXACTLY -- there was nothing to split -- so this is the
        # honest way to ship a stated convention: say what share of the answer
        # depends on it.  If ``orders_needing_split`` is near zero the choice of
        # rule cannot matter; if it dominates, the WR/PLR below are a statement
        # about the rule as much as about the trades.
        "attribution": {
            "orders_exact": orders_single_owner,
            "orders_needing_split": orders_multi_owner,
            "orders_unattributed": (
                hedge_orders - orders_single_owner - orders_multi_owner
            ),
            "unattributed_cost": unattributed_cost,
            "residual": attribution_residual,
        },
        # Popped out of this block by the caller -- it is a series, not a hedge
        # setting.  ``initial_nav`` travels with it because the curve's first
        # point is the *end* of step 0, by which time the opening trades and
        # their entry spreads are already paid; without the opening basis the
        # first day's return would be dropped and day 0's cost would vanish
        # from the series while staying in the total.
        "nav_curve": curve,
        "initial_nav": nav,
    }


def daily_closes(
    curve: Sequence[Mapping], *, initial_nav: float
) -> tuple[list[str], list[float]]:
    """Collapse the step curve to one NAV per trading day, plus the opening basis.

    Under ``--decision-sessions PM`` there is already one point per day and this
    is the identity.  Under ``AM,PM`` it keeps the *last* session of each day and
    drops the AM mark, which is not a loss: an AM-to-PM and a PM-to-AM return
    alternate between a full overnight gap and an intraday move, so their
    standard deviation scaled by ``sqrt(252)`` annualises a quantity that is not
    a daily return. Sampling the close makes the series comparable across
    session settings, which is the only way an arm run at PM and an arm run at
    AM+PM can be put in the same table.

    The opening NAV is prepended as the basis rather than the first level, so the
    first return is the one that carries day 0's entry spreads.  Dropping it
    would leave those costs in ``total_profit`` but out of the return series --
    the single easiest way to publish a Sharpe that is better than the trading.
    """
    levels: list[float] = [initial_nav]
    labels: list[str] = ["open"]
    for point in curve:
        day = point["trade_date"]
        if labels[-1] == day:
            levels[-1] = point["nav"]
        else:
            labels.append(day)
            levels.append(point["nav"])
    return labels, levels


def performance_metrics(
    levels: Sequence[float],
    *,
    trading_days_per_year: int,
    risk_free_annual: float = 0.0,
) -> dict:
    """The six portfolio-level metrics, from a daily NAV series.

    These need no attribution convention.  TR, AVOL, MDD, ASR, ACR and ASoR are
    all functions of the equity path, so the hedge's PnL is already in them by
    construction and no question arises about which trade owns which share fill.
    WR and PLR are per-trade and so had that question to answer; ``trip_metrics``
    answers it by reading the per-position hedged walk, which prices each
    package's own share leg and therefore needs no split invented.  The two
    differ in *which* hedge they carry: these metrics carry the replay's
    portfolio-level netting, ``trip_metrics`` carries the search's per-position
    hedge.  The bracket's two ends are not expected to agree.

    Every denominator that can be zero returns ``None`` rather than an infinity.
    An infinite Calmar on a window that happened never to draw down is not a
    good result, it is an undefined one, and a ``None`` survives a round trip
    through JSON into a table without becoming the best row in it.

    Annualisation is stated, not assumed: ``ASR`` is the daily Sharpe times
    ``sqrt(252)`` (not ``CAGR/AVOL``, which differs whenever returns compound),
    and ``ACR`` is ``CAGR/MDD``.  Over a three-month window ``CAGR`` raises a
    quarter's result to the fourth power, so the annualised fields are a
    rescaling of a short sample and not a forecast of a year.
    """
    n = len(levels) - 1
    empty = {
        "trading_days": max(n, 0),
        "years": None,
        "risk_free_annual": risk_free_annual,
        "total_return": None,
        "log_return": None,
        "cagr": None,
        "annual_volatility": None,
        "max_drawdown": None,
        "annual_sharpe": None,
        "annual_calmar": None,
        "annual_sortino": None,
        "ruined_on_day": None,
    }
    if n < 1:
        return empty

    # A NAV that reaches zero ends the return series: every later ratio is
    # undefined or sign-flipped, and carrying on would produce finite-looking
    # numbers for a path that is already bankrupt.  Reported as a day index
    # rather than swallowed.
    for i, level in enumerate(levels):
        if level <= 0.0:
            return {**empty, "trading_days": n, "ruined_on_day": i}

    returns = [levels[i] / levels[i - 1] - 1.0 for i in range(1, len(levels))]
    years = n / trading_days_per_year
    total_return = levels[-1] / levels[0] - 1.0
    cagr = (levels[-1] / levels[0]) ** (1.0 / years) - 1.0

    peak = levels[0]
    max_dd = 0.0
    for level in levels:
        peak = max(peak, level)
        max_dd = max(max_dd, (peak - level) / peak)

    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days_per_year) - 1.0
    excess = [r - rf_daily for r in returns]
    mean_excess = fmean(excess)
    # ddof=1: this is a sample of a return process, not the population of it.
    # On 62 observations the difference from ddof=0 is 0.8% of the volatility --
    # small, but the sample form is the one every comparison table means.
    sd = stdev(excess) if n >= 2 else 0.0
    # Sortino's denominator averages the squared shortfalls over *all* n
    # observations, not over the losing ones.  Dividing by the loss count
    # instead inflates the ratio for a strategy that rarely loses, which is
    # exactly the strategy the metric is supposed to distinguish.
    downside = sqrt(fmean([min(e, 0.0) ** 2 for e in excess]))
    root = sqrt(trading_days_per_year)
    return {
        "trading_days": n,
        "years": years,
        "risk_free_annual": risk_free_annual,
        "total_return": total_return,
        "log_return": log(levels[-1] / levels[0]),
        "cagr": cagr,
        "annual_volatility": sd * root if sd > 0.0 else None,
        "max_drawdown": max_dd,
        "annual_sharpe": (mean_excess / sd * root) if sd > 0.0 else None,
        "annual_calmar": (cagr / max_dd) if max_dd > 0.0 else None,
        "annual_sortino": (mean_excess / downside * root) if downside > 0.0 else None,
        "ruined_on_day": None,
    }


def replay_trip_metrics(per_trip: Sequence[Mapping], *, attribution: Mapping) -> dict:
    """WR and PLR on the **replayed, portfolio-level** hedge.

    ``trip_metrics`` below reports the same two statistics on the *search's*
    per-position hedge, where each package carries its own share ledger and no
    split exists to be argued about.  This function answers the harder version
    the user asked for: the attainable end, where the hedger nets every position
    on a ticker into one order and a per-trade number does not fall out.

    **The convention, and why it is not a new one.**  Each trip's hedge PnL is

        sum_s  h_s^i * (S_{s+1} - S_s)   -   sum_orders  cost * w_i

    where ``h_s^i`` is trip ``i``'s share of the ticker's single fungible balance
    and ``w_i`` is the same fraction.  Both come from splitting pro rata to
    ``|dollar_delta|`` -- which is verbatim the rule
    ``HedgeResolver._per_position`` already states for dividing a share balance:
    *"the split under which every package is hedged to the same fraction of its
    own exposure, which is the split the band rule would have produced had it
    been applied package by package from flat."*  So the question "who owns
    these shares" is answered here the same way the hedger answers it, rather
    than by a rule invented to make a metric come out.

    **What it costs, stated plainly.**  Pro rata on the absolute delta charges a
    position in proportion to the exposure it *brings*, not the exposure it
    *causes*.  A position that offsets the rest of the book reduces the hedge
    the group needs and is still charged a positive share of what the hedge
    cost.  There is no split that avoids this and also conserves -- netting
    benefits are joint, and joint benefits have no non-arbitrary division.  That
    is why ``attribution.orders_needing_split`` travels with these numbers: an
    order with a single owner is attributed exactly, with no convention in play
    at all, and the fraction of orders that needed a split is the fraction of
    this answer that rests on the rule rather than on the data.

    **These are not comparable with the per-position figures from
    ``trip_metrics``, and the difference is not error.** Per-position hedging
    nets nothing and is strictly the more expensive hedge; portfolio-level nets
    and is cheaper. The two ends of the bracket are supposed to disagree.
    """
    if not per_trip:
        return {
            "trades": 0,
            "win_rate": None,
            "profit_loss_ratio": None,
            "win_rate_option_only": None,
            "profit_loss_ratio_option_only": None,
            "hedge_profit_attributed": 0.0,
            "attribution": dict(attribution),
        }
    hedged = [float(t["profit"]) for t in per_trip]
    option = [float(t["option_profit"]) for t in per_trip]

    def pair(values: Sequence[float]) -> tuple[float, float | None]:
        wins = [v for v in values if v > 0.0]
        losses = [v for v in values if v < 0.0]
        plr = (fmean(wins) / abs(fmean(losses))) if wins and losses else None
        return len(wins) / len(values), plr

    win_rate, plr = pair(hedged)
    win_rate_option, plr_option = pair(option)
    return {
        "trades": len(per_trip),
        # Unlike the search's ``win_rate``, this is **not** 1.0 by construction.
        # The schedule was filtered on the *search's* hedged profit, which is a
        # per-position number; replayed portfolio-level on a running NAV, a trip
        # the flow banked can come back a loser.  A value below 1.0 here is the
        # honest cost of the bracket having two ends, not a bug.
        "win_rate": win_rate,
        "profit_loss_ratio": plr,
        "win_rate_option_only": win_rate_option,
        "profit_loss_ratio_option_only": plr_option,
        "hedge_profit_attributed": sum(h - o for h, o in zip(hedged, option, strict=True)),
        "attribution": dict(attribution),
    }


def trip_metrics(selected: Sequence[Trip], *, grid: Sequence) -> dict:
    """WR, PLR and HP over the chosen schedule.

    **WR and PLR are on hedged PnL** (user ruling, 2026-09-23: *"WR/PLR use
    hedged PnL"*).  They read ``Trip.profit``, which under ``--hedge`` is the
    package round trip *including its own share leg* -- entry spread, exit
    spread, both legs of fees, and every share ``cash_delta`` the hedge paid.
    The option-only pair is kept beside it, off ``Trip.option_profit``, because
    the difference between the two is the only per-trip statement of what the
    hedge cost.

    **These are the search's hedge, not the replay's, and that is a real
    caveat.**  The allocation problem that kept these null until today has not
    been solved -- it has been made irrelevant at this end of the bracket.
    ``hedged_best_exits`` hedges each package *per position* on its own share
    ledger, so a per-trip hedged PnL exists without inventing a split.  The
    attainable end still nets portfolio-level and still has no per-trade
    decomposition; ``replay_hedged`` reports its hedge as a residual and no
    per-trip figure here refers to it.  So WR and PLR describe the schedule **as
    scored**, which is the schedule the flow actually chose.

    Under ``--hedge volatility`` a directional trip never took a share leg, so
    its ``profit`` and ``option_profit`` are the same number and it contributes
    identically to both pairs.  The two diverge only on the hedged families,
    which is exactly where the question was.

    HP needs no convention: it is a property of the schedule, not of the PnL.
    Measured in distinct trading days spanned rather than in grid indices, so it
    means the same thing under ``PM`` and under ``AM,PM``.
    """
    if not selected:
        return {
            "trades": 0,
            "win_rate": None,
            "profit_loss_ratio": None,
            "win_rate_option_only": None,
            "profit_loss_ratio_option_only": None,
            "avg_holding_trading_days": None,
            "avg_holding_calendar_days": None,
        }
    wins = [t.profit for t in selected if t.profit > 0.0]
    losses = [t.profit for t in selected if t.profit < 0.0]
    option_wins = [t.option_profit for t in selected if t.option_profit > 0.0]
    option_losses = [t.option_profit for t in selected if t.option_profit < 0.0]
    held_sessions = [
        (t.candidate.open_index, t.close_index) for t in selected
    ]
    trading_days = [
        len({grid[i].trade_date for i in range(o, c + 1)}) - 1 for o, c in held_sessions
    ]
    calendar_days = [
        (grid[c].trade_date - grid[o].trade_date).days for o, c in held_sessions
    ]
    return {
        "trades": len(selected),
        # Only trips with positive *hedged* profit are enumerated upstream, so on
        # the oracle's own output this is 1.0 by construction and is *not*
        # evidence that the schedule is good.  Kept because the same function has
        # to read a policy run, where it is informative, and because a value
        # below 1.0 here would mean the profit filter had stopped working.
        "win_rate": len(wins) / len(selected),
        "profit_loss_ratio": (
            (fmean(wins) / abs(fmean(losses))) if wins and losses else None
        ),
        # The option-only pair is *not* 1.0 by construction under ``--hedge``:
        # the filter runs on hedged profit, and the headline walk test is a case
        # where the banked exit has a losing option leg.  A hedged win rate of
        # 1.0 beside an option-only rate below it is the schedule saying how much
        # of its profit is the share leg.
        "win_rate_option_only": len(option_wins) / len(selected),
        "profit_loss_ratio_option_only": (
            (fmean(option_wins) / abs(fmean(option_losses)))
            if option_wins and option_losses
            else None
        ),
        "avg_holding_trading_days": fmean(trading_days),
        "avg_holding_calendar_days": fmean(calendar_days),
    }


def select_greedy(
    trips: Sequence[Trip],
    *,
    cap: int,
    steps: int,
    per_underlying: int | None = None,
    per_step: int | None = None,
) -> list[Trip]:
    """Highest profit first, skipping anything that would exceed a cap.

    This is the *lower* end of the bracket and the only schedule here that obeys
    every environment rule, because unlike the flow it can see the name on a
    position and so can honour ``per_underlying``, and it can see which step an
    order lands in and so can honour ``per_step``.  Being feasible by
    construction, its total is a result the environment would genuinely have
    permitted, and the flow — which optimises over a relaxation — must never come
    out below it.  If it does, the flow is wrong.

    ``per_step`` is ``max_orders_per_step``, and admitting a trip spends a line
    at its open step and one at its close step unless a roll absorbs it; see
    ``_lines``.  The rejection is checked against the roll-credited charge, so a
    trip whose open pairs with an already-taken close on the same name and head
    is free at that step.
    """
    occupancy = [0] * (steps + 1)
    by_name: dict[tuple[int, str], int] = {}
    opens_at: dict[int, dict[RollKey, int]] = {}
    closes_at: dict[int, dict[RollKey, int]] = {}
    taken: list[Trip] = []
    for trip in sorted(trips, key=lambda t: -t.profit):
        span = range(trip.candidate.open_index, trip.close_index)
        name = trip.candidate.underlying
        if any(occupancy[i] >= cap for i in span):
            continue
        if per_underlying is not None and any(
            by_name.get((i, name), 0) >= per_underlying for i in span
        ):
            continue
        key = (name, trip.candidate.head)
        opening, closing = trip.candidate.open_index, trip.close_index
        at_open = opens_at.setdefault(opening, {})
        at_close = closes_at.setdefault(closing, {})
        # Tentatively, then rolled back on refusal.  The per-step counters are
        # tiny, so recomputing the whole charge beats reasoning about a delta
        # that would have to know whether this order landed on the matched or
        # the unmatched side of the pairing.
        at_open[key] = at_open.get(key, 0) + 1
        at_close[key] = at_close.get(key, 0) + 1
        if per_step is not None and (
            _lines(at_open, closes_at.get(opening, {})) > per_step
            or _lines(opens_at.get(closing, {}), at_close) > per_step
        ):
            at_open[key] -= 1
            at_close[key] -= 1
            continue
        for i in span:
            occupancy[i] += 1
            by_name[(i, name)] = by_name.get((i, name), 0) + 1
        taken.append(trip)
    return taken


def per_underlying_excess(selected: Sequence[Trip], *, limit: int) -> list[dict]:
    """Every (step, name) at which the chosen schedule holds more than ``limit``.

    The flow enforces the global slot cap structurally but is blind to a cap that
    is per name, because that couples arcs which share an underlying and a flow
    has no way to see a label on an arc.  Rather than approximate it inside the
    optimisation — which would make the result neither exact nor a bound — the
    breach is measured here and published.

    A ceiling computed with breaches in it is still a ceiling; it is just a
    ceiling on a slightly larger action set than the environment allows, so it
    stays an upper bound on the achievable result.  The danger is only in not
    knowing, which is what this returns.
    """
    held: dict[tuple[int, str], int] = {}
    for trip in selected:
        for step in range(trip.candidate.open_index, trip.close_index):
            key = (step, trip.candidate.underlying)
            held[key] = held.get(key, 0) + 1
    return [
        {"step": step, "underlying": name, "held": count, "limit": limit}
        for (step, name), count in sorted(held.items())
        if count > limit
    ]


def _lines(opens: Mapping[RollKey, int], closes: Mapping[RollKey, int]) -> int:
    """The fewest order lines that express these opens and closes in one step.

    A trip costs two lines in general -- an ``O`` where it starts and a ``C``
    where it ends -- but a close and an open in the *same* step on the same
    underlying and the same head are one ``X``, not two lines.  ``X`` is the
    weakest verb: it may change tenor and coordinates and nothing else, so
    ``(underlying, head)`` is exactly the equivalence it preserves, ``head``
    being family plus orientation.

    Pairing greedily within a key is optimal because edges exist only between
    identical keys, so the maximum matching is ``sum(min(opens, closes))`` per
    key and no cleverer assignment exists.

    Charging two lines per trip regardless would be the safe-looking choice and
    it is the wrong one: it would understate the ceiling for an encoding reason,
    which is the exact cost the roll verb was introduced to remove.
    """
    matched = sum(min(opens[k], closes[k]) for k in opens.keys() & closes.keys())
    return sum(opens.values()) + sum(closes.values()) - matched


def order_lines_per_step(
    selected: Sequence[Trip],
) -> dict[int, tuple[int, int]]:
    """``step -> (lines needed, rolls credited)`` for a schedule."""
    opens: dict[int, dict[RollKey, int]] = {}
    closes: dict[int, dict[RollKey, int]] = {}
    for trip in selected:
        key = (trip.candidate.underlying, trip.candidate.head)
        at_open = opens.setdefault(trip.candidate.open_index, {})
        at_open[key] = at_open.get(key, 0) + 1
        at_close = closes.setdefault(trip.close_index, {})
        at_close[key] = at_close.get(key, 0) + 1
    charges: dict[int, tuple[int, int]] = {}
    for step in sorted(opens.keys() | closes.keys()):
        o = opens.get(step, {})
        c = closes.get(step, {})
        lines = _lines(o, c)
        charges[step] = (lines, sum(o.values()) + sum(c.values()) - lines)
    return charges


def orders_per_step_excess(selected: Sequence[Trip], *, limit: int) -> list[dict]:
    """Every step at which the chosen schedule would have to send too many orders.

    ``max_orders_per_step`` caps the number of order *lines* in one completion.
    This is a second rule the flow cannot see, and for the same reason as the
    per-name cap: it couples arcs by the step they touch rather than by the slot
    they occupy.

    Unlike the per-name cap it also binds on *greedy*, and hard -- measured on
    2025-04-01..04-21, greedy wanted 18 lines at one step against a limit of 8,
    and breached at 9 of 28 steps.  That made the published lower end of the
    bracket not a lower bound at all, because it was not a schedule the
    environment would have accepted.  Greedy now honours it; this stays as the
    check that it did, and is what must be empty for the attainable schedule.
    """
    return [
        {"step": step, "orders": lines, "limit": limit, "rolls_credited": rolled}
        for step, (lines, rolled) in order_lines_per_step(selected).items()
        if lines > limit
    ]


def select_flow(trips: Sequence[Trip], *, cap: int, steps: int) -> list[Trip]:
    """The exact best set of trips that never holds more than ``cap`` at once.

    **Why this is a flow and not a knapsack.**  The binding resource is a
    position *slot* held over an interval, and slots are interchangeable.  So
    model time as a line of nodes, one per decision step, and push ``cap`` units
    of "slot" from the first to the last.  A unit crossing step ``t`` on the
    zero-cost arc ``t -> t+1`` is a slot sitting idle; a unit crossing on a
    trip's ``open -> close`` arc is that slot occupied.  Every cut between
    consecutive steps carries exactly ``cap`` units, which is precisely the
    concurrency constraint — stated once, structurally, rather than checked.

    What enforces that is the **flow value**, not the capacity on the idle arcs:
    exactly ``cap`` units are pushed from the first node to the last, so every
    cut carries ``cap`` no matter how wide the idle arcs are.  Their capacity is
    set to ``cap`` for readability only and could be infinite without changing
    the answer — verified by mutation, since a reader who believed otherwise
    would look in the wrong place to change the limit.

    Giving each trip arc cost ``-profit`` makes the minimum-cost flow the
    maximum-profit schedule.  This is exact, where greedy is not: greedy will
    take a single large trip that straddles a week and forgo the six smaller
    ones that fit inside it, and no amount of tie-breaking fixes that, because
    the decision is not local.

    The residual graph never has a negative cycle — the original graph is a DAG,
    since every arc runs forward in time — so successive shortest paths with
    Johnson potentials is valid, and the first potentials come free from a
    single pass in node order.  ``cap`` is 12, so twelve Dijkstras settle it.

    ``max_positions_per_underlying`` is **not** expressible here: it couples
    arcs that share a name, which a flow cannot see. It is checked afterwards
    and reported, never silently assumed.
    """
    n = steps + 1
    # Adjacency as parallel arrays: head, capacity, cost, and the index of the
    # paired residual arc.  A dict-of-objects graph is 20x slower here and this
    # runs over millions of arcs.
    head: list[int] = []
    cap_: list[int] = []
    cost: list[float] = []
    graph: list[list[int]] = [[] for _ in range(n)]
    owner: dict[int, Trip] = {}

    def add(u: int, v: int, capacity: int, price: float, trip: Trip | None = None) -> None:
        if trip is not None:
            owner[len(head)] = trip
        graph[u].append(len(head))
        head.append(v)
        cap_.append(capacity)
        cost.append(price)
        graph[v].append(len(head))
        head.append(u)
        cap_.append(0)
        cost.append(-price)

    for t in range(steps):
        add(t, t + 1, cap, 0.0)
    for trip in trips:
        add(trip.candidate.open_index, trip.close_index, 1, -trip.profit, trip)

    source, sink = 0, steps
    # Initial potentials by one relaxation pass in topological (== index) order,
    # which is valid because every forward arc goes from a lower index to a
    # higher one.
    potential = [0.0] * n
    for u in range(n):
        for arc in graph[u]:
            if cap_[arc] > 0 and potential[u] + cost[arc] < potential[head[arc]]:
                potential[head[arc]] = potential[u] + cost[arc]

    # This, not the idle-arc capacity, is the concurrency limit: pushing exactly
    # ``cap`` units means every cut across the timeline carries ``cap``.
    remaining = cap
    while remaining > 0:
        dist = [float("inf")] * n
        dist[source] = 0.0
        prev_arc: list[int] = [-1] * n
        visited = [False] * n
        heap: list[tuple[float, int]] = [(0.0, source)]
        while heap:
            d, u = heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for arc in graph[u]:
                if cap_[arc] <= 0:
                    continue
                v = head[arc]
                nd = d + cost[arc] + potential[u] - potential[v]
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    prev_arc[v] = arc
                    heappush(heap, (nd, v))
        if dist[sink] == float("inf"):
            break
        for u in range(n):
            if dist[u] < float("inf"):
                potential[u] += dist[u]
        # Bottleneck is always 1 when the path uses a trip arc and up to
        # ``remaining`` when it is all idle; pushing one unit at a time keeps
        # the accounting trivial and costs at most ``cap`` iterations.
        push = remaining
        v = sink
        while v != source:
            arc = prev_arc[v]
            push = min(push, cap_[arc])
            v = head[arc ^ 1]
        v = sink
        while v != source:
            arc = prev_arc[v]
            cap_[arc] -= push
            cap_[arc ^ 1] += push
            v = head[arc ^ 1]
        remaining -= push

    return [trip for arc, trip in owner.items() if cap_[arc] == 0]


def _by(trips: Sequence[Trip], key) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for trip in trips:
        bucket = out.setdefault(str(key(trip)), {"trips": 0, "profit": 0.0})
        bucket["trips"] += 1
        bucket["profit"] += trip.profit
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["profit"]))


if __name__ == "__main__":
    raise SystemExit(main())
