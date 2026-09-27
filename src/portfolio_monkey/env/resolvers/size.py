"""SizeResolver: the only place a position size is decided.

``docs/env_contract.md`` section 6A.  Size left the action space (``⟨Q11⟩``) so
the policy expresses a view and never a contract count.  Everything that makes
a book survive — the per-family risk unit, the per-name budget, the portfolio
cap, the delta cap and the cash floor — is enforced here, once, on the way in.

The rule is a **minimum over the limits ``config.size.size_rule`` selects**, not
a sum and not a priority order.  The full vocabulary is:

1. ``scenario``     — the one-day risk budget from the Taylor expansion
2. ``nav_fraction`` — gross premium as a fraction of NAV.  *This is the size*
   under the shipped default.
3. ``name``         — remaining budget on this underlying
4. ``total``        — remaining budget across the book
5. ``delta``        — remaining net dollar delta headroom
6. ``cash``         — collateral plus debit must fit in buying power

**The ``max_loss`` ceiling used to be in this list and was removed 2026-09-22**
by user ruling (*"drop the max_loss_cap from the size bounds"*).  It was
``floor(cap_for(family) * NAV / max_loss)``, it was the binding limit on
essentially every package the oracle selected, and that is why it went: it made
``nav_fraction`` -- the default and the toggle-OFF arm -- behave as a max-loss
rule wearing a premium label, so sweeping ``f`` moved nothing.  The two rules
that were built out of it, ``max_loss`` and ``scenario_max_loss``, no longer
exist.  See ``spec.SIZE_RULES``.

Read the historical measurement in that light: across 77 ``algo_v2_core`` arms
and 13,770 approved fills the family/max-loss limit bound 99.5% of them and
``delta`` 0.5%, while ``name``, ``total`` and ``cash`` bound *zero* times.  The
limit that bound 99.5% is the one now deleted, so that measurement says what the
*old* sizer did and must not be quoted as what this one does.  ``cash`` in
particular went from unreachable to load-bearing, because the arithmetic that
made it unreachable was ``max_positions * cap * (1 + buffer)`` = 25.2% of NAV,
and there is no ``cap`` any more.

**Solvency is no longer provable from the config.** It used to be: cash out per
package is at most ``max_loss * (1 + collateral_buffer)``, the ceiling bounded
``max_loss`` per package, so the book could not commit more than that product.
Every surviving rule therefore carries ``cash``, and ``EnvConfig.__post_init__``
refuses any rule that does not.  The guarantee is now per-order and exact rather
than a-priori and conservative.

Taking the minimum is what makes the limits composable.  If they were applied
in sequence with early exit, the *order* of the checks would change the answer,
and the binding constraint would be whichever one happened to be tested first.
Under a minimum, ``binding`` is well defined and is reported back to the policy
— which matters, because "your NVDA budget is full" and "you are out of cash"
call for different next actions and a bare rejection tells the policy neither.

Sizes are integer packages and round **down**.  A package that rounds to zero
is refused rather than filled at one lot: the cheapest way to breach a risk cap
is to let every marginal order through as a single contract.

**Why the unit is a scenario and not ``max_loss``.**  Budgeting max loss reads
as risk parity and is not, because max loss is reachable for some structures
and fictional for others.  Measured over 687 PM position-steps under the old
rule, at an identical 1%-of-NAV max-loss budget, an outright carried a median
``|$delta|`` of 10,307 against a debit vertical's 1,123 and a per-step PnL
standard deviation of $6,681 against $2,019.  Nine times the exposure for the
same budget is a standing subsidy to the simplest structure, and the policy
collected it: 664 of those 687 steps were outrights, so eight of the nine
families in the grammar were being priced out by the sizing rule rather than
rejected on their merits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from ..book import Book
from ..spec import SIZE_RULE_LIMITS, EnvConfig
from .contract import ResolvedPackage

__all__ = ["SizeDecision", "SizeRefusal", "SizeResolver", "SIZE_RESOLVER_VERSION"]

SIZE_RESOLVER_VERSION = "size_resolver.v1"


@dataclass(frozen=True, slots=True)
class SizeDecision:
    """An approved package count, plus every number that produced it."""

    quantity: int
    collateral: float
    entry_debit: float
    risk_budget_used: float
    binding: str
    limits: Mapping[str, float]
    #: Which vol the spot shock was taken from: ``atm_iv_30d`` when the per-name
    #: surface supplied one, ``package_iv`` when it did not and the package's own
    #: vega-weighted leg vol stood in, ``none`` when no scenario was evaluated.
    #: Recorded on the fill rather than inferred, so an audit can *assert* the
    #: fallback never fired instead of trusting that it did not -- a silent
    #: substitution would reintroduce exactly the package-dependent shock the
    #: 2026-09-22 ruling removed, and nothing downstream would show it.
    spot_sigma_source: str = "none"
    resolver_version: str = SIZE_RESOLVER_VERSION

    @property
    def approved(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True, slots=True)
class SizeRefusal:
    code: str
    detail: str
    binding: str
    limits: Mapping[str, float] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return False


class SizeResolver:
    """Decides how many packages of a resolved intent the book may carry."""

    version = SIZE_RESOLVER_VERSION

    def __init__(self, config: EnvConfig) -> None:
        self._config = config

    def resolve(
        self,
        package: ResolvedPackage,
        book: Book,
        *,
        as_of: datetime,
        conviction: str | None = None,
        atm_iv: float | None = None,
    ) -> SizeDecision | SizeRefusal:
        """``atm_iv`` is the per-name 30-day ATM implied vol for the spot shock.

        Passed in rather than read here because the resolver has no feature
        source and should not grow one: the caller already holds a point-in-time
        gated view of the step, and giving the sizer its own reader would create
        a second path that could disagree with the state block about what the
        policy was shown.  ``None`` falls back to the package's own vol and says
        so on the decision -- see ``spot_sigma_source``.
        """
        config = self._config
        bounds = config.size
        nav = book.nav
        if nav <= 0:
            return SizeRefusal("E_LIMIT", "book has no equity", "nav")

        underlying = package.order.underlying
        family = package.order.family

        # Count caps are hard gates, not size inputs: they bound how many
        # *distinct* decisions are live, which is what keeps the POS block
        # inside the context budget (section 1.3 caps P at ~12-15).
        if len(book.positions) >= bounds.max_positions:
            return SizeRefusal(
                "E_LIMIT", f"{bounds.max_positions} positions already open", "max_positions"
            )
        held = len(book.positions_for(underlying))
        if held >= bounds.max_positions_per_underlying:
            return SizeRefusal(
                "E_LIMIT",
                f"{held} positions already open on {underlying}",
                "max_positions_per_underlying",
            )

        loss = package.max_loss
        if not math.isfinite(loss) or loss <= 0:
            return SizeRefusal("E_LIMIT", "package has no finite positive max loss", "max_loss")

        debit_per = max(0.0, package.mid_cost) + package.half_spread_cost
        # Only the part of the worst case the debit has not already funded.
        # Charging ``loss`` in full alongside the debit reserves the premium of
        # a long option twice -- it is spent, it is an asset in ``mtm_value``,
        # and it cannot be lost a second time.  Measured on every long call in
        # the four arms, ``collateral / mtm_value`` was 1.05 exactly, so the
        # book reserved a second premium against each one.  It never bound, but
        # it flows through ``collateral_used`` into ``buying_power``, so the
        # ``bp`` and ``util`` cells the policy reads were wrong by roughly the
        # whole options book.
        collateral_per = max(0.0, loss - debit_per) * (1.0 + bounds.collateral_buffer)

        multiplier = bounds.conviction_multipliers.get(conviction or "", 1.0)
        scenario_target = bounds.target_scenario_risk * multiplier

        # Computed lazily, one thunk per limit, so an inactive limit is not just
        # excluded from the minimum but never evaluated.  That matters for more
        # than speed: ``_delta_limit`` and the two budget limits read book state,
        # so an eagerly-built dict would make a rule that claims to ignore the
        # book still depend on it, and a bug in a disabled limit could still
        # raise.  "Off" should mean the code did not run.
        available = {
            "scenario": lambda: self._scenario_limit(package, scenario_target, nav, atm_iv),
            "nav_fraction": lambda: self._nav_fraction_limit(package, multiplier, nav),
            "name": lambda: max(
                0.0, bounds.max_risk_per_underlying - book.risk_for(underlying)
            ) * nav / loss,
            "total": lambda: max(
                0.0, bounds.max_total_open_risk - book.total_open_risk
            ) * nav / loss,
            "delta": lambda: self._delta_limit(package, book, nav),
            "cash": lambda: self._cash_limit(book, collateral_per, debit_per),
        }
        active = SIZE_RULE_LIMITS[bounds.size_rule]
        limits = {name: available[name]() for name in active}

        # Guarded by the membership test and not merely ordered after it: under
        # a rule without the scenario this must not touch a greek, or the
        # "off means the code did not run" tripwire in ``test_size_rules`` is a
        # lie for a field nobody reads.
        if "scenario" not in active or _package_iv(package) is None:
            spot_sigma_source = "none"
        elif atm_iv is not None and atm_iv > 0.0:
            spot_sigma_source = "atm_iv_30d"
        else:
            spot_sigma_source = "package_iv"

        # ``min`` over the dict would break the tie by insertion order, which is
        # the rule's declaration order and therefore arbitrary.  Ties happen: two
        # limits that are both ``inf`` is the ordinary case for a small package
        # under ``scenario``.  Sorting by ``(value, name)`` at least makes the
        # reported ``binding`` reproducible across runs and rules.
        binding = min(limits, key=lambda name: (limits[name], name))
        quantity = int(math.floor(min(limits.values())))

        if quantity < 1:
            return SizeRefusal(
                "E_LIMIT",
                f"{binding} limit allows {min(limits.values()):.2f} packages",
                binding,
                limits,
            )

        return SizeDecision(
            quantity=quantity,
            collateral=collateral_per * quantity,
            entry_debit=debit_per * quantity,
            risk_budget_used=loss * quantity / nav,
            binding=binding,
            limits=limits,
            spot_sigma_source=spot_sigma_source,
        )

    # -- individual limits -----------------------------------------------

    def _nav_fraction_limit(
        self, package: ResolvedPackage, conviction_multiplier: float, nav: float
    ) -> float:
        """Packages whose gross premium is at most ``nav_fraction``·NAV.

        The whole rule is one division.  That is the point of it: it is the
        control arm against which the scenario budget has to justify reading
        four greeks, an implied vol and a spot.  If a constant proportion of NAV
        scores the same, the Taylor expansion is not earning its complexity.

        **Gross, not net.**  ``mid_cost`` is signed and negative for ``cv``,
        ``ip``, ``ic`` and ``ib``, so dividing by it hands back a negative
        quantity on the four credit families -- which then floors to something
        below 1 and refuses, so every credit structure would be silently
        unsizeable under this rule and the arm would quietly become "outrights
        and debit spreads only".  Gross premium is the sum over ``|ratio|``, so
        it is positive whenever any leg has a positive mid.

        **Gross premium, and since 2026-09-22 that is the whole story.**  With
        the ceiling removed this is usually the binding limit, so what it does
        *not* bound is now the sizer's main exposure: it caps what a package
        costs, not what it can lose.  Ultimate risk per package is therefore
        ``f * (max_loss / gross_premium) * NAV`` -- equal to ``f * NAV`` for an
        outright, where max loss is the premium, and larger by exactly that ratio
        for anything whose worst case is a strike width.  Nothing here bounds the
        ratio, so nothing here bounds ultimate risk.

        Zero gross premium returns ``inf`` rather than refusing, matching
        ``_scenario_limit``: a package every leg of which is marked at zero is a
        pricing failure, and it is ``cash`` -- not this rule -- that owns the
        question of whether it can be funded.  The caller has already refused any
        package without a finite positive ``max_loss``.

        The conviction multiplier scales the budget here exactly as it scales
        ``target_scenario_risk`` for the scenario rule, so that switching rules
        does not silently disable conviction.
        """
        budget = self._config.size.nav_fraction * conviction_multiplier * nav
        gross = package.gross_premium
        if gross <= 0.0:
            return float("inf")
        return budget / gross

    def _scenario_limit(
        self,
        package: ResolvedPackage,
        target: float,
        nav: float,
        atm_iv: float | None,
    ) -> float:
        """Packages whose joint one-day adverse move costs at most ``target``·NAV.

        The scenario is a one-standard-deviation move in the underlying, taken
        in whichever direction hurts, *plus* an adverse move in implied vol,
        *plus* one day of carry:

            risk = |Δ$|·σ₁ᵈ − ½·Γ·(S·σ₁ᵈ)² + |ν|·(σ_pkg·shock) − Θ

        **The two vols are different on purpose** (2026-09-22,
        ``docs/env_contract.md`` 6A.1.1).  ``σ₁ᵈ = σ_ATM,name/√252`` is a
        property of the *underlying*: the spot moves the same way whichever
        strikes the package happens to sit at.  The vega term keeps ``σ_pkg``,
        the package's own vega-weighted leg vol, because that genuinely is the
        vol it is exposed to, read off the quotes it was priced against.

        Before that ruling both terms read ``σ_pkg``, which made a move in the
        underlying depend on the option: measured 2026-09-21, the ratio of the
        derived shock to the shortest-tenor ATM outright was 0.996 at 0-14 DTE
        but 0.809 at 31-60, so the same name on the same day got a 19% smaller
        spot move for sitting two months out.  ``atm_iv`` of ``None`` falls back
        to ``σ_pkg`` and the substitution is recorded on the decision rather
        than absorbed.

        ``scenario_sigmas`` (``k``) was retired in the same ruling.  It was
        pinned at 1.0, and 1.0 is precisely the value at which
        ``Θ_day = −½·Γ·(σ_day·S)²`` makes the gamma and theta terms cancel for a
        delta-hedged package.  Leaving it configurable made that identity a
        matter of configuration; removing it makes it a matter of construction.

        Signs are the point.  Long gamma subtracts, because a long-gamma package
        is helped by a move in either direction and should be allowed to be
        larger for the same budget; short gamma adds.  Theta subtracts when the
        package is paid to wait and adds when it pays, which is the whole
        distinction between a 7-DTE long straddle and a 7-DTE credit spread that
        ``max_loss`` sizing cannot see.

        Units were read off the chain rather than assumed, because the four
        greeks are in four different ones and a wrong guess mis-sizes silently
        rather than raising.  There are two levels, and conflating them is the
        easy mistake because the fields carry the same names at both.

        On ``ChainQuote`` everything is **per share**: ``delta`` dimensionless,
        ``gamma`` per $1 of spot, ``vega`` per **1.00** of vol and not per
        point, ``theta`` per **trading day** -- annual decay over 252, matching
        the ``σ/√252`` used for the move below.  Verified against Black-Scholes
        on a live SPY 550C at 44 DTE: vega 75.42 against S√Tφ(d₁) = 75.7, gamma
        0.012777 against φ(d₁)/(Sσ√T) = 0.01267.

        The theta day-count is load-bearing here rather than cosmetic.  For a
        delta-hedged package the gamma and theta terms are the same quantity
        with opposite signs -- Θ_day = −½·Γ·(σ_day·S)² -- so they cancel, and
        the scenario correctly charges such a package nothing for the pair.
        That cancellation only holds when both are divided into the same day.
        Under the 365-day theta this read until 2026-09-20, the two were
        1.4484x apart and 31.0% of the gamma term survived uncancelled: a
        standing credit to long gamma and charge to short gamma, invisible
        because each term was individually right.

        ``contract.py`` then scales all four by ``ratio * multiplier``, and
        ``delta`` alone by spot on top of that.  So the ``ResolvedPackage``
        fields this method reads are: ``delta`` dollars of exposure per 100%
        move; ``gamma`` **per package** per $1 -- the multiplier is already in
        it, so it is neither the per-share number nor dollar gamma, since no S²
        is applied; ``vega`` dollars per 1.00 of vol; ``theta`` dollars per
        trading day.

        Consistency of the scenario is a dimensional check, term by term, and
        it is worth doing rather than assuming: a ``gamma`` missing its
        multiplier would leave that term 100x too small, which in production is
        indistinguishable from a formula that works.

            |Δ$|·σ₁ᵈ        [$]·[1]           -> $
            ½·Γ·(S·σ₁ᵈ)²    [1/$]·[$²]        -> $
            |ν|·(σ·shock)   [$/1.00]·[1]      -> $
            Θ               [$/trading day]   -> $

        The corresponding fields on ``PositionMark`` are **not** these: that
        path dollarises ``gamma`` as ``Γ·S²/100`` and leaves ``vega``/``theta``
        raw, so the two conventions differ by ``S²/100`` on gamma alone.  A
        portfolio greek cap written against ``PositionMark`` cannot reuse the
        arithmetic here without converting.

        Non-positive risk returns ``inf`` rather than refusing.  A deep
        short-gamma package with large positive theta can genuinely show no
        one-day loss under this scenario, and that is not a reason to refuse it
        -- it is a reason to let another limit be the thing that bounds it, which
        is exactly what ``inf`` here arranges.  Until 2026-09-22 that other limit
        was the ``max_loss`` ceiling; it is now ``nav_fraction`` and ``cash``,
        which bound such a package by what it costs and what it ties up rather
        than by what it can ultimately lose.
        """
        sigma = _package_iv(package)
        spot = package.underlying_price
        if sigma is None or sigma <= 0.0 or spot <= 0.0:
            return float("inf")

        bounds = self._config.size
        reference = atm_iv if (atm_iv is not None and atm_iv > 0.0) else sigma
        shocked = reference / math.sqrt(bounds.trading_days_per_year)
        move = spot * shocked
        risk = (
            abs(package.delta) * shocked
            - 0.5 * package.gamma * move * move
            + abs(package.vega) * sigma * bounds.vol_shock_relative
            - package.theta
        )
        if risk <= 0.0:
            return float("inf")
        return target * nav / risk

    def _delta_limit(self, package: ResolvedPackage, book: Book, nav: float) -> float:
        """Packages that fit inside the remaining net dollar delta headroom.

        Signed, and it is the *net* that is capped: a bearish package on a book
        that is already long delta has unlimited headroom by this measure, which
        is correct — it reduces the exposure the cap exists to bound.  Only the
        component that pushes the net further from zero is charged.
        """
        cap = self._config.size.max_net_dollar_delta * nav
        current = sum(p.dollar_delta for p in book.positions.values())
        per_package = package.delta
        if abs(per_package) < 1e-9:
            return float("inf")
        if per_package > 0:
            headroom = cap - current
        else:
            headroom = cap + current
        if headroom <= 0:
            return 0.0
        return headroom / abs(per_package)

    @staticmethod
    def _cash_limit(book: Book, collateral_per: float, debit_per: float) -> float:
        """Packages that fit in buying power.

        Collateral and debit are charged together because both are cash that
        leaves the free balance at the same instant: with no margin engine
        (``⟨Q10⟩``) there is nothing else to fund either from.
        """
        per_package = collateral_per + debit_per
        if per_package <= 0:
            return float("inf")
        return max(0.0, book.buying_power) / per_package


def _package_iv(package: ResolvedPackage) -> float | None:
    """The package's own implied vol, weighted by where its vega sits.

    Taken from the legs rather than from a volatility feed so that sizing reads
    exactly the surface the package was priced against — no second source to
    disagree with the fill, and no new look-ahead surface to audit, because the
    quotes are already gated as of the decision instant.

    Weighting by ``|ratio·vega|`` and not averaging: a vertical's two legs sit
    at different strikes on a skewed smile, and the leg carrying the vega is the
    one whose vol the package is actually exposed to.  Falls back to a plain
    mean when every leg's vega is zero, which happens on the expiry-day rows
    where the chain zeroes greeks.
    """
    ivs = [leg.quote.iv for leg in package.legs if leg.quote.iv > 0.0]
    if not ivs:
        return None
    weights = [
        abs(leg.ratio * leg.quote.vega) for leg in package.legs if leg.quote.iv > 0.0
    ]
    total = sum(weights)
    if total <= 0.0:
        return sum(ivs) / len(ivs)
    return sum(iv * w for iv, w in zip(ivs, weights, strict=True)) / total
