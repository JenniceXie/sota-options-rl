"""ContractResolver: strategy intent -> concrete legs, or a named failure.

``docs/env_contract.md`` section 6.  The policy names a family, an orientation
and delta coordinates; this turns that into contracts that exist, are quoted,
and satisfy the topology the family claims.

Two design commitments.

**One expiry per package, chosen once.**  Every admitted family requires the
same expiry across legs (calendars and diagonals are the two that do not, and
both are excluded).  So the expiry is picked first, from the DTE bucket, and
every leg resolves within it.  Resolving legs independently would let a
vertical straddle two expiry cycles and stop being a vertical.

**Topology is verified after resolution, not assumed from the family.**  Delta
targets are approximate — the resolver takes the nearest listed strike — and
two nearby delta targets on a coarse strike ladder can collapse onto the same
contract or invert their order.  A "credit vertical" whose short and long legs
resolved to the same strike is worth exactly zero and has zero max loss, so it
would pass every risk gate and be sized to the cap.  ``_verify`` is what stops
that, and it is the reason resolution can fail with ``E_TOPOLOGY`` even when
every individual leg resolved fine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from ..actions import OpenOrder
from ..chain import ChainQuote, ChainSlice
from ..payoff import PayoffLeg, is_unbounded, max_loss
from ..spec import EnvConfig

__all__ = ["ResolvedLeg", "ResolvedPackage", "ResolutionFailure", "ContractResolver", "RESOLVER_VERSION"]

RESOLVER_VERSION = "contract_resolver.v1"


@dataclass(frozen=True, slots=True)
class ResolvedLeg:
    quote: ChainQuote
    ratio: int
    role: str

    @property
    def contract_id(self) -> str:
        return self.quote.contract_id


@dataclass(frozen=True, slots=True)
class ResolvedPackage:
    """One package at unit quantity.  Size is decided later, by ``SizeResolver``."""

    order: OpenOrder
    legs: tuple[ResolvedLeg, ...]
    expiry: date
    dte: int
    mid_cost: float
    half_spread_cost: float
    #: Sum of ``|ratio| * mid * multiplier`` over the legs -- what the package
    #: costs to put on *gross*, before the short legs net against the long ones.
    #: Unlike ``mid_cost`` it is unsigned and strictly positive, which is the
    #: reason it exists: ``mid_cost`` is a net and is *negative* for the four
    #: credit families, so any rule dividing by "the cost of the package" gets a
    #: negative answer on exactly the structures the grammar was widened to
    #: admit.  ``package_relative_spread`` below already divides by it for the
    #: same reason.
    gross_premium: float
    max_loss: float
    package_relative_spread: float
    delta: float
    gamma: float
    vega: float
    theta: float
    underlying_price: float
    resolver_version: str = RESOLVER_VERSION

    @property
    def is_debit(self) -> bool:
        return self.mid_cost > 0

    def payoff_legs(self, prices: Mapping[str, float] | None = None) -> tuple[PayoffLeg, ...]:
        return tuple(
            PayoffLeg(
                right=leg.quote.right,
                strike=leg.quote.strike,
                ratio=leg.ratio,
                entry_price=(prices or {}).get(leg.contract_id, leg.quote.mid),
                multiplier=leg.quote.multiplier,
            )
            for leg in self.legs
        )


@dataclass(frozen=True, slots=True)
class ResolutionFailure:
    code: str
    detail: str
    order: OpenOrder
    diagnostics: Mapping[str, object] = None  # type: ignore[assignment]

    def as_result(self) -> dict[str, str]:
        return {"order": self.order.raw, "status": self.code, "detail": self.detail}


#: ``(role, right_for_bullish, delta_expression, ratio)`` per family.
#: ``delta_expression`` is evaluated against the order's coordinates.  Keeping
#: the topology as data rather than nine branches means a new family is a table
#: entry, and means ``tests`` can enumerate every family mechanically.
_TOPOLOGY: Mapping[str, tuple[tuple[str, str, str, int], ...]] = {
    "outright": (("long", "same", "long_delta", +1),),
    "debit_vertical": (
        ("long", "same", "long_delta", +1),
        ("short", "same", "long_delta - width_delta", -1),
    ),
    "credit_vertical": (
        ("short", "opposite", "short_delta", -1),
        ("long", "opposite", "short_delta - width_delta", +1),
    ),
    "defined_risk_reversal": (
        ("long", "same", "directional_delta", +1),
        ("short", "opposite", "directional_delta", -1),
        ("wing", "opposite", "tail_wing_delta", +1),
    ),
    "butterfly": (
        ("lower", "same", "center_delta + lower_width_delta", +1),
        ("body", "same", "center_delta", -2),
        ("upper", "same", "center_delta - upper_width_delta", +1),
    ),
    "long_straddle": (
        ("call", "call", "call_delta", +1),
        ("put", "put", "put_delta", +1),
    ),
    "long_strangle": (
        ("put", "put", "put_delta", +1),
        ("call", "call", "call_delta", +1),
    ),
    "iron_butterfly": (
        ("short_call", "call", "short_delta", -1),
        ("short_put", "put", "short_delta", -1),
        ("long_call", "call", "short_delta - wing_delta", +1),
        ("long_put", "put", "short_delta - wing_delta", +1),
    ),
    "iron_condor": (
        ("short_call", "call", "short_delta", -1),
        ("long_call", "call", "short_delta - width_delta", +1),
        ("short_put", "put", "short_delta", -1),
        ("long_put", "put", "short_delta - width_delta", +1),
    ),
}


class ContractResolver:
    """Turns one ``OpenOrder`` into a ``ResolvedPackage`` or a failure."""

    version = RESOLVER_VERSION

    def __init__(self, config: EnvConfig) -> None:
        self._config = config

    def resolve(self, order: OpenOrder, chain: ChainSlice) -> ResolvedPackage | ResolutionFailure:
        config = self._config
        topology = _TOPOLOGY.get(order.family)
        if topology is None:
            return ResolutionFailure("E_UNKNOWN_FAMILY", f"no topology for {order.family}", order)

        expiry = self._pick_expiry(order.tenor_bucket, chain)
        if expiry is None:
            low, high = config.tenor.ranges[order.tenor_bucket]
            listed = [q.dte for q in chain.quotes]
            return ResolutionFailure(
                "E_TENOR",
                f"no expiry in {low}-{high} DTE; listed {sorted(set(listed))[:6]}",
                order,
                {"rejected": dict(chain.rejected), "n_raw": chain.n_raw},
            )

        legs: list[ResolvedLeg] = []
        used: set[str] = set()
        for role, right_rule, expression, ratio in topology:
            right = _right_for(right_rule, order.orientation)
            target = _evaluate(expression, order.coordinates)
            if not 0.0 < target < 1.0:
                return ResolutionFailure(
                    "E_COORD_BOUNDS", f"{role} leg targets delta {target:.2f}", order
                )
            quote = chain.nearest_delta(expiry, right, target, exclude=frozenset(used))
            if quote is None:
                return ResolutionFailure(
                    "E_NO_CONTRACT",
                    f"no admitted {right} at {expiry.isoformat()} for the {role} leg",
                    order,
                    {"rejected": dict(chain.rejected)},
                )
            used.add(quote.contract_id)
            legs.append(ResolvedLeg(quote=quote, ratio=ratio, role=role))

        problem = self._verify(order, legs)
        if problem is not None:
            return ResolutionFailure("E_TOPOLOGY", problem, order)

        package = self._price(order, tuple(legs), expiry, chain)

        # The debit/credit sign is the family's economic claim, and it is the
        # one topology property that survives orientation unchanged.  A "credit
        # vertical" that resolved to a net debit is not a mispriced version of
        # what was asked for; it is the other family.
        if order.family.startswith("debit") and package.mid_cost <= 0:
            return ResolutionFailure("E_TOPOLOGY", "debit structure resolved to a credit", order)
        if order.family.startswith("credit") and package.mid_cost >= 0:
            return ResolutionFailure("E_TOPOLOGY", "credit structure resolved to a debit", order)

        if package.package_relative_spread > config.resolver.max_relative_spread_package:
            return ResolutionFailure(
                "E_WIDE_PACKAGE",
                f"package spread {package.package_relative_spread:.1%} exceeds "
                f"{config.resolver.max_relative_spread_package:.1%}",
                order,
            )
        if package.max_loss == float("inf"):
            # Unreachable through the admitted families; kept because it is the
            # invariant that makes "no margin engine" safe, and an unenforced
            # invariant is a comment.
            return ResolutionFailure("E_NAKED", "package has unbounded loss", order)
        if package.max_loss <= 0:
            return ResolutionFailure(
                "E_TOPOLOGY", "package has zero max loss; legs collapsed", order
            )
        return package

    # -- pieces ----------------------------------------------------------

    def _pick_expiry(self, bucket: str, chain: ChainSlice) -> date | None:
        """Nearest listed expiry to the bucket's anchor, inside the bucket.

        The anchor matters because the listed ladder is sparse: on 2025-06-09
        the front expiries for every policy name are 11, 18, 24, 32 DTE, so
        "the shortest in the bucket" and "the closest to 14" are different
        contracts with materially different gamma.
        """
        low, high = self._config.tenor.ranges[bucket]
        anchor = self._config.tenor.preferred_dte[bucket]
        candidates = {q.expiry: q.dte for q in chain.quotes if low <= q.dte <= high}
        if not candidates:
            return None
        return min(candidates, key=lambda e: (abs(candidates[e] - anchor), candidates[e]))

    @staticmethod
    def _verify(order: OpenOrder, legs: Sequence[ResolvedLeg]) -> str | None:
        """Reject packages whose resolved strikes do not honour the topology.

        Every check here is expressed on the *strike-ordered ratio sequence*
        rather than on named roles, because a role's position in the ladder
        flips with orientation: the high-delta leg of a call structure is the
        low strike, and of a put structure the high strike.  Checking
        ``lower < body < upper`` by role passes for a call butterfly and fails
        for the identical put butterfly.  The ratio sequence does not have that
        problem — a 1/-2/1 in strike order is a butterfly on either right.
        """
        if len({leg.contract_id for leg in legs}) != len(legs):
            return "two legs resolved to the same contract"

        family = order.family
        shape = _ratio_shape(legs)

        if family in ("debit_vertical", "credit_vertical"):
            if len({leg.quote.strike for leg in legs}) != 2:
                return "both vertical legs resolved to the same strike"
        elif family == "butterfly":
            if _flatten(shape) != (1, -2, 1):
                return f"strike-ordered ratios {_flatten(shape)} are not a 1/-2/1 butterfly"
        elif family in ("iron_condor", "iron_butterfly"):
            flat = _flatten(shape)
            if flat not in ((1, -1, -1, 1), (1, -2, 1)):
                # (1, -2, 1) is the legitimate degenerate case: an iron
                # butterfly whose two short legs land on the same strike, which
                # is what "iron butterfly" means when the ladder cooperates.
                return f"strike-ordered ratios {flat} are not a four-leg iron structure"
        elif family == "long_strangle":
            if len({leg.quote.strike for leg in legs}) != 2:
                return "strangle legs resolved to the same strike"
        elif family == "defined_risk_reversal":
            short = next(leg for leg in legs if leg.role == "short")
            wing = next(leg for leg in legs if leg.role == "wing")
            if abs(wing.quote.delta) >= abs(short.quote.delta):
                return "tail wing is not further out of the money than the short leg"

        # The ban, checked on the resolved legs rather than trusted from the
        # family table: a topology typo that dropped a long leg would otherwise
        # produce a naked short that every later stage accepts.
        payoff_legs = [
            PayoffLeg(leg.quote.right, leg.quote.strike, leg.ratio, leg.quote.mid, leg.quote.multiplier)
            for leg in legs
        ]
        if is_unbounded(payoff_legs):
            return "resolved legs leave an uncovered short call"
        return None

    @staticmethod
    def _price(
        order: OpenOrder, legs: tuple[ResolvedLeg, ...], expiry: date, chain: ChainSlice
    ) -> ResolvedPackage:
        """Mid cost, half-spread cost and package greeks, at unit quantity.

        ``mid_cost`` is signed: positive is a net debit paid.  The half-spread
        is summed over the *absolute* leg ratios, because crossing costs the
        same whether the leg is being bought or sold — netting them would make
        a four-leg iron condor look as cheap to trade as an outright.
        """
        mid_cost = 0.0
        half_spread = 0.0
        gross = 0.0
        delta = gamma = vega = theta = 0.0
        for leg in legs:
            q = leg.quote
            notional = q.multiplier
            mid_cost += leg.ratio * q.mid * notional
            half_spread += abs(leg.ratio) * q.half_spread * notional
            gross += abs(leg.ratio) * q.mid * notional
            # ``model_delta`` is already signed by right, so a put contributes
            # negatively without any adjustment here.
            delta += leg.ratio * q.delta * notional
            gamma += leg.ratio * q.gamma * notional
            vega += leg.ratio * q.vega * notional
            theta += leg.ratio * q.theta * notional

        payoff_legs = [
            PayoffLeg(leg.quote.right, leg.quote.strike, leg.ratio, leg.quote.mid, leg.quote.multiplier)
            for leg in legs
        ]
        loss = max_loss(payoff_legs)
        spot = legs[0].quote.underlying_price

        return ResolvedPackage(
            order=order,
            legs=legs,
            expiry=expiry,
            dte=legs[0].quote.dte,
            mid_cost=mid_cost,
            half_spread_cost=half_spread,
            gross_premium=gross,
            max_loss=loss,
            # Relative to gross premium traded, not to the net: a credit spread
            # can have a near-zero net and a real cost to cross.
            package_relative_spread=0.0 if gross <= 0 else half_spread * 2 / gross,
            delta=delta * spot if spot else delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            underlying_price=spot,
        )


def _ratio_shape(legs: Sequence[ResolvedLeg]) -> tuple[tuple[int, ...], ...]:
    """Ratios grouped by strike, in ascending strike order.

    Grouping rather than listing matters for the iron butterfly, whose two
    short legs are meant to share a strike: as a flat list they read as two
    entries, as a group they read as the single ``-2`` body that they
    economically are.
    """
    grouped: dict[float, list[int]] = {}
    for leg in legs:
        grouped.setdefault(leg.quote.strike, []).append(leg.ratio)
    return tuple(tuple(grouped[strike]) for strike in sorted(grouped))


def _flatten(shape: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sum(group) for group in shape)


def _right_for(rule: str, orientation: str) -> str:
    if rule in ("call", "put"):
        return rule
    directional = "call" if orientation == "bullish" else "put"
    if rule == "same":
        return directional
    return "put" if directional == "call" else "call"


def _evaluate(expression: str, coordinates: Mapping[str, float]) -> float:
    """Evaluate a ``a + b`` / ``a - b`` delta expression.

    Deliberately not ``eval``: the expressions are a closed set defined in
    ``_TOPOLOGY`` above, and a two-term parser cannot be made to execute
    anything.
    """
    for operator, sign in (("+", 1.0), ("-", -1.0)):
        if operator in expression:
            left, right = (part.strip() for part in expression.split(operator, 1))
            return coordinates[left] + sign * coordinates[right]
    return coordinates[expression.strip()]
