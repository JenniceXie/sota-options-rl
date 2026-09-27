"""Expiry payoff of a package, and the max loss derived from it.

``docs/env_contract.md`` section 8.4 makes collateral equal to cash-held max
loss, so max loss is load-bearing: it sets the collateral, the per-name risk
budget, the position size and the stop-loss denominator.  Getting it from a
per-family formula would mean nine formulas, nine chances to be wrong, and a
silent failure the moment a tenth family is added.

Instead it is *computed* from the leg set.  A European-style package's payoff is
piecewise linear in the terminal spot with kinks only at the strikes, so
evaluating at ``{0} ∪ strikes ∪ {2·max_strike}`` finds the true minimum exactly
— no grid, no tolerance.  That is a property of the payoff, not an
approximation of it.

The same evaluation answers the question the naked-shorts ban asks.  A package
is unbounded above iff its net call ratio is negative, and unbounded below iff
its net put ratio is negative *and* the puts are not covered — both are read
off the leg set directly rather than asserted per family.  ``⟨Q10⟩`` says such
a package is inadmissible, so ``max_loss`` returns infinity and the caller
refuses the order rather than sizing it.

American exercise is not modelled here.  ``docs/env_contract.md`` section 8.4
declines a margin engine, and early assignment against a *covered* short leg
changes when the loss is realized, not how large it can be.  The residual is
early assignment breaking a topology, which section 8.7 handles as an event and
not as a collateral number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "PayoffLeg",
    "package_payoff",
    "max_loss",
    "is_unbounded",
    "breakeven_points",
    "value_bounds",
]


@dataclass(frozen=True, slots=True)
class PayoffLeg:
    """One leg at single-package quantity.

    ``ratio`` is signed: ``+1`` long, ``-1`` short, ``-2`` for the body of a
    butterfly.  ``entry_price`` is the per-contract price actually transacted,
    which is what makes the payoff a PnL rather than a terminal value.
    """

    right: str
    strike: float
    ratio: int
    entry_price: float
    multiplier: int = 100


def _intrinsic(right: str, strike: float, spot: float) -> float:
    return max(0.0, spot - strike) if right == "call" else max(0.0, strike - spot)


def package_payoff(legs: Sequence[PayoffLeg], spot: float) -> float:
    """PnL of the package at expiry for a terminal ``spot``, in account currency."""
    total = 0.0
    for leg in legs:
        intrinsic = _intrinsic(leg.right, leg.strike, spot)
        total += leg.ratio * (intrinsic - leg.entry_price) * leg.multiplier
    return total


def is_unbounded(legs: Sequence[PayoffLeg]) -> bool:
    """True when the loss has no finite bound in either tail.

    Above: the payoff slope as ``spot -> inf`` is the net call ratio, so a
    negative net call ratio loses without limit.  Below: as ``spot -> 0`` the
    slope is the negated net put ratio, so a negative net put ratio loses down
    to zero — bounded in principle, but only because the underlying cannot go
    below zero, and that bound is still finite so it is *not* reported here.
    Only the upside case is genuinely unbounded, which is why a cash-secured
    put is admissible and a naked call is not.
    """
    net_calls = sum(leg.ratio for leg in legs if leg.right == "call")
    return net_calls < 0


def breakeven_points(legs: Sequence[PayoffLeg]) -> tuple[float, ...]:
    """Evaluation points that are guaranteed to contain the payoff minimum."""
    strikes = sorted({leg.strike for leg in legs})
    if not strikes:
        return (0.0,)
    return (0.0, *strikes, strikes[-1] * 2.0)


def value_bounds(legs: Sequence[PayoffLeg]) -> tuple[float, float]:
    """The interval a package's *value* can occupy, at any time before expiry.

    Value, not PnL: ``entry_price`` is ignored, so this is what the package is
    worth rather than what it made.  Returns ``(low, high)`` per package, in
    account currency, with ``high`` possibly ``inf``.

    **Why the terminal range bounds the value at every earlier time.**  The
    package's value is the risk-neutral expectation of its payoff at exercise,
    and every attainable payoff lies between the minimum and maximum over
    terminal spot.  An expectation of a quantity confined to an interval is
    confined to that interval, so the bound holds at any ``t`` — and it holds
    for American exercise too, because the holder's optimal stopping time is
    still a stopping time and the payoff at it is still in the same range.
    Both facts require one expiry, which every admitted family has; a calendar
    spread would break this and ``⟨Q10⟩`` does not admit one.

    Evaluated at ``breakeven_points`` for the same reason ``max_loss`` is: the
    payoff is piecewise linear with kinks only at the strikes, so the endpoints
    of the interval are attained exactly and not approximated.

    This is the check the mixed-provenance mark defeated.  On 2025-04-07 an
    AAPL 220/222.5/225 put butterfly was closed at 38.425 per share against a
    range of ``[0.0, 2.5]`` — 15.4x the most it could ever have been worth — and
    booked +$1,188,243 on a structure whose maximum profit was $74,167.  A
    per-family formula would not have caught it; the leg set does.
    """
    if not legs:
        return (0.0, 0.0)

    def value_at(spot: float) -> float:
        return sum(
            leg.ratio * _intrinsic(leg.right, leg.strike, spot) * leg.multiplier
            for leg in legs
        )

    points = breakeven_points(legs)
    low = min(value_at(spot) for spot in points)
    high = max(value_at(spot) for spot in points)

    # Beyond the largest strike every payoff is linear in spot with slope equal
    # to the net call ratio, so a positive net ratio runs away upward and the
    # sampled maximum is merely the last point looked at rather than a bound.
    # The mirror case below zero cannot arise: spot is floored at 0 and that
    # endpoint is already sampled.
    if sum(leg.ratio for leg in legs if leg.right == "call") > 0:
        high = float("inf")
    return (low, high)


def max_loss(legs: Sequence[PayoffLeg]) -> float:
    """Worst-case loss of one package, non-negative.

    Returns ``inf`` for an unbounded package so that every downstream
    consumer — collateral, the per-name budget, sizing — degrades to "refuse"
    rather than to a large-but-finite number that would still be sized and
    filled.
    """
    if not legs:
        return 0.0
    if is_unbounded(legs):
        return float("inf")
    worst = min(package_payoff(legs, spot) for spot in breakeven_points(legs))
    return max(0.0, -worst)
