"""The four deterministic resolvers that stand between intent and the book.

The policy emits a strategy *intent*; these turn it into contracts, a size, a
mark and a hedge.  They are separate objects rather than one ``resolve()``
because each one is a different kind of decision with a different failure mode,
and the run manifest records their versions independently:

``ContractResolver``
    intent -> specific legs.  Fails when the chain cannot support the topology.
``SizeResolver``
    package -> contract count.  Fails when a risk limit binds.
``MarketResolver``
    book -> marks.  Never fails; degrades to ``stale`` or ``intrinsic``.
``HedgeResolver``
    book -> share orders.  Runs on the hedge grid, after the policy acts.

None of them mutates the book.  Each returns a value object that the execution
layer applies, so that the one place cash and positions change is the one place
the ledger is written.
"""

from __future__ import annotations

from .contract import (
    RESOLVER_VERSION as CONTRACT_RESOLVER_VERSION,
    ContractResolver,
    ResolutionFailure,
    ResolvedLeg,
    ResolvedPackage,
)
from .hedge import HEDGE_RESOLVER_VERSION, HedgePlan, HedgeResolver, ShareOrder
from .market import MARKET_RESOLVER_VERSION, MarketResolver, MarkReport
from .size import SIZE_RESOLVER_VERSION, SizeDecision, SizeRefusal, SizeResolver

__all__ = [
    "ContractResolver",
    "ResolvedLeg",
    "ResolvedPackage",
    "ResolutionFailure",
    "CONTRACT_RESOLVER_VERSION",
    "SizeResolver",
    "SizeDecision",
    "SizeRefusal",
    "SIZE_RESOLVER_VERSION",
    "MarketResolver",
    "MarkReport",
    "MARKET_RESOLVER_VERSION",
    "HedgeResolver",
    "HedgePlan",
    "ShareOrder",
    "HEDGE_RESOLVER_VERSION",
    "resolver_versions",
]


def resolver_versions() -> dict[str, str]:
    """The version stamp that goes into the run manifest and ``BookState``.

    Two runs that agree on this dict and on ``EnvConfig.fingerprint()`` resolved
    identical intents to identical contracts, which is the claim a reproduction
    needs to make.
    """
    return {
        "contract": CONTRACT_RESOLVER_VERSION,
        "size": SIZE_RESOLVER_VERSION,
        "market": MARKET_RESOLVER_VERSION,
        "hedge": HEDGE_RESOLVER_VERSION,
    }
