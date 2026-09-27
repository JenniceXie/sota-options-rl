"""Stable option-strategy templates and canonical strategy handles.

The policy selects these relative-coordinate templates.  Exact contracts are a
point-in-time resolver concern and are deliberately absent from the handle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote


#: The version new handles are **emitted** at.
HANDLE_VERSION = "v2"

#: Versions that still parse but can no longer be written (user ruling,
#: 2026-09-22: *"Emit v2, freeze v1 for Q47."*).
#:
#: ``v1`` carried three ``coordinate_token`` literals that disagreed with the
#: coordinates in their own ``TemplateSpec`` -- ``defined_risk_reversal``
#: (``tw15`` against ``tail_wing_delta = 0.20``), ``butterfly`` (``d55`` against
#: ``center_delta = 0.45``) and ``long_straddle`` (``d50`` against a topology
#: that has two coordinates with distinct prefixes).  Because
#: ``parse_strategy_handle`` validates an incoming token against the literal
#: while ``env`` builds it from the coordinates the policy actually chose, a
#: handle emitted at the family default could not be read back.
#:
#: Freezing rather than rewriting is the whole point: ``d35:tw15`` is on disk in
#: all 246 ``features/option_candidates`` partitions and in the already-labelled
#: SFT cohort.  Those handles keep parsing, and keep meaning what they meant.
FROZEN_HANDLE_VERSIONS: tuple[str, ...] = ("v1",)
SUPPORTED_HANDLE_VERSIONS: tuple[str, ...] = (*FROZEN_HANDLE_VERSIONS, HANDLE_VERSION)

TENOR_BUCKETS = frozenset({"0_7", "8_30", "31_90", "91_180", "181_plus"})
_UNDERLYING_SAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-"

#: Declared coordinate order per family, with the handle prefix each one takes.
#:
#: **This is the only declaration.**  ``env.actions.COORDINATE_ORDER`` is this
#: object, imported -- it used to be a second copy, and a second copy is what
#: let ``long_straddle`` end up with one prefix here and two there.  It lives in
#: this module rather than in ``env`` because ``env.actions`` already imports
#: ``TEMPLATE_SPECS`` from here, so the dependency only runs one way.
#:
#: The order is positional on the wire, so it is part of the contract and must
#: not be reordered without a handle-version bump.
COORDINATE_ORDER: Mapping[str, tuple[tuple[str, str], ...]] = {
    "outright": (("long_delta", "d"),),
    "debit_vertical": (("long_delta", "d"), ("width_delta", "w")),
    "credit_vertical": (("short_delta", "d"), ("width_delta", "w")),
    "defined_risk_reversal": (("directional_delta", "d"), ("tail_wing_delta", "tw")),
    "butterfly": (
        ("center_delta", "d"),
        ("lower_width_delta", "lw"),
        ("upper_width_delta", "uw"),
    ),
    "long_straddle": (("call_delta", "dc"), ("put_delta", "dp")),
    "long_strangle": (("put_delta", "dp"), ("call_delta", "dc")),
    "iron_butterfly": (("short_delta", "d"), ("wing_delta", "w")),
    "iron_condor": (("short_delta", "d"), ("width_delta", "w")),
    # Not in ``env.actions.FAMILY_CODES``: the action space does not admit a
    # two-expiry structure, so these are feature-layer only.
    "calendar": (("strike_delta", "d"),),
    "diagonal": (("near_delta", "nd"), ("far_delta", "fd")),
}

#: What ``v1`` froze, for the three families where it disagreed with itself.
#: Read only when parsing a ``v1`` handle; never written.
V1_COORDINATE_TOKENS: Mapping[str, str] = {
    "defined_risk_reversal": "d35:tw15",
    "butterfly": "d55:lw10:uw10",
    "long_straddle": "d50",
}


def coordinate_order(family: str) -> tuple[tuple[str, str], ...]:
    try:
        return COORDINATE_ORDER[family]
    except KeyError as exc:
        raise KeyError(f"no declared coordinate order for family {family!r}") from exc


def coordinate_token(family: str, coordinates: Mapping[str, float]) -> str:
    """Render coordinates into the ``d55:w10`` half of a strategy handle.

    Deltas are written as integer percent, which is the same resolution as
    ``EnvConfig.coordinates.delta_step`` at its default of 0.05.
    """
    return ":".join(
        f"{prefix}{int(round(coordinates[name] * 100))}"
        for name, prefix in coordinate_order(family)
    )


def canonical_underlying(value: str) -> str:
    """Normalize ordinary lowercase symbols without collapsing mixed-case IDs."""

    symbol = value.strip()
    if symbol and symbol == symbol.lower() and any(character.isalpha() for character in symbol):
        symbol = symbol.upper()
    if not symbol or len(symbol) > 64:
        raise ValueError(f"invalid underlying in strategy handle: {value!r}")
    return symbol


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    """One bounded strategy topology with fixed relative coordinates."""

    strategy_type: str
    orientations: tuple[str, ...]
    strike_coordinates: Mapping[str, float]
    topology: str
    requires_same_expiry: bool
    near_far_expiry: bool = False
    emission_supported: bool = True

    @property
    def coordinate_token(self) -> str:
        """Derived, never stored -- ``<Q47>``, ruled 2026-09-22.

        This was a literal field sitting beside the coordinates it was supposed
        to describe, and on three of eleven families the two had drifted apart.
        Deriving it is what makes ``v2`` correct; keeping it derived is what
        stops a ``v3`` from needing the same ruling.  The frozen ``v1`` spellings
        live in ``V1_COORDINATE_TOKENS`` and are reached only through
        ``coordinate_token_for_version``.
        """
        return coordinate_token(self.strategy_type, self.strike_coordinates)


TEMPLATE_SPECS: dict[str, TemplateSpec] = {
    "outright": TemplateSpec(
        "outright",
        ("bullish", "bearish"),
        {"long_delta": 0.55},
        "buy_directional_option",
        True,
    ),
    "debit_vertical": TemplateSpec(
        "debit_vertical",
        ("bullish", "bearish"),
        {"long_delta": 0.55, "width_delta": 0.10},
        "buy_option_sell_farther_otm_same_right_same_expiry",
        True,
    ),
    "credit_vertical": TemplateSpec(
        "credit_vertical",
        ("bullish", "bearish"),
        {"short_delta": 0.35, "width_delta": 0.10},
        "sell_option_buy_farther_otm_same_right_same_expiry",
        True,
    ),
    "long_straddle": TemplateSpec(
        "long_straddle",
        ("neutral",),
        {"call_delta": 0.50, "put_delta": 0.50},
        "buy_atm_call_and_put_same_expiry",
        True,
        emission_supported=False,
    ),
    "long_strangle": TemplateSpec(
        "long_strangle",
        ("neutral",),
        {"put_delta": 0.25, "call_delta": 0.25},
        "buy_otm_put_and_call_same_expiry",
        True,
        emission_supported=False,
    ),
    "iron_butterfly": TemplateSpec(
        "iron_butterfly",
        ("neutral",),
        {"short_delta": 0.50, "wing_delta": 0.25},
        "short_atm_straddle_buy_otm_wings_same_expiry",
        True,
        emission_supported=False,
    ),
    "iron_condor": TemplateSpec(
        "iron_condor",
        ("neutral",),
        {"short_delta": 0.30, "width_delta": 0.10},
        "short_otm_strangle_buy_farther_otm_wings_same_expiry",
        True,
        emission_supported=False,
    ),
    "defined_risk_reversal": TemplateSpec(
        "defined_risk_reversal",
        ("bullish", "bearish"),
        {"directional_delta": 0.35, "tail_wing_delta": 0.20},
        "directional_option_financed_by_opposite_option_with_tail_wing",
        True,
    ),
    "butterfly": TemplateSpec(
        "butterfly",
        ("bullish", "bearish"),
        {"center_delta": 0.45, "lower_width_delta": 0.10, "upper_width_delta": 0.10},
        "one_two_one_same_right_same_expiry",
        True,
    ),
    "calendar": TemplateSpec(
        "calendar",
        ("bullish", "bearish"),
        {"strike_delta": 0.50},
        "sell_near_buy_far_same_right_similar_strike",
        False,
        near_far_expiry=True,
        emission_supported=False,
    ),
    "diagonal": TemplateSpec(
        "diagonal",
        ("bullish", "bearish"),
        {"near_delta": 0.40, "far_delta": 0.55},
        "sell_near_buy_far_same_right_different_strikes",
        False,
        near_far_expiry=True,
        emission_supported=False,
    ),
}


def _validate_tenor(spec: TemplateSpec, tenor_bucket: str) -> None:
    buckets = tenor_bucket.split("-")
    if spec.near_far_expiry:
        if len(buckets) != 2 or any(bucket not in TENOR_BUCKETS for bucket in buckets):
            raise ValueError(f"{spec.strategy_type} requires two valid tenor buckets")
        order = ["0_7", "8_30", "31_90", "91_180", "181_plus"]
        if order.index(buckets[0]) >= order.index(buckets[1]):
            raise ValueError("near tenor bucket must precede far tenor bucket")
    elif len(buckets) != 1 or tenor_bucket not in TENOR_BUCKETS:
        raise ValueError(f"invalid tenor bucket: {tenor_bucket!r}")


def coordinate_token_for_version(strategy_type: str, version: str) -> str:
    """The default coordinate token a family takes under ``version``.

    Differs from ``TemplateSpec.coordinate_token`` only on the three families
    ``v1`` froze wrong.  Every other family, and every family under ``v2``,
    renders straight from its coordinates.
    """
    if version in FROZEN_HANDLE_VERSIONS and strategy_type in V1_COORDINATE_TOKENS:
        return V1_COORDINATE_TOKENS[strategy_type]
    return TEMPLATE_SPECS[strategy_type].coordinate_token


def strategy_handle(
    *,
    underlying: str,
    strategy_type: str,
    orientation: str,
    tenor_bucket: str,
    version: str = HANDLE_VERSION,
    allow_frozen: bool = False,
) -> str:
    """Return a canonical stable handle for a registered template.

    ``allow_frozen`` exists so that ``parse_strategy_handle`` can rebuild the
    canonical form of an *old* handle in order to check it, without that being
    a way to mint new ones.  Callers that write handles must not set it: a
    frozen version is readable, not writable.
    """

    symbol = canonical_underlying(underlying)
    symbol_token = quote(symbol, safe=_UNDERLYING_SAFE)
    try:
        spec = TEMPLATE_SPECS[strategy_type]
    except KeyError as exc:
        raise ValueError(f"unknown strategy type: {strategy_type!r}") from exc
    if orientation not in spec.orientations:
        raise ValueError(
            f"orientation {orientation!r} is invalid for {strategy_type!r}"
        )
    _validate_tenor(spec, tenor_bucket)
    if version not in SUPPORTED_HANDLE_VERSIONS:
        raise ValueError(f"unsupported strategy handle version: {version!r}")
    if version in FROZEN_HANDLE_VERSIONS and not allow_frozen:
        raise ValueError(
            f"strategy handle version {version!r} is frozen and cannot be "
            f"emitted; new handles are {HANDLE_VERSION!r}"
        )
    return ":".join(
        (
            symbol_token,
            strategy_type,
            orientation,
            tenor_bucket,
            coordinate_token_for_version(strategy_type, version),
            version,
        )
    )


def parse_strategy_handle(value: str) -> dict[str, str]:
    """Parse a handle and reject non-canonical or topology-inconsistent input."""

    if not isinstance(value, str) or not value:
        raise ValueError("strategy_handle must be a non-empty string")
    parts = value.split(":")
    if len(parts) < 6:
        raise ValueError("malformed strategy handle")
    underlying_token, strategy_type, orientation, tenor_bucket = parts[:4]
    underlying = unquote(underlying_token)
    if quote(underlying, safe=_UNDERLYING_SAFE) != underlying_token:
        raise ValueError("underlying token is not canonically encoded")
    coordinate_token = ":".join(parts[4:-1])
    version = parts[-1]
    # ``allow_frozen``: reading a ``v1`` handle is the point of freezing it.
    canonical = strategy_handle(
        underlying=underlying,
        strategy_type=strategy_type,
        orientation=orientation,
        tenor_bucket=tenor_bucket,
        version=version,
        allow_frozen=True,
    )
    if coordinate_token != coordinate_token_for_version(strategy_type, version):
        raise ValueError("strategy handle coordinates do not match topology")
    if value != canonical:
        raise ValueError("strategy handle is not canonical")
    return {
        "underlying": underlying,
        "strategy_type": strategy_type,
        "orientation": orientation,
        "tenor_bucket": tenor_bucket,
        "coordinate_token": coordinate_token,
        "version": version,
    }


def template_legs(strategy_type: str, orientation: str) -> list[dict[str, Any]]:
    """Expand a registered topology into unresolved relative-coordinate legs."""

    if strategy_type not in TEMPLATE_SPECS:
        raise ValueError(f"unknown strategy type: {strategy_type!r}")
    if orientation not in TEMPLATE_SPECS[strategy_type].orientations:
        raise ValueError("orientation is incompatible with strategy topology")
    right = "call" if orientation == "bullish" else "put"
    opposite = "put" if orientation == "bullish" else "call"

    if strategy_type == "outright":
        legs = [("buy", 1, right, 0.55, "single")]
    elif strategy_type == "debit_vertical":
        legs = [("buy", 1, right, 0.55, "single"), ("sell", 1, right, 0.45, "single")]
    elif strategy_type == "credit_vertical":
        right = "put" if orientation == "bullish" else "call"
        legs = [("sell", 1, right, 0.35, "single"), ("buy", 1, right, 0.25, "single")]
    elif strategy_type == "long_straddle":
        legs = [("buy", 1, "call", 0.50, "single"), ("buy", 1, "put", 0.50, "single")]
    elif strategy_type == "long_strangle":
        legs = [("buy", 1, "call", 0.25, "single"), ("buy", 1, "put", 0.25, "single")]
    elif strategy_type == "iron_butterfly":
        legs = [
            ("sell", 1, "call", 0.50, "single"),
            ("sell", 1, "put", 0.50, "single"),
            ("buy", 1, "call", 0.25, "single"),
            ("buy", 1, "put", 0.25, "single"),
        ]
    elif strategy_type == "iron_condor":
        legs = [
            ("sell", 1, "call", 0.30, "single"),
            ("buy", 1, "call", 0.20, "single"),
            ("sell", 1, "put", 0.30, "single"),
            ("buy", 1, "put", 0.20, "single"),
        ]
    elif strategy_type == "defined_risk_reversal":
        legs = [
            ("buy", 1, right, 0.35, "single"),
            ("sell", 1, opposite, 0.35, "single"),
            ("buy", 1, opposite, 0.20, "single"),
        ]
    elif strategy_type == "butterfly":
        legs = [
            ("buy", 1, right, 0.55, "single"),
            ("sell", 2, right, 0.45, "single"),
            ("buy", 1, right, 0.35, "single"),
        ]
    elif strategy_type == "calendar":
        legs = [("sell", 1, right, 0.50, "near"), ("buy", 1, right, 0.50, "far")]
    elif strategy_type == "diagonal":
        legs = [("sell", 1, right, 0.40, "near"), ("buy", 1, right, 0.55, "far")]
    else:  # pragma: no cover - registry and expansion must evolve together.
        raise ValueError(f"no leg expansion for {strategy_type!r}")

    return [
        {
            "side": side,
            "ratio": ratio,
            "right": option_right,
            "expiry_role": expiry_role,
            "expiry": None,
            "contract_id": None,
            "strike": None,
            "target_delta": target_delta,
        }
        for side, ratio, option_right, target_delta, expiry_role in legs
    ]


def resolver_spec(strategy_type: str) -> dict[str, Any]:
    spec = TEMPLATE_SPECS[strategy_type]
    return {
        "template_version": HANDLE_VERSION,
        "topology": spec.topology,
        "requires_same_expiry": spec.requires_same_expiry,
        "requires_ordered_near_far_expiry": spec.near_far_expiry,
    }
