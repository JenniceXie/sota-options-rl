"""Candidate-set generation from point-in-time daily feature rows.

Candidates are option strategies, not contracts. A strategy may contain one or
more legs. When input rows carry ``strategy_id`` that identifier is used to
group legs before filters run. Rows without ``strategy_id`` are treated as
single-leg strategies for backward compatibility with early daily features.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from portfolio_monkey.data.features.strategy_templates import (
    TEMPLATE_SPECS,
    canonical_underlying,
    resolver_spec,
    strategy_handle,
)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class CandidateFilterConfig:
    """Configurable liquidity, cost, expiry, and moneyness filters."""

    min_volume: float | None = None
    min_open_interest: float | None = None
    max_relative_spread: float | None = None
    min_days_to_expiry: float | None = None
    max_days_to_expiry: float | None = None
    min_moneyness: float | None = None
    max_moneyness: float | None = None


def _passes_min(value: object, threshold: float | None) -> bool:
    if threshold is None:
        return True
    number = _number(value)
    return number is not None and number >= threshold


def _passes_max(value: object, threshold: float | None) -> bool:
    if threshold is None:
        return True
    number = _number(value)
    return number is not None and number <= threshold


def _strategy_id(row: Mapping[str, Any]) -> str | None:
    strategy_id = row.get("strategy_id")
    if strategy_id:
        return str(strategy_id)
    contract_id = row.get("contract_id")
    if contract_id:
        return f"single:{contract_id}"
    return None


def _string_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[str]:
    values = {str(row[name]) for row in rows if row.get(name) is not None}
    return sorted(values)


def _numbers(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    return [number for row in rows if (number := _number(row.get(name))) is not None]


def _max_text(rows: Sequence[Mapping[str, Any]], name: str) -> str | None:
    values = [str(row[name]) for row in rows if row.get(name) is not None]
    return max(values) if values else None


def strategy_candidate_row(
    *,
    strategy_id: str,
    rows: Sequence[Mapping[str, Any]],
    target_date: date,
) -> dict[str, Any]:
    """Build one strategy-level candidate row from one or more leg rows."""

    volumes = _numbers(rows, "volume")
    open_interests = _numbers(rows, "open_interest")
    relative_spreads = _numbers(rows, "relative_spread")
    days_to_expiry = _numbers(rows, "days_to_expiry")
    moneyness = _numbers(rows, "moneyness")
    strike_over_spot = _numbers(rows, "strike_over_spot")
    buy_volumes = _numbers(rows, "buy_contract_volume")
    sell_volumes = _numbers(rows, "sell_contract_volume")
    classified_volume = sum(buy_volumes) + sum(sell_volumes)

    event_time = _max_text(rows, "event_time")
    available_time = _max_text(rows, "available_time")
    ingested_time = _max_text(rows, "ingested_time")
    legs = [
        {
            "contract_id": row.get("contract_id"),
            "underlying": row.get("underlying"),
            "expiry": row.get("expiry"),
            "strike": row.get("strike"),
            "option_type": row.get("option_type"),
            "side": row.get("side"),
            "leg_role": row.get("leg_role"),
            "leg_ratio": row.get("leg_ratio", 1),
            "instrument_type": row.get("instrument_type"),
            "is_flex": row.get("is_flex"),
            "days_to_expiry": row.get("days_to_expiry"),
            "moneyness": row.get("moneyness"),
            "strike_over_spot": row.get("strike_over_spot"),
            "moneyness_convention": row.get("moneyness_convention"),
            "event_time": row.get("event_time"),
            "available_time": row.get("available_time"),
            "ingested_time": row.get("ingested_time"),
        }
        for row in rows
    ]

    ratios = [abs(_number(row.get("leg_ratio")) or 1.0) for row in rows]
    volume_capacities = [
        volume / ratio
        for volume, ratio in zip(
            (_number(row.get("volume")) for row in rows), ratios, strict=True
        )
        if volume is not None and ratio > 0
    ]
    oi_capacities = [
        oi / ratio
        for oi, ratio in zip(
            (_number(row.get("open_interest")) for row in rows), ratios, strict=True
        )
        if oi is not None and ratio > 0
    ]

    return {
        "strategy_id": strategy_id,
        "candidate_date": target_date.isoformat(),
        "strategy_type": next(
            (str(row["strategy_type"]) for row in rows if row.get("strategy_type")),
            "single_leg" if len(rows) == 1 else "multi_leg",
        ),
        "leg_count": len(rows),
        "contract_ids": _string_values(rows, "contract_id"),
        "underlyings": _string_values(rows, "underlying"),
        "event_time": event_time,
        "available_time": available_time,
        "ingested_time": ingested_time,
        # Capacity is constrained by the least liquid ratio-adjusted leg.
        "strategy_volume": (
            min(volume_capacities)
            if len(volume_capacities) == len(rows) and volume_capacities
            else None
        ),
        "strategy_open_interest": (
            min(oi_capacities)
            if len(oi_capacities) == len(rows) and oi_capacities
            else None
        ),
        # Without package bid/ask economics, the conservative leg spread is
        # auditable; summing relative spreads has no economic interpretation.
        "strategy_relative_spread": (
            max(relative_spreads)
            if len(relative_spreads) == len(rows) and relative_spreads
            else None
        ),
        "gross_leg_volume": sum(volumes) if volumes else None,
        "gross_leg_open_interest": sum(open_interests) if open_interests else None,
        "buy_contract_volume": sum(buy_volumes) if buy_volumes else None,
        "sell_contract_volume": sum(sell_volumes) if sell_volumes else None,
        "seller_share": (
            sum(sell_volumes) / classified_volume if classified_volume else None
        ),
        "instrument_types": _string_values(rows, "instrument_type"),
        "is_flex": any(
            row.get("is_flex") is True
            or "flex" in str(row.get("instrument_type") or "").lower()
            for row in rows
        ),
        "min_days_to_expiry": min(days_to_expiry) if days_to_expiry else None,
        "max_days_to_expiry": max(days_to_expiry) if days_to_expiry else None,
        "min_moneyness": min(moneyness) if moneyness else None,
        "max_moneyness": max(moneyness) if moneyness else None,
        "min_strike_over_spot": min(strike_over_spot)
        if strike_over_spot
        else None,
        "max_strike_over_spot": max(strike_over_spot)
        if strike_over_spot
        else None,
        "moneyness_convention": (
            "strike_over_spot" if strike_over_spot else "spot_over_strike"
        ),
        "legs": legs,
    }


def passes_candidate_filters(
    strategy: Mapping[str, Any],
    config: CandidateFilterConfig,
) -> bool:
    """Return whether a strategy belongs in the candidate set."""

    if not strategy.get("strategy_id") or not strategy.get("legs"):
        return False
    return (
        _passes_min(strategy.get("strategy_volume"), config.min_volume)
        and _passes_min(
            strategy.get("strategy_open_interest"), config.min_open_interest
        )
        and _passes_max(
            strategy.get("strategy_relative_spread"), config.max_relative_spread
        )
        and _passes_min(strategy.get("min_days_to_expiry"), config.min_days_to_expiry)
        and _passes_max(strategy.get("max_days_to_expiry"), config.max_days_to_expiry)
        and _passes_min(strategy.get("min_moneyness"), config.min_moneyness)
        and _passes_max(strategy.get("max_moneyness"), config.max_moneyness)
    )


def candidate_score(strategy: Mapping[str, Any]) -> float:
    """Simple deterministic score for initial candidate ordering."""

    volume = _number(strategy.get("strategy_volume")) or 0.0
    open_interest = _number(strategy.get("strategy_open_interest")) or 0.0
    relative_spread = _number(strategy.get("strategy_relative_spread"))
    spread_penalty = max(relative_spread if relative_spread is not None else 0.01, 1e-6)
    return (volume + 0.1 * open_interest) / spread_penalty


def generate_candidate_rows(
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    target_date: date,
    config: CandidateFilterConfig,
) -> list[dict[str, Any]]:
    """Build a date-level option candidate set without underlying partitioning."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        strategy_id = _strategy_id(row)
        if strategy_id is not None:
            grouped[strategy_id].append(row)

    strategies = [
        strategy_candidate_row(
            strategy_id=strategy_id,
            rows=rows,
            target_date=target_date,
        )
        for strategy_id, rows in grouped.items()
    ]
    candidates = [
        {
            **strategy,
            "candidate_score": candidate_score(strategy),
        }
        for strategy in strategies
        if passes_candidate_filters(strategy, config)
    ]
    candidates.sort(
        key=lambda row: (
            -(row["candidate_score"] or 0.0),
            ",".join(row.get("underlyings") or ()),
            row.get("strategy_id") or "",
            row.get("event_time") or "",
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["candidate_rank"] = rank
    return candidates


POINT_IN_TIME_RULE = (
    "available_time <= decision_time; information_cutoff == decision_time; "
    "max_input_available_time <= decision_time"
)
CANDIDATE_SCHEMA_VERSION = "option_strategy_candidate.v1"
RANKING_VERSION = "state_template_rank.v1"


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and math.isfinite(result) else None


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


def _normalized(flow: Mapping[str, Any], field: str) -> float | None:
    validity = flow.get("normalization_valid")
    values = flow.get("normalized_feature_values")
    if not isinstance(validity, Mapping) or not bool(validity.get(field)):
        return None
    if not isinstance(values, Mapping):
        return None
    value = _finite(values.get(field))
    return _clip(value / 3.0) if value is not None else None


def _weighted_available(
    values: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> float | None:
    available = [
        (value, weights[name])
        for name, value in values.items()
        if value is not None and weights.get(name, 0.0) > 0
    ]
    denominator = sum(weight for _, weight in available)
    if denominator <= 0:
        return None
    return sum(value * weight for value, weight in available) / denominator


def _state_signals(row: Mapping[str, Any]) -> dict[str, Any]:
    flow = row.get("option_flow")
    context = row.get("stock_context")
    market = row.get("underlying_market")
    flow = flow if isinstance(flow, Mapping) else {}
    context = context if isinstance(context, Mapping) else {}
    market = market if isinstance(market, Mapping) else {}

    predicted = _finite(flow.get("predicted_weighted_demand"))
    uncertainty = _finite(flow.get("predicted_weighted_demand_uncertainty"))
    forecast_signal = (
        _clip(predicted / uncertainty / 3.0)
        if predicted is not None and uncertainty is not None and uncertainty > 0
        else None
    )
    news_direction = _finite(context.get("news_direction_mean"))
    catalyst = _finite(context.get("news_catalyst_score"))
    news_signal = (
        _clip(news_direction) * _clip(catalyst, 0.0, 1.0)
        if news_direction is not None and catalyst is not None
        else None
    )
    spot_z = _finite(market.get("spot_return_step_z"))
    direction_sources = {
        "delta_flow": _normalized(flow, "signed_delta_demand"),
        "contract_flow": _normalized(flow, "signed_contract_demand"),
        "forecast": forecast_signal,
        "news": news_signal,
        "spot": _clip(spot_z / 3.0) if spot_z is not None else None,
    }
    directional = _weighted_available(
        direction_sources,
        {
            "delta_flow": 0.35,
            "contract_flow": 0.20,
            "forecast": 0.25,
            "news": 0.15,
            "spot": 0.05,
        },
    )

    persistence = _normalized(flow, "response_persistence_score_600s")
    if persistence is None:
        raw_persistence = _finite(flow.get("response_persistence_score_600s"))
        persistence = _clip(raw_persistence) if raw_persistence is not None else None
    reversal_weight = _finite(flow.get("reversal_response_weight_600s"))
    consistent_weight = _finite(
        flow.get("direction_consistent_response_weight_600s")
    )
    response_total = (reversal_weight or 0.0) + (consistent_weight or 0.0)
    reversal = (
        _clip(((reversal_weight or 0.0) - (consistent_weight or 0.0)) / response_total)
        if response_total > 0
        else None
    )
    surface_share = _finite(flow.get("top_surface_bucket_share"))
    return {
        "directional_signal": directional,
        "directional_sources": direction_sources,
        "persistence_signal": persistence,
        "reversal_signal": reversal,
        "skew_concentration": (
            _clip(surface_share, 0.0, 1.0) if surface_share is not None else None
        ),
        "top_surface_bucket": flow.get("top_surface_bucket"),
        "catalyst_signal": (
            _clip(catalyst, 0.0, 1.0) if catalyst is not None else None
        ),
        "hours_to_scheduled_event": _finite(
            context.get("hours_to_scheduled_event")
        ),
        # These are unavailable in stock_decision_state.v4 and remain explicit.
        "volatility_signal": None,
        "term_structure_signal": None,
    }


def _surface_tenor(value: Any) -> str | None:
    text = str(value or "")
    suffixes = {
        "000_007": "0_7",
        "008_030": "8_30",
        "031_090": "31_90",
        "091_180": "91_180",
        "181_plus": "181_plus",
    }
    return next((bucket for suffix, bucket in suffixes.items() if suffix in text), None)


def _candidate_score_components(
    signals: Mapping[str, Any],
    *,
    directional_alignment: float,
    reversal: bool,
    skew: bool,
) -> dict[str, float | None]:
    persistence = _finite(signals.get("persistence_signal"))
    reversal_signal = _finite(signals.get("reversal_signal"))
    return {
        "directional_signal": round(abs(directional_alignment), 6),
        "volatility_signal": None,
        "persistence_signal": (
            round(max(persistence or 0.0, 0.0), 6)
            if persistence is not None and not reversal
            else None
        ),
        "reversal_signal": (
            round(max(reversal_signal or 0.0, 0.0), 6) if reversal else None
        ),
        "skew_concentration": (
            round(float(signals["skew_concentration"]), 6)
            if skew and signals.get("skew_concentration") is not None
            else None
        ),
        "catalyst_signal": (
            round(float(signals["catalyst_signal"]), 6)
            if signals.get("catalyst_signal") is not None
            else None
        ),
    }


def _score_components(components: Mapping[str, float | None]) -> float:
    weights = {
        "directional_signal": 0.50,
        "volatility_signal": 0.20,
        "persistence_signal": 0.20,
        "reversal_signal": 0.20,
        "skew_concentration": 0.15,
        "catalyst_signal": 0.10,
    }
    value = _weighted_available(components, weights)
    return _clip(value or 0.0, 0.0, 1.0)


def _template_candidate(
    *,
    row: Mapping[str, Any],
    target_date: date,
    ingested_time: str,
    strategy_type: str,
    orientation: str,
    thesis_horizon: str,
    tenor_bucket: str,
    signals: Mapping[str, Any],
    directional_alignment: float,
    reversal: bool = False,
    skew: bool = False,
) -> dict[str, Any]:
    underlying = canonical_underlying(str(row["underlying"]))
    decision_time = str(row["decision_time"])
    spec = TEMPLATE_SPECS[strategy_type]
    handle = strategy_handle(
        underlying=underlying,
        strategy_type=strategy_type,
        orientation=orientation,
        tenor_bucket=tenor_bucket,
    )
    components = _candidate_score_components(
        signals,
        directional_alignment=directional_alignment,
        reversal=reversal,
        skew=skew,
    )
    missing_count = int(row.get("missing_block_count") or 0)
    quality_multiplier = max(0.60, 1.0 - 0.08 * missing_count)
    score = round(_score_components(components) * quality_multiplier, 6)
    availability = {name: value is not None for name, value in components.items()}
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_date": target_date.isoformat(),
        "underlying": underlying,
        "underlyings": [underlying],
        "step": row["step"],
        "decision_time": decision_time,
        "information_cutoff": decision_time,
        "event_time": decision_time,
        # The joined decision state is the final dependency.  The template can
        # first be computed at the decision boundary, not at an earlier source.
        "available_time": decision_time,
        "ingested_time": ingested_time,
        "max_input_available_time": row.get("max_known_available_time"),
        "strategy_handle": handle,
        "strategy_type": strategy_type,
        "orientation": orientation,
        "thesis_horizon": thesis_horizon,
        "tenor_bucket": tenor_bucket,
        "strike_coordinates": dict(spec.strike_coordinates),
        "candidate_rank": None,
        "candidate_score": score,
        "score_components": components,
        "score_component_availability": availability,
        "state_quality": {
            "missing_block_count": missing_count,
            "resolution_status": "template_available",
            "directional_source_availability": {
                name: value is not None
                for name, value in signals["directional_sources"].items()
            },
        },
        "resolver_spec": resolver_spec(strategy_type),
    }


def _validate_candidate_source(row: Mapping[str, Any], target_date: date) -> None:
    decision = _timestamp(row.get("decision_time"), field="decision_time")
    if decision.date() != target_date:
        raise ValueError("state decision_time does not match candidate date")
    cutoff = _timestamp(row.get("information_cutoff"), field="information_cutoff")
    if cutoff != decision:
        raise ValueError("state information_cutoff must equal decision_time")
    available = _timestamp(row.get("available_time"), field="available_time")
    if available > decision:
        raise ValueError("state available_time exceeds decision_time")
    max_known = row.get("max_known_available_time")
    if max_known is not None and _timestamp(
        max_known, field="max_known_available_time"
    ) > decision:
        raise ValueError("state contains future input availability")
    if row.get("step") not in {"market_open", "market_close"}:
        raise ValueError("state step must be market_open or market_close")
    if not row.get("underlying"):
        raise ValueError("state requires underlying")


def generate_strategy_template_candidates(
    state_rows: Iterable[Mapping[str, Any]],
    *,
    target_date: date,
    max_candidates_per_underlying: int = 8,
    ingested_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Map point-in-time stock beliefs into a bounded deterministic slate."""

    if max_candidates_per_underlying <= 0:
        raise ValueError("max_candidates_per_underlying must be positive")
    created = (ingested_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ingested = created.isoformat()
    output: list[dict[str, Any]] = []
    family_priority = {
        "debit_vertical": 0,
        "outright": 1,
        "credit_vertical": 2,
        "defined_risk_reversal": 3,
        "butterfly": 4,
    }

    for row in state_rows:
        _validate_candidate_source(row, target_date)
        signals = _state_signals(row)
        directional = _finite(signals.get("directional_signal"))
        primary_directional_evidence = any(
            signals["directional_sources"].get(name) is not None
            for name in ("delta_flow", "contract_flow", "forecast", "news")
        )
        if (
            not primary_directional_evidence
            or directional is None
            or abs(directional) < 0.12
        ):
            continue
        orientation = "bullish" if directional > 0 else "bearish"
        candidates: list[dict[str, Any]] = []
        for strategy_type in ("debit_vertical", "outright", "credit_vertical"):
            candidates.append(
                _template_candidate(
                    row=row,
                    target_date=target_date,
                    ingested_time=ingested,
                    strategy_type=strategy_type,
                    orientation=orientation,
                    thesis_horizon="1_5d",
                    tenor_bucket="31_90",
                    signals=signals,
                    directional_alignment=directional,
                )
            )

        reversal_signal = _finite(signals.get("reversal_signal"))
        if reversal_signal is not None and reversal_signal >= 0.15:
            reversal_orientation = "bearish" if orientation == "bullish" else "bullish"
            for strategy_type in ("debit_vertical", "credit_vertical"):
                candidates.append(
                    _template_candidate(
                        row=row,
                        target_date=target_date,
                        ingested_time=ingested,
                        strategy_type=strategy_type,
                        orientation=reversal_orientation,
                        thesis_horizon="0_1d",
                        tenor_bucket="8_30",
                        signals=signals,
                        directional_alignment=-directional,
                        reversal=True,
                    )
                )

        surface_share = _finite(signals.get("skew_concentration"))
        if surface_share is not None and surface_share >= 0.35:
            surface_tenor = _surface_tenor(signals.get("top_surface_bucket"))
            tenor = surface_tenor or "31_90"
            for strategy_type in ("defined_risk_reversal", "butterfly"):
                candidates.append(
                    _template_candidate(
                        row=row,
                        target_date=target_date,
                        ingested_time=ingested,
                        strategy_type=strategy_type,
                        orientation=orientation,
                        thesis_horizon="1_5d",
                        tenor_bucket=tenor,
                        signals=signals,
                        directional_alignment=directional,
                        skew=True,
                    )
                )

        # A stable handle can be proposed by more than one thesis channel.  Keep
        # the strongest auditable proposal, then rank within this decision only.
        by_handle: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            existing = by_handle.get(candidate["strategy_handle"])
            if existing is None or candidate["candidate_score"] > existing["candidate_score"]:
                by_handle[candidate["strategy_handle"]] = candidate
        ranked = sorted(
            by_handle.values(),
            key=lambda candidate: (
                -float(candidate["candidate_score"]),
                family_priority.get(str(candidate["strategy_type"]), 99),
                str(candidate["strategy_handle"]),
            ),
        )[:max_candidates_per_underlying]
        for rank, candidate in enumerate(ranked, start=1):
            candidate["candidate_rank"] = rank
            output.append(candidate)

    return sorted(
        output,
        key=lambda candidate: (
            candidate["decision_time"],
            candidate["underlying"],
            candidate["candidate_rank"],
            candidate["strategy_handle"],
        ),
    )


def evaluate_strategy_package(
    legs: Sequence[Mapping[str, Any]],
    *,
    decision_time: str,
    requires_same_expiry: bool = True,
    max_quote_age_seconds: float = 300.0,
    max_relative_spread: float | None = None,
    minimum_package_premium: float = 0.01,
) -> dict[str, Any]:
    """Evaluate exact legs jointly using executable package economics."""

    decision = _timestamp(decision_time, field="decision_time")
    missing: set[str] = set()
    failed: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for leg in legs:
        side = str(leg.get("side") or "")
        ratio = _finite(leg.get("ratio"))
        if side not in {"buy", "sell"} or ratio is None or ratio <= 0:
            failed.add("invalid_leg_side_or_ratio")
            continue
        if not leg.get("contract_id") or leg.get("strike") is None or not leg.get("expiry"):
            missing.add("resolved_contracts")
        bid = _finite(leg.get("bid"))
        ask = _finite(leg.get("ask"))
        if bid is None or ask is None:
            missing.add("point_in_time_chain_quotes")
        elif bid < 0 or ask < bid:
            failed.add("crossed_or_invalid_quote")
        quote_time_raw = leg.get("quote_time") or leg.get("available_time")
        if quote_time_raw is None:
            missing.add("quote_time")
        else:
            quote_time = _timestamp(quote_time_raw, field="quote_time")
            if quote_time > decision:
                failed.add("future_quote")
            elif (decision - quote_time).total_seconds() > max_quote_age_seconds:
                failed.add("stale_quote")
        normalized.append(
            {
                **leg,
                "side": side,
                "ratio": ratio,
                "bid": bid,
                "ask": ask,
            }
        )

    expiries = {str(leg.get("expiry")) for leg in normalized if leg.get("expiry")}
    if requires_same_expiry and len(expiries) > 1:
        failed.add("expiry_relationship")
    if missing or failed or len(normalized) != len(legs) or not legs:
        return {
            "passed": False if failed else None,
            "missing_inputs": sorted(missing),
            "failed_checks": sorted(failed),
            "package_market": {
                key: None
                for key in (
                    "bid",
                    "ask",
                    "mid",
                    "relative_spread",
                    "capacity_volume",
                    "capacity_open_interest",
                    "delta",
                    "gamma",
                    "vega",
                    "theta",
                    "max_loss",
                    "margin",
                    "assignment_exposure",
                )
            },
        }

    package_ask = sum(
        leg["ratio"] * (leg["ask"] if leg["side"] == "buy" else -leg["bid"])
        for leg in normalized
    )
    package_bid = sum(
        leg["ratio"] * (leg["bid"] if leg["side"] == "buy" else -leg["ask"])
        for leg in normalized
    )
    package_mid = (package_bid + package_ask) / 2.0
    width = package_ask - package_bid
    denominator = max(abs(package_mid), minimum_package_premium)
    relative_spread = width / denominator
    if max_relative_spread is not None and relative_spread > max_relative_spread:
        failed.add("package_relative_spread")

    def capacity(field: str) -> float | None:
        values = [_finite(leg.get(field)) for leg in normalized]
        if any(value is None for value in values):
            return None
        return min(
            float(value) / leg["ratio"]
            for value, leg in zip(values, normalized, strict=True)
        )

    def signed_greek(field: str) -> float | None:
        values = [_finite(leg.get(field)) for leg in normalized]
        if any(value is None for value in values):
            return None
        return sum(
            (1.0 if leg["side"] == "buy" else -1.0)
            * leg["ratio"]
            * float(value)
            for value, leg in zip(values, normalized, strict=True)
        )

    max_loss: float | None = None
    strikes = [_finite(leg.get("strike")) for leg in normalized]
    if requires_same_expiry and all(strike is not None for strike in strikes):
        call_tail_slope = sum(
            (1.0 if leg["side"] == "buy" else -1.0) * leg["ratio"]
            for leg in normalized
            if str(leg.get("right")) == "call"
        )
        if call_tail_slope < 0:
            failed.add("unbounded_max_loss")
        else:
            points = [
                0.0,
                *(float(strike) for strike in strikes),
                max(float(strike) for strike in strikes) * 3.0 + 1.0,
            ]
            pnl: list[float] = []
            for spot in points:
                payoff = 0.0
                for leg in normalized:
                    strike = float(leg["strike"])
                    intrinsic = (
                        max(spot - strike, 0.0)
                        if leg.get("right") == "call"
                        else max(strike - spot, 0.0)
                    )
                    sign = 1.0 if leg["side"] == "buy" else -1.0
                    payoff += sign * leg["ratio"] * intrinsic
                pnl.append(payoff - package_ask)
            max_loss = max(0.0, -min(pnl))

    assignment = sum(
        leg["ratio"] for leg in normalized if leg["side"] == "sell"
    )
    market = {
        "bid": round(package_bid, 10),
        "ask": round(package_ask, 10),
        "mid": round(package_mid, 10),
        "relative_spread": round(relative_spread, 10),
        "relative_spread_denominator": "max(abs(package_mid), minimum_package_premium)",
        "capacity_volume": capacity("volume"),
        "capacity_open_interest": capacity("open_interest"),
        "delta": signed_greek("delta"),
        "gamma": signed_greek("gamma"),
        "vega": signed_greek("vega"),
        "theta": signed_greek("theta"),
        "max_loss": max_loss,
        "margin": max_loss,
        "assignment_exposure": assignment,
    }
    return {
        "passed": not failed,
        "missing_inputs": [],
        "failed_checks": sorted(failed),
        "package_market": market,
    }
