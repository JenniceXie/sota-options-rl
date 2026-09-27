"""Point-in-time option-flow classification and compact RL state features."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import log, log1p
from statistics import mean, median
from typing import Any

from portfolio_monkey.data.schemas import CanonicalRecord, RecordType
from portfolio_monkey.env.clock import DecisionPoint


class TradeDirection(str, Enum):
    """Direction inferred from the latest causally available quote."""

    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClassifiedOptionTrade:
    """An option trade plus auditable quote-rule classification details."""

    trade: CanonicalRecord
    direction: TradeDirection
    method: str
    quote: CanonicalRecord | None
    quote_mid: float | None
    quote_age_seconds: float | None
    price_location: float | None
    size_to_displayed_depth: float | None


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_number(value: object) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _condition_text(payload: Mapping[str, Any]) -> str:
    labels = payload.get("condition_labels")
    if isinstance(labels, (list, tuple)):
        return " ".join(str(label) for label in labels).lower()
    return str(labels or "").lower()


def is_opra_auction(payload: Mapping[str, Any]) -> bool:
    """Return an explicit or descriptively mapped OPRA auction indicator.

    Numeric OPRA condition codes are deliberately not hard-coded here. They
    should be mapped point-in-time by the upstream condition-reference feed.
    Auction prints are retained as a separate control and are not treated as
    evidence that a trade is uninformed.
    """

    if _truthy(payload.get("is_pim")):
        return True
    text = _condition_text(payload)
    markers = (
        "price improvement",
        "pim",
        "automated improvement",
        "aim auction",
        "complex order auction",
        "coa auction",
        "single leg auction",
        "multi leg auction",
        "stock options auction",
    )
    return any(marker in text for marker in markers)


def is_price_improvement_auction(payload: Mapping[str, Any]) -> bool:
    """Backward-compatible alias for the broader OPRA auction diagnostic."""

    return is_opra_auction(payload)


def is_mechanical_trade(payload: Mapping[str, Any]) -> bool:
    """Use only explicit upstream mechanical-flow evidence."""

    if _truthy(payload.get("is_mechanical")):
        return True
    probability = _number(payload.get("mechanical_probability"))
    return probability is not None and probability >= 0.5


class QuoteRuleClassifier:
    """Classify option trades using quotes available when each trade arrived.

    A trade above the quote midpoint is buyer initiated and one below is seller
    initiated. Midpoint trades use the tick test, including the last non-zero
    tick for equal-price prints. Missing, invalid, future, or stale quotes leave
    direction unknown.
    """

    def __init__(
        self,
        quotes: Sequence[CanonicalRecord],
        *,
        max_quote_age: timedelta = timedelta(seconds=60),
        price_tolerance: float = 1e-12,
    ) -> None:
        self.max_quote_age = max_quote_age
        self.price_tolerance = price_tolerance
        grouped: dict[str, list[CanonicalRecord]] = defaultdict(list)
        for quote in quotes:
            if quote.record_type == RecordType.OPTION_QUOTE:
                grouped[quote.instrument.alignment_key].append(quote)
        self._quotes = {
            key: sorted(
                values,
                key=lambda quote: (
                    quote.event_time,
                    quote.available_time,
                    quote.ingested_time,
                ),
            )
            for key, values in grouped.items()
        }
        self._event_times = {
            key: [quote.event_time for quote in values]
            for key, values in self._quotes.items()
        }

    def quote_as_of(
        self,
        contract_key: str,
        event_time: datetime,
        available_time: datetime,
    ) -> tuple[CanonicalRecord | None, str]:
        """Return the latest valid quote observable by ``available_time``."""

        quotes = self._quotes.get(contract_key, ())
        if not quotes:
            return None, "no_quote"
        index = bisect_right(self._event_times[contract_key], event_time)
        candidate: CanonicalRecord | None = None
        for quote_index in range(index - 1, -1, -1):
            quote = quotes[quote_index]
            if quote.available_time <= available_time:
                candidate = quote
                break
        if candidate is None:
            return None, "no_causal_quote"

        bid = _number(candidate.payload.get("bid"))
        ask = _number(candidate.payload.get("ask"))
        if bid is None or ask is None or bid < 0 or ask < bid:
            return None, "invalid_quote"
        if event_time - candidate.event_time > self.max_quote_age:
            return None, "stale_quote"
        return candidate, "quote"

    def _quote_as_of_trade(
        self,
        trade: CanonicalRecord,
    ) -> tuple[CanonicalRecord | None, str]:
        return self.quote_as_of(
            trade.instrument.alignment_key,
            trade.event_time,
            trade.available_time,
        )

    def classify_many(
        self,
        trades: Sequence[CanonicalRecord],
    ) -> list[ClassifiedOptionTrade]:
        """Classify trades in event order so midpoint prints can use tick tests."""

        grouped: dict[str, list[CanonicalRecord]] = defaultdict(list)
        for trade in trades:
            if trade.record_type == RecordType.OPTION_TRADE:
                grouped[trade.instrument.alignment_key].append(trade)

        classified: list[ClassifiedOptionTrade] = []
        for key in sorted(grouped):
            previous_price: float | None = None
            last_tick = TradeDirection.UNKNOWN
            ordered = sorted(
                grouped[key],
                key=lambda trade: (
                    trade.event_time,
                    trade.available_time,
                    trade.ingested_time,
                ),
            )
            for trade in ordered:
                price = _number(trade.payload.get("price"))
                size = _positive_number(trade.payload.get("size"))
                quote, method = self._quote_as_of_trade(trade)
                direction = TradeDirection.UNKNOWN
                midpoint: float | None = None
                price_location: float | None = None
                size_to_depth: float | None = None

                if price is None:
                    method = "missing_price"
                elif quote is not None:
                    bid = float(quote.payload["bid"])
                    ask = float(quote.payload["ask"])
                    midpoint = (bid + ask) / 2
                    spread = ask - bid
                    if spread > 0:
                        price_location = (price - midpoint) / spread
                    if price > midpoint + self.price_tolerance:
                        direction = TradeDirection.BUY
                        method = "quote"
                    elif price < midpoint - self.price_tolerance:
                        direction = TradeDirection.SELL
                        method = "quote"
                    elif previous_price is not None:
                        if price > previous_price + self.price_tolerance:
                            direction = TradeDirection.BUY
                            method = "tick"
                            last_tick = direction
                        elif price < previous_price - self.price_tolerance:
                            direction = TradeDirection.SELL
                            method = "tick"
                            last_tick = direction
                        elif last_tick != TradeDirection.UNKNOWN:
                            direction = last_tick
                            method = "zero_tick"
                        else:
                            method = "at_mid"
                    else:
                        method = "at_mid"

                    depth_field = (
                        "ask_size"
                        if direction == TradeDirection.BUY
                        else "bid_size"
                        if direction == TradeDirection.SELL
                        else None
                    )
                    displayed_depth = (
                        _positive_number(quote.payload.get(depth_field))
                        if depth_field
                        else None
                    )
                    if size is not None and displayed_depth is not None:
                        size_to_depth = size / displayed_depth

                quote_age = (
                    (trade.event_time - quote.event_time).total_seconds()
                    if quote is not None
                    else None
                )
                classified.append(
                    ClassifiedOptionTrade(
                        trade=trade,
                        direction=direction,
                        method=method,
                        quote=quote,
                        quote_mid=midpoint,
                        quote_age_seconds=quote_age,
                        price_location=price_location,
                        size_to_displayed_depth=size_to_depth,
                    )
                )

                if price is not None:
                    if previous_price is not None:
                        if price > previous_price + self.price_tolerance:
                            last_tick = TradeDirection.BUY
                        elif price < previous_price - self.price_tolerance:
                            last_tick = TradeDirection.SELL
                    previous_price = price
        return sorted(
            classified,
            key=lambda item: (
                item.trade.event_time,
                item.trade.instrument.alignment_key,
            ),
        )


def _quote_midpoint_and_spread(
    quote: CanonicalRecord | None,
) -> tuple[float | None, float | None]:
    if quote is None:
        return None, None
    bid = _number(quote.payload.get("bid"))
    ask = _number(quote.payload.get("ask"))
    if bid is None or ask is None or bid < 0 or ask < bid:
        return None, None
    return (bid + ask) / 2, ask - bid


def _delta_bucket(delta: float | None) -> str | None:
    if delta is None or abs(delta) > 1.0:
        return None
    absolute_delta = abs(delta)
    for upper, label in (
        (0.10, "00_10"),
        (0.25, "10_25"),
        (0.40, "25_40"),
        (0.60, "40_60"),
        (float("inf"), "60_100"),
    ):
        if absolute_delta <= upper:
            return label
    return None


def _tenor_bucket(days_to_expiry: int | None) -> str | None:
    if days_to_expiry is None or days_to_expiry < 0:
        return None
    for upper, label in (
        (7, "000_007"),
        (30, "008_030"),
        (90, "031_090"),
        (180, "091_180"),
        (10**9, "181_plus"),
    ):
        if days_to_expiry <= upper:
            return label
    return None


def _surface_bucket(
    trade: CanonicalRecord,
    delta: float | None,
) -> tuple[str | None, int | None]:
    option_type = trade.instrument.option_type
    expiry = trade.instrument.expiry
    delta_label = _delta_bucket(delta)
    days_to_expiry = (
        (expiry - trade.event_time.date()).days if expiry is not None else None
    )
    tenor_label = _tenor_bucket(days_to_expiry)
    if option_type is None or delta_label is None or tenor_label is None:
        return None, days_to_expiry
    return (
        f"{option_type.value}:{delta_label}:{tenor_label}",
        days_to_expiry,
    )


def _concentration(
    bucket_values: Mapping[str, float],
) -> tuple[float | None, float | None, float | None, list[dict[str, float]]]:
    positive = {
        bucket: abs(value)
        for bucket, value in bucket_values.items()
        if abs(value) > 0
    }
    total = sum(positive.values())
    if total <= 0:
        return None, None, None, []
    shares = {bucket: value / total for bucket, value in positive.items()}
    hhi = sum(share * share for share in shares.values())
    if len(shares) == 1:
        entropy = 0.0
    else:
        entropy = -sum(share * log(share) for share in shares.values()) / log(
            len(shares)
        )
    ordered = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
    top = [
        {"bucket": bucket, "share": share}
        for bucket, share in ordered[:3]
    ]
    return hhi, entropy, ordered[0][1], top


class CompactOptionFlowStateBuilder:
    """Aggregate point-in-time option flow into a stable decision-step schema."""

    SCHEMA_VERSION = "compact_option_flow_state.v2"

    def __init__(
        self,
        *,
        large_trade_size: float = 100.0,
        max_quote_age_seconds: float = 60.0,
        response_horizons_seconds: Sequence[int] = (1, 5, 30, 300),
    ) -> None:
        if large_trade_size <= 0:
            raise ValueError("large_trade_size must be positive")
        if max_quote_age_seconds <= 0:
            raise ValueError("max_quote_age_seconds must be positive")
        horizons = tuple(sorted(set(response_horizons_seconds)))
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("response_horizons_seconds must contain positive values")
        self.large_trade_size = float(large_trade_size)
        self.max_quote_age = timedelta(seconds=max_quote_age_seconds)
        self.response_horizons_seconds = horizons

    def build(
        self,
        decision: DecisionPoint,
        records: Sequence[CanonicalRecord],
        *,
        session_open: datetime,
        underlying: str | None = None,
    ) -> dict[str, Any]:
        """Build one state using only ``available_time <= decision.timestamp``."""

        known = [
            record
            for record in records
            if record.available_time <= decision.timestamp
            and record.event_time <= decision.timestamp
        ]
        quotes = [
            record for record in known if record.record_type == RecordType.OPTION_QUOTE
        ]
        trades = [
            record
            for record in known
            if record.record_type == RecordType.OPTION_TRADE
            and session_open <= record.event_time <= decision.timestamp
        ]
        classifier = QuoteRuleClassifier(
            quotes,
            max_quote_age=self.max_quote_age,
        )
        classified = classifier.classify_many(trades)

        total_volume = 0.0
        classified_volume = 0.0
        buy_volume = 0.0
        sell_volume = 0.0
        unknown_volume = 0.0
        total_notional = 0.0
        signed_notional = 0.0
        large_volume = 0.0
        auction_volume = 0.0
        mechanical_volume = 0.0
        signed_contract_demand = 0.0
        signed_delta_demand = 0.0
        signed_gamma_demand = 0.0
        signed_vega_demand = 0.0
        greek_covered_volume = 0.0
        trade_sizes: list[float] = []
        quote_ages: list[float] = []
        price_locations: list[float] = []
        size_to_depth_values: list[float] = []
        methods: dict[str, int] = defaultdict(int)
        underlyings: set[str] = set()
        contracts: set[str] = set()
        package_keys: set[str] = set()
        weighting_numerators: dict[str, float] = defaultdict(float)
        weighting_denominators: dict[str, float] = defaultdict(float)
        weighting_covered_volume: dict[str, float] = defaultdict(float)
        response_matured_volume: dict[int, float] = defaultdict(float)
        response_covered_volume: dict[int, float] = defaultdict(float)
        response_trade_count: dict[int, int] = defaultdict(int)
        quote_impact_numerator: dict[int, float] = defaultdict(float)
        realized_spread_numerator: dict[int, float] = defaultdict(float)
        surface_signed_flow: dict[str, float] = defaultdict(float)
        surface_covered_volume = 0.0
        persistent_surface_impact: dict[str, float] = defaultdict(float)
        persistent_impact_total = 0.0
        persistent_delta_numerator = 0.0
        persistent_tenor_numerator = 0.0
        downside_put_impact = 0.0
        upside_call_impact = 0.0
        atm_vega_impact = 0.0
        short_dated_impact = 0.0
        persistent_horizon = (
            30
            if 30 in self.response_horizons_seconds
            else self.response_horizons_seconds[-1]
        )

        for index, item in enumerate(classified):
            trade = item.trade
            payload = trade.payload
            size = _positive_number(payload.get("size")) or 0.0
            price = _positive_number(payload.get("price")) or 0.0
            multiplier = _positive_number(payload.get("contract_multiplier")) or 100.0
            sign = (
                1.0
                if item.direction == TradeDirection.BUY
                else -1.0
                if item.direction == TradeDirection.SELL
                else 0.0
            )
            auction = is_opra_auction(payload)
            mechanical = is_mechanical_trade(payload)
            large = size >= self.large_trade_size or (
                item.size_to_displayed_depth is not None
                and item.size_to_displayed_depth >= 1.0
            )

            total_volume += size
            total_notional += price * size * multiplier
            trade_sizes.append(size)
            methods[item.method] += 1
            if item.quote_age_seconds is not None:
                quote_ages.append(item.quote_age_seconds)
            if item.price_location is not None:
                price_locations.append(item.price_location)
            if item.size_to_displayed_depth is not None:
                size_to_depth_values.append(item.size_to_displayed_depth)

            if sign > 0:
                buy_volume += size
                classified_volume += size
            elif sign < 0:
                sell_volume += size
                classified_volume += size
            else:
                unknown_volume += size

            signed = sign * size
            signed_contract_demand += signed
            signed_notional += sign * price * size * multiplier
            if large:
                large_volume += size
            if auction:
                auction_volume += size
            if mechanical:
                mechanical_volume += size

            delta = _number(payload.get("delta"))
            gamma = _number(payload.get("gamma"))
            vega = _number(payload.get("vega"))
            if sign and any(value is not None for value in (delta, gamma, vega)):
                greek_covered_volume += size
            signed_delta_demand += sign * size * multiplier * (delta or 0.0)
            signed_gamma_demand += sign * size * multiplier * (gamma or 0.0)
            signed_vega_demand += sign * size * multiplier * (vega or 0.0)

            if sign:
                size_weight = size * log1p(size)
                weighting_numerators["size"] += sign * size_weight
                weighting_denominators["size"] += size_weight
                weighting_covered_volume["size"] += size

                if item.size_to_displayed_depth is not None:
                    depth_weight = size * min(item.size_to_displayed_depth, 10.0)
                    weighting_numerators["depth"] += sign * depth_weight
                    weighting_denominators["depth"] += depth_weight
                    weighting_covered_volume["depth"] += size

                if item.price_location is not None:
                    aggressiveness = min(2.0, 2.0 * abs(item.price_location))
                    aggressiveness_weight = size * aggressiveness
                    weighting_numerators["aggressiveness"] += (
                        sign * aggressiveness_weight
                    )
                    weighting_denominators["aggressiveness"] += aggressiveness_weight
                    weighting_covered_volume["aggressiveness"] += size

                for name, greek in (
                    ("delta", delta),
                    ("gamma", gamma),
                    ("vega", vega),
                ):
                    if greek is None:
                        continue
                    exposure = size * multiplier * abs(greek)
                    direction = sign * (1.0 if greek >= 0 else -1.0)
                    weighting_numerators[name] += direction * exposure
                    weighting_denominators[name] += exposure
                    weighting_covered_volume[name] += size

            surface_bucket, days_to_expiry = _surface_bucket(trade, delta)
            if sign and surface_bucket is not None:
                surface_signed_flow[surface_bucket] += sign * size
                surface_covered_volume += size

            pretrade_mid, pretrade_spread = _quote_midpoint_and_spread(item.quote)
            if (
                not sign
                or not size
                or pretrade_mid is None
                or pretrade_spread is None
            ):
                continue
            price_scale = max(pretrade_spread, 0.01)
            for horizon in self.response_horizons_seconds:
                horizon_time = trade.event_time + timedelta(seconds=horizon)
                if horizon_time > decision.timestamp:
                    continue
                response_matured_volume[horizon] += size
                response_quote, _ = classifier.quote_as_of(
                    trade.instrument.alignment_key,
                    horizon_time,
                    decision.timestamp,
                )
                response_mid, _ = _quote_midpoint_and_spread(response_quote)
                if response_mid is None:
                    continue
                response_covered_volume[horizon] += size
                response_trade_count[horizon] += 1
                quote_impact = sign * (response_mid - pretrade_mid) / price_scale
                realized_spread = sign * (price - response_mid) / price_scale
                quote_impact_numerator[horizon] += size * quote_impact
                realized_spread_numerator[horizon] += size * realized_spread

                if (
                    horizon == persistent_horizon
                    and surface_bucket is not None
                    and delta is not None
                    and days_to_expiry is not None
                ):
                    persistent_weight = size * max(quote_impact, 0.0)
                    if persistent_weight <= 0:
                        continue
                    persistent_surface_impact[surface_bucket] += persistent_weight
                    persistent_impact_total += persistent_weight
                    persistent_delta_numerator += persistent_weight * abs(delta)
                    persistent_tenor_numerator += persistent_weight * days_to_expiry
                    option_type = trade.instrument.option_type
                    if option_type is not None:
                        if option_type.value == "put" and abs(delta) <= 0.40:
                            downside_put_impact += persistent_weight
                        if option_type.value == "call" and abs(delta) <= 0.40:
                            upside_call_impact += persistent_weight
                    if 0.40 < abs(delta) <= 0.60 and vega is not None:
                        atm_vega_impact += persistent_weight
                    if days_to_expiry <= 30:
                        short_dated_impact += persistent_weight

            if trade.instrument.underlying:
                underlyings.add(trade.instrument.underlying)
            contracts.add(trade.instrument.alignment_key)
            package_id = payload.get("package_id")
            package_keys.add(str(package_id) if package_id else f"singleton:{index}")

        latest_quotes: dict[str, CanonicalRecord] = {}
        for quote in quotes:
            key = quote.instrument.alignment_key
            previous = latest_quotes.get(key)
            if previous is None or (
                quote.available_time,
                quote.event_time,
                quote.ingested_time,
            ) > (
                previous.available_time,
                previous.event_time,
                previous.ingested_time,
            ):
                latest_quotes[key] = quote

        relative_spreads: list[float] = []
        depth_imbalances: list[float] = []
        latest_quote_ages: list[float] = []
        for quote in latest_quotes.values():
            bid = _number(quote.payload.get("bid"))
            ask = _number(quote.payload.get("ask"))
            bid_size = _number(quote.payload.get("bid_size"))
            ask_size = _number(quote.payload.get("ask_size"))
            if bid is not None and ask is not None and ask >= bid:
                mid = (bid + ask) / 2
                if mid > 0:
                    relative_spreads.append((ask - bid) / mid)
            if (
                bid_size is not None
                and ask_size is not None
                and bid_size + ask_size > 0
            ):
                depth_imbalances.append(
                    (bid_size - ask_size) / (bid_size + ask_size)
                )
            latest_quote_ages.append(
                max(0.0, (decision.timestamp - quote.event_time).total_seconds())
            )

        max_event_time = max((record.event_time for record in known), default=None)
        max_available_time = max(
            (record.available_time for record in known),
            default=None,
        )
        session_elapsed = max(
            0.0,
            (decision.timestamp - session_open).total_seconds(),
        )
        surface_hhi, surface_entropy, top_surface_share, top_surface_buckets = (
            _concentration(surface_signed_flow)
        )
        (
            impact_surface_hhi,
            _,
            _,
            top_impact_surface_buckets,
        ) = _concentration(persistent_surface_impact)

        state = {
            "schema_version": self.SCHEMA_VERSION,
            "date": decision.timestamp.date().isoformat(),
            "scope": "underlying" if underlying is not None else "universe",
            "underlying": underlying,
            "step": decision.kind.value,
            "decision_time": decision.timestamp.isoformat(),
            "information_cutoff": decision.timestamp.isoformat(),
            "session_open": session_open.isoformat(),
            "session_elapsed_seconds": session_elapsed,
            "information_set_record_count": len(known),
            "max_known_event_time": max_event_time.isoformat()
            if max_event_time
            else None,
            "max_known_available_time": max_available_time.isoformat()
            if max_available_time
            else None,
            "underlying_count": len(underlyings),
            "contract_count": len(contracts),
            "package_count": len(package_keys),
            "trade_count": len(classified),
            "total_contract_volume": total_volume,
            "buy_contract_volume": buy_volume,
            "sell_contract_volume": sell_volume,
            "unknown_contract_volume": unknown_volume,
            "signed_contract_demand": signed_contract_demand,
            "contract_volume_imbalance": _ratio(
                buy_volume - sell_volume,
                buy_volume + sell_volume,
            ),
            "total_option_notional": total_notional,
            "signed_option_notional": signed_notional,
            "mean_trade_size": mean(trade_sizes) if trade_sizes else 0.0,
            "large_trade_volume_share": _ratio(large_volume, total_volume),
            "opra_auction_volume_share": _ratio(auction_volume, total_volume),
            "price_improvement_auction_volume_share": _ratio(
                auction_volume,
                total_volume,
            ),
            "mechanical_volume_share": _ratio(mechanical_volume, total_volume),
            "quote_rule_classified_volume_share": _ratio(
                classified_volume,
                total_volume,
            ),
            "quote_rule_classified_trade_share": _ratio(
                sum(
                    1
                    for item in classified
                    if item.direction != TradeDirection.UNKNOWN
                ),
                len(classified),
            ),
            "quote_rule_buy_trade_count": sum(
                1 for item in classified if item.direction == TradeDirection.BUY
            ),
            "quote_rule_sell_trade_count": sum(
                1 for item in classified if item.direction == TradeDirection.SELL
            ),
            "quote_rule_unknown_trade_count": sum(
                1 for item in classified if item.direction == TradeDirection.UNKNOWN
            ),
            "quote_rule_midpoint_fallback_count": (
                methods.get("tick", 0) + methods.get("zero_tick", 0)
            ),
            "mean_trade_quote_age_seconds": mean(quote_ages)
            if quote_ages
            else None,
            "mean_trade_price_location": mean(price_locations)
            if price_locations
            else None,
            "mean_size_to_displayed_depth": mean(size_to_depth_values)
            if size_to_depth_values
            else None,
            "informed_candidate_volume_share": None,
            "uninformed_candidate_volume_share": None,
            "informed_signed_demand_proxy": None,
            "uninformed_signed_demand_proxy": None,
            "residual_signed_demand": signed_contract_demand,
            "signed_delta_demand": signed_delta_demand,
            "signed_gamma_demand": signed_gamma_demand,
            "signed_vega_demand": signed_vega_demand,
            "greek_covered_volume_share": _ratio(
                greek_covered_volume,
                classified_volume,
            ),
            "quoted_contract_count": len(latest_quotes),
            "median_relative_spread": median(relative_spreads)
            if relative_spreads
            else None,
            "mean_relative_spread": mean(relative_spreads)
            if relative_spreads
            else None,
            "mean_quote_depth_imbalance": mean(depth_imbalances)
            if depth_imbalances
            else None,
            "mean_latest_quote_age_seconds": mean(latest_quote_ages)
            if latest_quote_ages
            else None,
            "size_weighted_flow_imbalance": _ratio(
                weighting_numerators["size"],
                weighting_denominators["size"],
            ),
            "depth_weighted_flow_imbalance": _ratio(
                weighting_numerators["depth"],
                weighting_denominators["depth"],
            ),
            "aggressiveness_weighted_flow_imbalance": _ratio(
                weighting_numerators["aggressiveness"],
                weighting_denominators["aggressiveness"],
            ),
            "delta_weighted_flow_imbalance": _ratio(
                weighting_numerators["delta"],
                weighting_denominators["delta"],
            ),
            "gamma_weighted_flow_imbalance": _ratio(
                weighting_numerators["gamma"],
                weighting_denominators["gamma"],
            ),
            "vega_weighted_flow_imbalance": _ratio(
                weighting_numerators["vega"],
                weighting_denominators["vega"],
            ),
            "depth_weight_covered_volume_share": _ratio(
                weighting_covered_volume["depth"],
                classified_volume,
            ),
            "aggressiveness_weight_covered_volume_share": _ratio(
                weighting_covered_volume["aggressiveness"],
                classified_volume,
            ),
            "surface_flow_coverage_volume_share": _ratio(
                surface_covered_volume,
                classified_volume,
            ),
            "surface_flow_hhi": surface_hhi,
            "surface_flow_entropy": surface_entropy,
            "top_surface_bucket_share": top_surface_share,
            "top_surface_buckets": top_surface_buckets,
            "persistent_impact_horizon_seconds": persistent_horizon,
            "impact_weighted_surface_hhi": impact_surface_hhi,
            "impact_weighted_abs_delta_centroid": _ratio(
                persistent_delta_numerator,
                persistent_impact_total,
            )
            if persistent_impact_total
            else None,
            "impact_weighted_tenor_days_centroid": _ratio(
                persistent_tenor_numerator,
                persistent_impact_total,
            )
            if persistent_impact_total
            else None,
            "downside_put_persistent_impact_share": _ratio(
                downside_put_impact,
                persistent_impact_total,
            ),
            "upside_call_persistent_impact_share": _ratio(
                upside_call_impact,
                persistent_impact_total,
            ),
            "atm_vega_persistent_impact_share": _ratio(
                atm_vega_impact,
                persistent_impact_total,
            ),
            "short_dated_persistent_impact_share": _ratio(
                short_dated_impact,
                persistent_impact_total,
            ),
            "top_persistent_impact_surface_buckets": top_impact_surface_buckets,
        }
        for horizon in self.response_horizons_seconds:
            suffix = f"{horizon}s"
            state[f"quote_impact_{suffix}"] = (
                _ratio(
                    quote_impact_numerator[horizon],
                    response_covered_volume[horizon],
                )
                if response_covered_volume[horizon]
                else None
            )
            state[f"realized_spread_{suffix}"] = (
                _ratio(
                    realized_spread_numerator[horizon],
                    response_covered_volume[horizon],
                )
                if response_covered_volume[horizon]
                else None
            )
            state[f"quote_response_matured_volume_share_{suffix}"] = _ratio(
                response_matured_volume[horizon],
                classified_volume,
            )
            state[f"quote_response_coverage_{suffix}"] = _ratio(
                response_covered_volume[horizon],
                response_matured_volume[horizon],
            )
            state[f"quote_response_trade_count_{suffix}"] = response_trade_count[
                horizon
            ]
        return state
