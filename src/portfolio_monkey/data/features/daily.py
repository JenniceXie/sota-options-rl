"""Adaptive daily feature generation.

The builder emits only features supported by the input records. This keeps the
same code path usable for sparse public feeds and richer licensed datasets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

from portfolio_monkey.data.schemas import CanonicalRecord, DataSource, RecordType

Number = int | float


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _put(features: dict[str, float], name: str, value: object) -> None:
    number = _number(value)
    if number is not None:
        features[name] = number


@dataclass(frozen=True, slots=True)
class AdaptiveDailyFeatureBuilder:
    """Build daily features without requiring a fixed vendor field set."""

    source: DataSource = DataSource.INTERNAL

    def build(self, records: Iterable[CanonicalRecord]) -> list[CanonicalRecord]:
        grouped: dict[str, list[CanonicalRecord]] = defaultdict(list)
        for record in records:
            grouped[record.instrument.alignment_key].append(record)

        feature_records: list[CanonicalRecord] = []
        for key, group in grouped.items():
            latest = max(group, key=lambda item: (item.available_time, item.event_time))
            payload = self._features_for_group(group)
            if not payload:
                continue
            feature_records.append(
                CanonicalRecord(
                    source=self.source,
                    record_type=RecordType.DAILY_FEATURE,
                    instrument=latest.instrument,
                    event_time=max(record.event_time for record in group),
                    available_time=max(record.available_time for record in group),
                    ingested_time=max(record.ingested_time for record in group),
                    payload=payload,
                    vendor_id=f"daily:{key}",
                )
            )
        return sorted(feature_records, key=lambda record: record.instrument.alignment_key)

    def _features_for_group(self, records: list[CanonicalRecord]) -> dict[str, Number]:
        features: dict[str, Number] = {}
        latest = max(records, key=lambda item: (item.available_time, item.event_time))
        payload = latest.payload

        open_price = _number(payload.get("open"))
        close_price = _number(payload.get("close"))
        high = _number(payload.get("high"))
        low = _number(payload.get("low"))
        volume = _number(payload.get("volume"))
        bid = _number(payload.get("bid"))
        ask = _number(payload.get("ask"))
        mid = _number(payload.get("mid"))
        underlying_price = _number(payload.get("underlying_price"))

        _put(features, "open", open_price)
        _put(features, "close", close_price)
        _put(features, "high", high)
        _put(features, "low", low)
        _put(features, "volume", volume)
        _put(features, "vwap", payload.get("vwap"))
        _put(features, "transactions", payload.get("transactions"))
        _put(features, "open_interest", payload.get("open_interest"))
        _put(features, "implied_volatility", payload.get("implied_volatility"))

        for greek in ("delta", "gamma", "theta", "vega"):
            _put(features, greek, payload.get(greek))

        if open_price and close_price:
            features["return"] = close_price / open_price - 1.0
        if low and high and low > 0:
            features["range_pct"] = high / low - 1.0
        if bid is not None and ask is not None:
            calculated_mid = mid if mid and mid > 0 else (bid + ask) / 2.0
            features["mid"] = calculated_mid
            features["bid_ask_spread"] = ask - bid
            if calculated_mid:
                features["relative_spread"] = (ask - bid) / calculated_mid
        if underlying_price and latest.instrument.strike:
            features["moneyness"] = underlying_price / latest.instrument.strike
            # Keep the legacy spot/strike feature above for compatibility, but
            # expose the option-selection convention explicitly.  Covered-call
            # rules are normally stated as strike relative to spot.
            features["strike_over_spot"] = latest.instrument.strike / underlying_price
        if latest.instrument.expiry:
            days_to_expiry = (latest.instrument.expiry - latest.event_time.date()).days
            features["days_to_expiry"] = float(days_to_expiry)

        volumes = [
            value
            for record in records
            if (value := _number(record.payload.get("volume"))) is not None
        ]
        if len(volumes) > 1:
            features["mean_volume"] = mean(volumes)
            features["volume_count"] = float(len(volumes))

        return features
