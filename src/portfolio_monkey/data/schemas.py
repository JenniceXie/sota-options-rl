"""Canonical point-in-time data records.

The key invariant for RL training data is that each observation carries both the
time the market event happened and the time the agent could have known it.
Alignment code must use ``available_time`` rather than ``event_time``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class DataSource(str, Enum):
    """Known upstream providers and internal feeds."""

    MASSIVE = "massive"
    OCC = "occ"
    NPORT = "nport"
    IBKR = "ibkr"
    ALPACA = "alpaca"
    SEC = "sec"
    SPIDERROCK = "spiderrock"
    INTERNAL = "internal"


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class RecordType(str, Enum):
    STOCK_TRADE = "stock_trade"
    STOCK_BAR = "stock_bar"
    STOCK_QUOTE = "stock_quote"
    OPTION_CONTRACT = "option_contract"
    OPTION_TRADE = "option_trade"
    OPTION_BAR = "option_bar"
    OPTION_QUOTE = "option_quote"
    OPTION_VOLUME = "option_volume"
    FUND_HOLDING = "fund_holding"
    NEWS = "news"
    EVENT = "event"
    POSITION = "position"
    ORDER = "order"
    FILL = "fill"
    FEATURE = "feature"
    DAILY_FEATURE = "daily_feature"


def _to_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class InstrumentKey:
    """Canonical security or option-contract identifiers.

    ``symbol`` is used for equities, funds, and underlyings. Option records can
    also carry OCC contract fields when available.
    """

    symbol: str | None = None
    contract_id: str | None = None
    underlying: str | None = None
    expiry: date | None = None
    strike: float | None = None
    option_type: OptionType | str | None = None
    exchange: str | None = None

    def __post_init__(self) -> None:
        if self.symbol is None and self.contract_id is None and self.underlying is None:
            raise ValueError("InstrumentKey needs at least symbol, contract_id, or underlying")
        if self.option_type is not None:
            object.__setattr__(self, "option_type", OptionType(self.option_type))

    @property
    def alignment_key(self) -> str:
        """Stable key used by state/alignment helpers."""

        if self.contract_id:
            return self.contract_id
        if self.symbol:
            return self.symbol
        assert self.underlying is not None
        return self.underlying


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    """Provider-neutral observation with point-in-time timestamps."""

    source: DataSource | str
    record_type: RecordType | str
    instrument: InstrumentKey
    event_time: datetime
    available_time: datetime
    ingested_time: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)
    vendor_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", DataSource(self.source))
        object.__setattr__(self, "record_type", RecordType(self.record_type))
        object.__setattr__(self, "event_time", _to_utc(self.event_time, "event_time"))
        object.__setattr__(
            self,
            "available_time",
            _to_utc(self.available_time, "available_time"),
        )
        object.__setattr__(
            self,
            "ingested_time",
            _to_utc(self.ingested_time, "ingested_time"),
        )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

        if self.available_time < self.event_time:
            raise ValueError("available_time cannot be earlier than event_time")
        if self.ingested_time < self.available_time:
            raise ValueError("ingested_time cannot be earlier than available_time")

    @property
    def symbol(self) -> str | None:
        return self.instrument.symbol

    @property
    def contract_id(self) -> str | None:
        return self.instrument.contract_id


def canonical_option_symbol(
    *,
    underlying: str,
    expiry: date,
    strike: float,
    option_type: OptionType | str,
) -> str:
    """Build a stable internal option key independent of vendor symbology."""

    option_type = OptionType(option_type)
    strike_text = f"{strike:.4f}".rstrip("0").rstrip(".")
    return f"{underlying.upper()}:{expiry.isoformat()}:{strike_text}:{option_type.value[0].upper()}"
