"""RL decision clock for scheduled and event-triggered trading decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Iterable
from zoneinfo import ZoneInfo


def _to_utc(value: datetime, field_name: str = "datetime") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class DecisionKind(str, Enum):
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    INTRADAY_EVENT = "intraday_event"


@dataclass(frozen=True, slots=True)
class ClockEvent:
    """A market/news/risk event that may trigger an RL decision."""

    event_id: str
    available_time: datetime
    kind: str = "event"

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_time", _to_utc(self.available_time, "available_time"))


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """One timestamp at which the RL policy may choose a new action."""

    timestamp: datetime
    kind: DecisionKind
    event_ids: tuple[str, ...] = ()
    queued_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _to_utc(self.timestamp, "timestamp"))
        object.__setattr__(self, "kind", DecisionKind(self.kind))


@dataclass(frozen=True, slots=True)
class MarketSessionConfig:
    """Regular-session schedule for the decision clock.

    This starter intentionally models weekdays plus an optional holiday list.
    Exchange half-days and full market calendars can be swapped in behind this
    interface later without changing downstream alignment code.
    """

    timezone_name: str = "America/New_York"
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    include_open: bool = True
    include_close: bool = True
    include_intraday_events: bool = True
    holidays: frozenset[date] = field(default_factory=frozenset)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)


class DecisionClock:
    """Generate scheduled and event-triggered RL decision timestamps."""

    def __init__(self, config: MarketSessionConfig | None = None) -> None:
        self.config = config or MarketSessionConfig()

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.config.holidays

    def session_open(self, day: date) -> datetime:
        local = datetime.combine(day, self.config.open_time, tzinfo=self.config.tz)
        return local.astimezone(timezone.utc)

    def session_close(self, day: date) -> datetime:
        local = datetime.combine(day, self.config.close_time, tzinfo=self.config.tz)
        return local.astimezone(timezone.utc)

    def next_trading_day(self, day: date) -> date:
        candidate = day
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def next_open_at_or_after(self, value: datetime) -> datetime:
        value = _to_utc(value)
        local = value.astimezone(self.config.tz)
        day = self.next_trading_day(local.date())
        open_at = self.session_open(day)
        if open_at >= value:
            return open_at
        return self.session_open(self.next_trading_day(day + timedelta(days=1)))

    def decisions(
        self,
        start: datetime,
        end: datetime,
        events: Iterable[ClockEvent] = (),
    ) -> list[DecisionPoint]:
        """Return all decision points in ``[start, end]``.

        Intraday events become their own decisions when enabled. Events outside
        regular hours are queued to the next market open.
        """

        start = _to_utc(start, "start")
        end = _to_utc(end, "end")
        if end < start:
            raise ValueError("end must be greater than or equal to start")

        scheduled: dict[datetime, DecisionPoint] = {}
        local_day = start.astimezone(self.config.tz).date()
        final_day = end.astimezone(self.config.tz).date()
        while local_day <= final_day:
            if self.is_trading_day(local_day):
                if self.config.include_open:
                    opened = self.session_open(local_day)
                    if start <= opened <= end:
                        scheduled[opened] = DecisionPoint(opened, DecisionKind.MARKET_OPEN)
                if self.config.include_close:
                    closed = self.session_close(local_day)
                    if start <= closed <= end:
                        scheduled[closed] = DecisionPoint(closed, DecisionKind.MARKET_CLOSE)
            local_day += timedelta(days=1)

        event_ids: dict[datetime, list[str]] = {}
        queued_ids: dict[datetime, list[str]] = {}
        for event in events:
            available = event.available_time
            local = available.astimezone(self.config.tz)
            day = local.date()

            if self.is_trading_day(day) and self.session_open(day) <= available < self.session_close(day):
                if self.config.include_intraday_events:
                    if start <= available <= end:
                        event_ids.setdefault(available, []).append(event.event_id)
            else:
                open_at = self.next_open_at_or_after(available)
                if start <= open_at <= end:
                    queued_ids.setdefault(open_at, []).append(event.event_id)

        timestamps = set(scheduled) | set(event_ids) | set(queued_ids)
        points: list[DecisionPoint] = []
        for timestamp in sorted(timestamps):
            existing = scheduled.get(timestamp)
            if existing is not None:
                points.append(
                    DecisionPoint(
                        timestamp=timestamp,
                        kind=existing.kind,
                        event_ids=tuple(event_ids.get(timestamp, ())),
                        queued_event_ids=tuple(queued_ids.get(timestamp, ())),
                    )
                )
            else:
                points.append(
                    DecisionPoint(
                        timestamp=timestamp,
                        kind=DecisionKind.INTRADAY_EVENT,
                        event_ids=tuple(event_ids.get(timestamp, ())),
                        queued_event_ids=tuple(queued_ids.get(timestamp, ())),
                    )
                )
        return points
