"""Point-in-time as-of alignment for heterogeneous feeds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from portfolio_monkey.data.schemas import CanonicalRecord
from portfolio_monkey.env.clock import DecisionPoint


def _decision_timestamp(decision: datetime | DecisionPoint) -> datetime:
    if isinstance(decision, DecisionPoint):
        return decision.timestamp
    if decision.tzinfo is None or decision.utcoffset() is None:
        raise ValueError("decision timestamp must be timezone-aware")
    return decision.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class AlignedObservation:
    """Latest known records for one RL decision time."""

    decision_time: datetime
    feeds: Mapping[str, Mapping[str, CanonicalRecord]]

    def __post_init__(self) -> None:
        feed_copy = {
            feed_name: MappingProxyType(dict(records))
            for feed_name, records in self.feeds.items()
        }
        object.__setattr__(self, "feeds", MappingProxyType(feed_copy))

    def record(self, feed_name: str, instrument_key: str) -> CanonicalRecord | None:
        return self.feeds.get(feed_name, {}).get(instrument_key)


def latest_as_of(
    decision_time: datetime | DecisionPoint,
    records: Sequence[CanonicalRecord],
) -> dict[str, CanonicalRecord]:
    """Return latest record per instrument using ``available_time <= decision``."""

    timestamp = _decision_timestamp(decision_time)
    latest: dict[str, CanonicalRecord] = {}
    for record in records:
        if record.available_time > timestamp:
            continue
        key = record.instrument.alignment_key
        previous = latest.get(key)
        if previous is None:
            latest[key] = record
            continue
        if (record.available_time, record.event_time, record.ingested_time) > (
            previous.available_time,
            previous.event_time,
            previous.ingested_time,
        ):
            latest[key] = record
    return latest


def as_of_align(
    decisions: Iterable[datetime | DecisionPoint],
    feeds: Mapping[str, Sequence[CanonicalRecord]],
) -> list[AlignedObservation]:
    """Align named feeds to RL decision timestamps without look-ahead.

    The returned observation for each decision contains only records whose
    ``available_time`` is at or before the decision timestamp.
    """

    observations: list[AlignedObservation] = []
    for decision in decisions:
        timestamp = _decision_timestamp(decision)
        aligned = {
            feed_name: latest_as_of(timestamp, records)
            for feed_name, records in feeds.items()
        }
        observations.append(AlignedObservation(timestamp, aligned))
    return observations

