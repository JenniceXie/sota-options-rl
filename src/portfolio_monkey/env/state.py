"""Feature assembly from aligned point-in-time observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

from portfolio_monkey.env.align import AlignedObservation

Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class StateObservation:
    """Flat feature view consumed by starter RL environments."""

    decision_time: datetime
    features: Mapping[str, Scalar]
    aligned: AlignedObservation

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


class StateAssembler:
    """Build simple scalar features from aligned observations.

    Feature names are ``{feed}.{instrument}.{field}``. If ``feature_fields`` is
    provided, only the listed payload fields are included per feed.
    """

    def __init__(self, feature_fields: Mapping[str, tuple[str, ...]] | None = None) -> None:
        self.feature_fields = dict(feature_fields or {})

    def assemble(self, observation: AlignedObservation) -> StateObservation:
        features: dict[str, Scalar] = {}
        for feed_name, records_by_instrument in observation.feeds.items():
            allowed_fields = self.feature_fields.get(feed_name)
            for instrument_key, record in records_by_instrument.items():
                for field_name, value in record.payload.items():
                    if allowed_fields is not None and field_name not in allowed_fields:
                        continue
                    if isinstance(value, str | int | float | bool) or value is None:
                        features[f"{feed_name}.{instrument_key}.{field_name}"] = value

                features[f"{feed_name}.{instrument_key}.source"] = record.source.value
                features[f"{feed_name}.{instrument_key}.record_type"] = record.record_type.value
        return StateObservation(observation.decision_time, features, observation)

    def assemble_many(self, observations: list[AlignedObservation]) -> list[StateObservation]:
        return [self.assemble(observation) for observation in observations]

