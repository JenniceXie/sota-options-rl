"""Stock-level context features from news, reports, and portfolio records."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from portfolio_monkey.data.schemas import CanonicalRecord, DataSource, RecordType
from portfolio_monkey.env.clock import DecisionPoint


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _side_sign(payload: Any) -> float:
    side = str(payload.get("side") or "").strip().lower()
    return -1.0 if side in {"sell", "short", "sold", "written", "write"} else 1.0


def stock_key(record: CanonicalRecord) -> str:
    """Return the stock-level key used by context-state aggregation."""

    return (
        record.instrument.underlying
        or record.instrument.symbol
        or "__UNKNOWN__"
    ).upper()


def context_evidence_handle(record: CanonicalRecord) -> str:
    """Return a stable handle for one causally available evidence version.

    ``available_time`` and content-bearing fields are deliberately part of the
    identity.  A later provider revision therefore cannot replace an earlier
    version that a market-open state already referenced.
    """

    revision = {
        key: record.payload.get(key)
        for key in (
            "article_url",
            "headline",
            "summary",
            "event_type",
            "scheduled_time",
            "surprise",
            "form",
            "items",
        )
        if record.payload.get(key) is not None
    }

    identity = "|".join(
        (
            record.source.value,
            record.record_type.value,
            record.vendor_id or "",
            stock_key(record),
            record.event_time.isoformat(),
            record.available_time.isoformat(),
            json.dumps(revision, sort_keys=True, separators=(",", ":")),
        )
    )
    return f"ctx_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"


class StockContextFeatureBuilder:
    """Build one fixed-schema context row per stock and decision step."""

    SCHEMA_VERSION = "stock_context_features.v5"

    def __init__(self, *, max_evidence_handles: int = 3) -> None:
        if max_evidence_handles < 0:
            raise ValueError("max_evidence_handles cannot be negative")
        self.max_evidence_handles = max_evidence_handles

    def build(
        self,
        decision: DecisionPoint,
        previous_decision_time: datetime,
        records: Sequence[CanonicalRecord],
        *,
        underlying: str,
        ingested_time: datetime | None = None,
    ) -> dict[str, Any]:
        if previous_decision_time >= decision.timestamp:
            raise ValueError("previous_decision_time must precede decision time")

        known_all = [
            record
            for record in records
            if record.available_time <= decision.timestamp
        ]
        scoped = [record for record in known_all if stock_key(record) == underlying]
        known = [
            record
            for record in scoped
            if record.available_time <= decision.timestamp
        ]
        arrived = [
            record
            for record in known
            if previous_decision_time < record.available_time <= decision.timestamp
        ]

        row = self._empty_row(
            decision=decision,
            previous_decision_time=previous_decision_time,
            underlying=underlying,
            ingested_time=ingested_time or datetime.now(timezone.utc),
        )
        self._add_news(row, arrived)
        self._add_events(row, known_all, decision.timestamp, underlying=underlying)
        self._add_occ(row, known)
        self._add_nport(
            row,
            known_all,
            decision.timestamp,
            underlying=underlying,
        )
        self._add_positions(row, known)
        self._add_activity(row, arrived)
        self._add_quality(row, known, arrived)
        row["feature_block_missing"] = {
            "news": row["news_count_step"] == 0,
            "events": not row["event_calendar_available"],
            "nport": not row["nport_available"],
            "occ": row["occ_activity_date"] is None,
            "portfolio": row["position_instrument_count"] == 0,
        }
        row["feature_block_stale"] = {
            "news": False,
            "events": False,
            "nport": (
                row["nport_max_report_age_days"] is not None
                and row["nport_max_report_age_days"] > 120
            ),
            "occ": (
                row["occ_report_age_days"] is not None
                and row["occ_report_age_days"] > 5
            ),
            "portfolio": False,
        }
        return row

    def _empty_row(
        self,
        *,
        decision: DecisionPoint,
        previous_decision_time: datetime,
        underlying: str,
        ingested_time: datetime,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "date": decision.timestamp.date().isoformat(),
            "underlying": underlying,
            "step": decision.kind.value,
            "decision_time": decision.timestamp.isoformat(),
            "event_time": decision.timestamp.isoformat(),
            "available_time": decision.timestamp.isoformat(),
            "ingested_time": ingested_time.isoformat(),
            "previous_decision_time": previous_decision_time.isoformat(),
            "information_cutoff": decision.timestamp.isoformat(),
            "step_seconds": (
                decision.timestamp - previous_decision_time
            ).total_seconds(),
            "session_elapsed_fraction": (
                0.0 if decision.kind.value == "market_open" else 1.0
            ),
            "underlying_count": 1,
            "news_count_step": 0,
            "news_direction_mean": None,
            "news_direction_max_abs": None,
            "news_uncertainty_mean": None,
            "news_novelty_mean": None,
            "news_catalyst_score": None,
            "news_provider_count_step": 0,
            "news_evidence_handles": [],
            "event_calendar_available": False,
            "hours_to_scheduled_event": None,
            "event_type": None,
            "event_surprise": None,
            "scheduled_event_count": 0,
            "event_evidence_handles": [],
            "ibkr_news_count_step": 0,
            "massive_news_count_step": 0,
            "sec_news_count_step": 0,
            "alpaca_news_count_step": 0,
            "occ_activity_date": None,
            "occ_report_age_days": None,
            "occ_account_side_volume": 0.0,
            "occ_customer_volume": 0.0,
            "occ_firm_volume": 0.0,
            "occ_market_maker_volume": 0.0,
            "occ_call_account_side_volume": 0.0,
            "occ_put_account_side_volume": 0.0,
            "occ_open_interest": 0.0,
            "occ_volume_to_open_interest": None,
            "occ_customer_volume_share": None,
            "occ_put_call_ratio": None,
            "nport_latest_report_date": None,
            "nport_max_report_age_days": None,
            "nport_fund_count": 0,
            "nport_holding_count": 0,
            "nport_long_value": 0.0,
            "nport_short_value": 0.0,
            "nport_net_value": 0.0,
            "nport_gross_notional": 0.0,
            "nport_net_delta": 0.0,
            "nport_net_gamma": 0.0,
            "nport_net_vega": 0.0,
            "nport_disclosed_position_pressure": 0.0,
            "nport_available": False,
            "position_instrument_count": 0,
            "position_quantity": 0.0,
            "position_market_value": 0.0,
            "position_nav_weight": None,
            "position_unrealized_pnl": 0.0,
            "position_delta": 0.0,
            "position_gamma": 0.0,
            "position_vega": 0.0,
            "position_theta": 0.0,
            "order_count_step": 0,
            "fill_count_step": 0,
            "buy_fill_notional_step": 0.0,
            "sell_fill_notional_step": 0.0,
            "signed_fill_notional_step": 0.0,
            "transaction_cost_step": 0.0,
            "ibkr_record_count": 0,
            "massive_record_count": 0,
            "sec_record_count": 0,
            "occ_record_count": 0,
            "nport_record_count": 0,
            "alpaca_record_count": 0,
            "internal_record_count": 0,
            "arrived_record_count": 0,
            "information_set_record_count": 0,
            "max_known_event_time": None,
            "max_known_available_time": None,
        }

    def _add_news(
        self,
        row: dict[str, Any],
        arrived: Sequence[CanonicalRecord],
    ) -> None:
        news = [record for record in arrived if record.record_type == RecordType.NEWS]
        deduplicated: dict[str, CanonicalRecord] = {}
        for record in sorted(
            news,
            key=lambda value: (
                value.available_time,
                value.event_time,
                value.ingested_time,
                value.source.value,
                value.vendor_id or "",
            ),
        ):
            article_url = str(record.payload.get("article_url") or "").strip()
            headline = " ".join(
                str(record.payload.get("headline") or "").lower().split()
            )
            key = (
                f"url:{article_url}"
                if article_url
                else (
                    f"headline:{record.event_time.isoformat()}:{headline}"
                    if headline
                    else f"{record.source.value}:{record.vendor_id}"
                )
            )
            previous = deduplicated.get(key)
            if previous is None or (
                record.available_time,
                record.event_time,
                record.ingested_time,
                record.source.value,
                record.vendor_id or "",
            ) > (
                previous.available_time,
                previous.event_time,
                previous.ingested_time,
                previous.source.value,
                previous.vendor_id or "",
            ):
                deduplicated[key] = record
        news = list(deduplicated.values())
        providers = {record.source for record in news}
        directions = [
            value
            for record in news
            if (value := _number(record.payload.get("direction"))) is not None
        ]
        uncertainties = [
            value
            for record in news
            if (value := _number(record.payload.get("uncertainty"))) is not None
        ]
        novelties = [
            value
            for record in news
            if (value := _number(record.payload.get("novelty"))) is not None
        ]
        catalyst_scores = []
        for record in news:
            direction = _number(record.payload.get("direction"))
            novelty = _number(record.payload.get("novelty"))
            uncertainty = _number(record.payload.get("uncertainty"))
            if direction is None:
                continue
            novelty_weight = novelty if novelty is not None else 1.0
            confidence_weight = 1.0 - uncertainty if uncertainty is not None else 1.0
            catalyst_scores.append(
                abs(direction) * max(0.0, novelty_weight) * max(0.0, confidence_weight)
            )

        row["news_count_step"] = len(news)
        row["news_direction_mean"] = _mean(directions)
        row["news_direction_max_abs"] = (
            max((abs(value) for value in directions), default=None)
        )
        row["news_uncertainty_mean"] = _mean(uncertainties)
        row["news_novelty_mean"] = _mean(novelties)
        row["news_catalyst_score"] = max(catalyst_scores, default=None)
        row["news_provider_count_step"] = len(providers)
        ranked_news = sorted(
            news,
            key=lambda record: (
                abs(_number(record.payload.get("direction")) or 0.0),
                record.available_time,
                record.event_time,
                record.source.value,
                record.vendor_id or "",
            ),
            reverse=True,
        )
        row["news_evidence_handles"] = [
            context_evidence_handle(record)
            for record in ranked_news[: self.max_evidence_handles]
        ]
        for source in (
            DataSource.IBKR,
            DataSource.MASSIVE,
            DataSource.SEC,
            DataSource.ALPACA,
        ):
            row[f"{source.value}_news_count_step"] = sum(
                1 for record in news if record.source == source
            )

    def _add_events(
        self,
        row: dict[str, Any],
        known: Sequence[CanonicalRecord],
        decision_time: datetime,
        *,
        underlying: str,
    ) -> None:
        candidates = [
            record
            for record in known
            if record.record_type == RecordType.EVENT
            and stock_key(record) in {underlying, "__MARKET__"}
        ]
        latest: dict[str, CanonicalRecord] = {}
        for record in candidates:
            key = str(
                record.payload.get("event_id")
                or record.vendor_id
                or (
                    f"{record.payload.get('event_type')}:"
                    f"{record.payload.get('scheduled_time')}"
                )
            )
            previous = latest.get(key)
            if previous is None or record.available_time > previous.available_time:
                latest[key] = record
        scheduled: list[tuple[datetime, CanonicalRecord]] = []
        completed: list[tuple[datetime, CanonicalRecord]] = []
        for record in latest.values():
            raw_time = record.payload.get("scheduled_time")
            if not isinstance(raw_time, str):
                continue
            scheduled_time = datetime.fromisoformat(
                raw_time.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            if scheduled_time >= decision_time:
                scheduled.append((scheduled_time, record))
            elif record.payload.get("surprise") is not None:
                completed.append((scheduled_time, record))
        row["event_calendar_available"] = bool(latest)
        row["scheduled_event_count"] = len(scheduled)
        ranked_events = sorted(
            latest.values(),
            key=lambda record: (
                record.available_time,
                record.event_time,
                record.vendor_id or "",
            ),
            reverse=True,
        )
        row["event_evidence_handles"] = [
            context_evidence_handle(record)
            for record in ranked_events[: self.max_evidence_handles]
        ]
        if scheduled:
            scheduled_time, event = min(scheduled, key=lambda item: item[0])
            row["hours_to_scheduled_event"] = (
                scheduled_time - decision_time
            ).total_seconds() / 3600
            row["event_type"] = event.payload.get("event_type")
        if completed:
            _, event = max(completed, key=lambda item: item[0])
            if row["event_type"] is None:
                row["event_type"] = event.payload.get("event_type")
            row["event_surprise"] = _number(event.payload.get("surprise"))

    def _add_occ(
        self,
        row: dict[str, Any],
        known: Sequence[CanonicalRecord],
    ) -> None:
        reports = [
            record
            for record in known
            if record.source == DataSource.OCC
            and record.record_type == RecordType.OPTION_VOLUME
        ]
        if not reports:
            return
        latest_date = max(record.event_time.date() for record in reports)
        latest = [record for record in reports if record.event_time.date() == latest_date]
        total = customer = firm = market_maker = calls = puts = 0.0
        open_interest_by_series: dict[tuple[Any, ...], float] = {}
        for record in latest:
            volume = _number(record.payload.get("volume")) or 0.0
            account_type = str(record.payload.get("account_type") or "").lower()
            total += volume
            if "customer" in account_type:
                customer += volume
            elif "market" in account_type and "maker" in account_type:
                market_maker += volume
            elif "firm" in account_type:
                firm += volume
            if record.instrument.option_type:
                if record.instrument.option_type.value == "call":
                    calls += volume
                elif record.instrument.option_type.value == "put":
                    puts += volume
            open_interest = _number(record.payload.get("open_interest"))
            if open_interest is not None:
                series_key = (
                    record.instrument.alignment_key,
                    record.instrument.expiry,
                    record.instrument.strike,
                    record.instrument.option_type,
                )
                open_interest_by_series[series_key] = max(
                    open_interest,
                    open_interest_by_series.get(series_key, 0.0),
                )

        row["occ_activity_date"] = latest_date.isoformat()
        row["occ_report_age_days"] = (
            datetime.fromisoformat(row["decision_time"]).date() - latest_date
        ).days
        row["occ_account_side_volume"] = total
        row["occ_customer_volume"] = customer
        row["occ_firm_volume"] = firm
        row["occ_market_maker_volume"] = market_maker
        row["occ_call_account_side_volume"] = calls
        row["occ_put_account_side_volume"] = puts
        row["occ_open_interest"] = sum(open_interest_by_series.values())
        row["occ_volume_to_open_interest"] = _ratio(
            total,
            row["occ_open_interest"],
        )
        row["occ_customer_volume_share"] = _ratio(customer, total)
        row["occ_put_call_ratio"] = _ratio(puts, calls)

    def _add_nport(
        self,
        row: dict[str, Any],
        known: Sequence[CanonicalRecord],
        decision_time: datetime,
        *,
        underlying: str,
    ) -> None:
        holdings = [
            record
            for record in known
            if record.source == DataSource.NPORT
            and record.record_type == RecordType.FUND_HOLDING
        ]
        if not holdings:
            return

        by_fund: dict[str, list[CanonicalRecord]] = defaultdict(list)
        for record in holdings:
            by_fund[str(record.payload.get("fund_id") or "__UNKNOWN__")].append(record)

        selected: list[CanonicalRecord] = []
        for fund_holdings in by_fund.values():
            latest_date = max(record.event_time.date() for record in fund_holdings)
            selected.extend(
                record
                for record in fund_holdings
                if record.event_time.date() == latest_date
                and stock_key(record) == underlying
            )
        if not selected:
            return

        selected_funds = {
            str(record.payload.get("fund_id")) for record in selected
        }

        long_value = short_value = gross_notional = 0.0
        net_delta = net_gamma = net_vega = 0.0
        report_dates: list[Any] = []
        for record in selected:
            payload = record.payload
            report_dates.append(record.event_time.date())
            position = str(
                payload.get("position") or payload.get("position_status") or ""
            ).lower()
            sign = -1.0 if position in {"written", "write", "short", "sold"} else 1.0
            quantity = abs(_number(payload.get("quantity")) or 0.0)
            value = abs(_number(payload.get("value")) or 0.0)
            notional = abs(_number(payload.get("notional")) or value)
            multiplier = _number(payload.get("contract_multiplier")) or (
                100.0 if record.instrument.contract_id else 1.0
            )
            if sign > 0:
                long_value += value
            else:
                short_value += value
            gross_notional += notional
            net_delta += sign * quantity * multiplier * (
                _number(payload.get("delta")) or 0.0
            )
            net_gamma += sign * quantity * multiplier * (
                _number(payload.get("gamma")) or 0.0
            )
            net_vega += sign * quantity * multiplier * (
                _number(payload.get("vega")) or 0.0
            )

        latest_report = max(report_dates)
        row["nport_latest_report_date"] = latest_report.isoformat()
        row["nport_max_report_age_days"] = max(
            (decision_time.date() - report_date).days for report_date in report_dates
        )
        row["nport_fund_count"] = len(selected_funds)
        row["nport_holding_count"] = len(selected)
        row["nport_long_value"] = long_value
        row["nport_short_value"] = short_value
        row["nport_net_value"] = long_value - short_value
        row["nport_gross_notional"] = gross_notional
        row["nport_net_delta"] = net_delta
        row["nport_net_gamma"] = net_gamma
        row["nport_net_vega"] = net_vega
        row["nport_disclosed_position_pressure"] = (
            net_vega if net_vega else net_delta
        )
        row["nport_available"] = True

    def _add_positions(
        self,
        row: dict[str, Any],
        known: Sequence[CanonicalRecord],
    ) -> None:
        positions = [
            record
            for record in known
            if record.source == DataSource.ALPACA
            and record.record_type == RecordType.POSITION
        ]
        latest: dict[tuple[str, str], CanonicalRecord] = {}
        for record in positions:
            key = (
                str(record.payload.get("account_id") or "__DEFAULT__"),
                record.instrument.alignment_key,
            )
            previous = latest.get(key)
            if previous is None or (
                record.available_time,
                record.event_time,
                record.ingested_time,
            ) > (
                previous.available_time,
                previous.event_time,
                previous.ingested_time,
            ):
                latest[key] = record

        nav_values: list[tuple[datetime, float]] = []
        for record in latest.values():
            payload = record.payload
            sign = _side_sign(payload)
            raw_quantity = _number(payload.get("quantity")) or 0.0
            quantity = raw_quantity if raw_quantity < 0 else raw_quantity * sign
            multiplier = _number(payload.get("contract_multiplier")) or (
                100.0 if record.instrument.contract_id else 1.0
            )
            row["position_quantity"] += quantity
            row["position_market_value"] += (
                _number(payload.get("market_value")) or 0.0
            ) * sign
            row["position_unrealized_pnl"] += (
                _number(payload.get("unrealized_pnl")) or 0.0
            )
            for greek in ("delta", "gamma", "vega", "theta"):
                row[f"position_{greek}"] += (
                    quantity
                    * multiplier
                    * (_number(payload.get(greek)) or 0.0)
                )
            nav = _number(payload.get("portfolio_nav"))
            if nav is not None:
                nav_values.append((record.available_time, nav))

        row["position_instrument_count"] = len(latest)
        if nav_values:
            nav = max(nav_values, key=lambda item: item[0])[1]
            row["position_nav_weight"] = _ratio(row["position_market_value"], nav)

    def _add_activity(
        self,
        row: dict[str, Any],
        arrived: Sequence[CanonicalRecord],
    ) -> None:
        for record in arrived:
            if record.source != DataSource.ALPACA:
                continue
            if record.record_type == RecordType.ORDER:
                row["order_count_step"] += 1
            elif record.record_type == RecordType.FILL:
                row["fill_count_step"] += 1
                payload = record.payload
                quantity = _number(payload.get("filled_quantity"))
                if quantity is None:
                    quantity = _number(payload.get("quantity")) or 0.0
                price = _number(payload.get("price")) or 0.0
                multiplier = _number(payload.get("contract_multiplier")) or (
                    100.0 if record.instrument.contract_id else 1.0
                )
                notional = abs(quantity * price * multiplier)
                sign = _side_sign(payload)
                if sign > 0:
                    row["buy_fill_notional_step"] += notional
                else:
                    row["sell_fill_notional_step"] += notional
                row["signed_fill_notional_step"] += sign * notional
                row["transaction_cost_step"] += abs(
                    _number(payload.get("commission")) or 0.0
                )

    def _add_quality(
        self,
        row: dict[str, Any],
        known: Sequence[CanonicalRecord],
        arrived: Sequence[CanonicalRecord],
    ) -> None:
        for source in (
            DataSource.IBKR,
            DataSource.MASSIVE,
            DataSource.SEC,
            DataSource.OCC,
            DataSource.NPORT,
            DataSource.ALPACA,
            DataSource.INTERNAL,
        ):
            row[f"{source.value}_record_count"] = sum(
                1 for record in known if record.source == source
            )
        row["arrived_record_count"] = len(arrived)
        row["information_set_record_count"] = len(known)
        if known:
            row["max_known_event_time"] = max(
                record.event_time for record in known
            ).isoformat()
            row["max_known_available_time"] = max(
                record.available_time for record in known
            ).isoformat()
