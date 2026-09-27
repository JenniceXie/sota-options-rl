"""Reading the materialized feature datasets, point-in-time.

One awkwardness this module exists to absorb: the datasets under
``$PM_DATA_ROOT/features`` are **not** all the same format.  Measured
2026-09-09:

===============================  ================
dataset                          file
===============================  ================
``underlying_market_features``   ``part-000.jsonl``
``option_iv_surface_features``   ``part-000.jsonl``
``option_open_interest_features`` ``part-000.jsonl``
``compact_option_flow_states``   ``part-000.jsonl``
``option_chain_snapshots``       ``part-000.parquet``
===============================  ================

``tools/resolve_strategy_handle.py`` globs ``part-*.jsonl`` and therefore reads
nothing at all from the chain.  Rather than propagate that assumption, the
reader here dispatches on what is actually on disk, so a rebuild that changes
format does not change any caller.

The step vocabulary differs too: the datasets say ``market_open`` /
``market_close`` while the environment and the state space say ``AM`` / ``PM``.
The translation is here and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .statespace import FeatureRecord

__all__ = [
    "DatasetError",
    "SESSION_TO_STEP",
    "STEP_TO_SESSION",
    "data_root",
    "partition_dir",
    "available_dates",
    "read_partition",
    "ParquetFeatureSource",
    "DatasetFeatureSource",
]


class DatasetError(RuntimeError):
    """Raised when a dataset is absent, empty, or in an unreadable format."""


SESSION_TO_STEP: Mapping[str, str] = {"AM": "market_open", "PM": "market_close"}
STEP_TO_SESSION: Mapping[str, str] = {v: k for k, v in SESSION_TO_STEP.items()}

#: Which column identifies the instrument.  ``underlying_market_features`` uses
#: ``underlying``; some builders write ``ticker``.  Both are accepted, in this
#: order, so a rebuild that renames the column does not break the join silently.
_KEY_COLUMNS = ("underlying", "ticker", "symbol")


def data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    value = os.environ.get("PM_DATA_ROOT")
    if not value:
        raise DatasetError("PM_DATA_ROOT is not set and no data root was passed")
    return Path(value)


def partition_dir(root: Path, dataset: str, trade_date: date) -> Path:
    return root / dataset / f"date={trade_date.isoformat()}"


def available_dates(root: Path, dataset: str) -> tuple[date, ...]:
    base = root / dataset
    if not base.is_dir():
        raise DatasetError(f"dataset {dataset!r} is not under {root}")
    dates = []
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("date="):
            try:
                dates.append(date.fromisoformat(entry.name[5:]))
            except ValueError:
                continue
    return tuple(sorted(dates))


def read_partition(
    root: Path,
    dataset: str,
    trade_date: date,
    columns: Sequence[str] | None = None,
    *,
    equals: Mapping[str, Any] | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Yield rows of one date partition, whatever format it is stored in.

    ``columns`` is a projection hint, honoured for parquet (where it is a real
    saving on a 120-column chain) and ignored for JSONL (where the whole line
    must be parsed anyway).

    ``equals`` is a row filter of ``column -> required value``, and unlike
    ``columns`` it is honoured for *both* formats, because it changes which rows
    a caller sees rather than only how fast they arrive.  For parquet it is
    pushed into the reader; for JSONL it is applied here.  A caller that also
    re-checks the predicate itself will simply never see it fire.

    Pushing it down is worth having because a date partition of
    ``option_chain_snapshots`` holds both sessions interleaved: a PM-only reader
    that filters in Python still pays to decode and box ~37k market_open rows it
    is about to discard, and that boxing -- not the disk read -- is the cost.
    Measured on ``date=2025-04-01`` (76,491 rows, 41 columns projected):
    3.53s unfiltered against 1.85s for ``equals={"step": "market_close"}``.

    Row-group statistics do not prune here (nine of ten groups span both
    sessions), so the saving is entirely in rows never materialised, not in
    bytes never read.  That is worth knowing before anyone expects this to scale
    with a sorted rewrite of the files.
    """
    directory = partition_dir(root, dataset, trade_date)
    if not directory.is_dir():
        raise DatasetError(f"no partition for {dataset} on {trade_date.isoformat()}")

    parquet = sorted(directory.glob("*.parquet"))
    if parquet:
        yield from _read_parquet(parquet, columns, equals)
        return

    jsonl = sorted(directory.glob("*.jsonl"))
    if jsonl:
        rows = _read_jsonl(jsonl)
        if equals:
            rows = (r for r in rows if all(r.get(k) == v for k, v in equals.items()))
        yield from rows
        return

    raise DatasetError(f"{directory} contains neither parquet nor jsonl parts")


def _read_parquet(
    paths: Sequence[Path],
    columns: Sequence[str] | None,
    equals: Mapping[str, Any] | None = None,
) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise DatasetError(
            "reading option_chain_snapshots needs pyarrow; install the 'chain' extra"
        ) from exc
    # A filter column need not be projected: pyarrow reads it to evaluate the
    # predicate and drops it afterwards.
    filters = [(k, "==", v) for k, v in equals.items()] if equals else None
    for path in paths:
        table = pq.read_table(
            path, columns=list(columns) if columns else None, filters=filters
        )
        yield from table.to_pylist()


def _read_jsonl(paths: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ---------------------------------------------------------------------------


class DatasetFeatureSource:
    """``FeatureSource`` over ``$PM_DATA_ROOT/features``.

    Partitions are cached per ``(dataset, date, session, keys)`` because a step
    reads four datasets and the next step reads the same four for the same date.
    The cache is bounded and the entries are small — they hold the universe's
    rows, not the partition's — and the chain is not read through this class at
    all (see ``chain.py``).
    """

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        prefix: str = "features",
        cache_size: int = 32,
        extract_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._root = data_root(root) / prefix
        self._cache: dict[
            tuple[str, date, str, tuple[str, ...]], Mapping[str, FeatureRecord]
        ] = {}
        self._cache_size = cache_size
        self._extract_dir = Path(extract_dir) if extract_dir is not None else None

    @property
    def root(self) -> Path:
        return self._root

    def coverage(self, dataset: str) -> tuple[date, ...]:
        return available_dates(self._root, dataset)

    def fetch(
        self,
        dataset: str,
        *,
        trade_date: date,
        session: str,
        keys: Sequence[str],
        decision_time: datetime,
    ) -> Mapping[str, FeatureRecord]:
        table = self._partition(dataset, trade_date, session, decision_time, tuple(keys))
        return {key: table[key] for key in keys if key in table}

    def _partition(
        self,
        dataset: str,
        trade_date: date,
        session: str,
        decision_time: datetime,
        keys: tuple[str, ...],
    ) -> Mapping[str, FeatureRecord]:
        cache_key = (dataset, trade_date, session, keys)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # The partition holds the whole listed market — 22k rows where the
        # universe wants six — so the key check comes *before* ``_flatten``,
        # which is otherwise the second-largest cost in the step loop.
        wanted = frozenset(keys)
        step = SESSION_TO_STEP.get(session, session)
        records: dict[str, FeatureRecord] = {}
        for row in self._rows(dataset, trade_date, wanted):
            if row.get("step") not in (step, None):
                continue
            key = _row_key(row)
            if key is None or key not in wanted:
                continue
            records[key] = FeatureRecord(
                dataset=dataset,
                key=key,
                values=_flatten(row),
                available_time=_parse_time(row.get("available_time")),
                decision_time=_parse_time(row.get("decision_time")) or decision_time,
                point_in_time_rule=row.get("point_in_time_rule"),
            )

        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = records
        return records

    # -- the on-disk extract ---------------------------------------------

    def _rows(
        self, dataset: str, trade_date: date, wanted: frozenset[str]
    ) -> Sequence[Mapping[str, Any]]:
        """The partition's rows for ``wanted``, through the extract if enabled.

        A feature partition holds the whole listed market: 40 MB and 22k rows
        per dataset per date, of which the universe wants six.  ``/ocean`` reads
        cold at roughly 8 MB/s, so a single 63-date arm spends about twenty
        minutes reading bytes it discards, and an RL rollout would pay that on
        every pass.  The extract is that filter, done once and kept.

        Deliberately *not* filtered by session: one extract serves AM and PM,
        and the point-in-time fields travel with the row so the guard still runs
        against the same stamps the source carried.
        """
        if self._extract_dir is None:
            return list(read_partition(self._root, dataset, trade_date))

        path = self._extract_path(dataset, trade_date, wanted)
        if path.exists():
            return list(_read_jsonl([path]))

        rows = [
            row
            for row in read_partition(self._root, dataset, trade_date)
            if _row_key(row) in wanted
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written under a temporary name and renamed: a job preempted mid-write
        # would otherwise leave a truncated extract that every later run would
        # read as complete.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        tmp.replace(path)
        return rows

    def _extract_path(self, dataset: str, trade_date: date, wanted: frozenset[str]) -> Path:
        """Name the extract after the bytes it was taken from.

        The signature is every source part's size and mtime, so a rebuilt
        feature partition misses the cache instead of being served last week's
        numbers under this week's name.
        """
        directory = partition_dir(self._root, dataset, trade_date)
        if not directory.is_dir():
            raise DatasetError(f"no partition for {dataset} on {trade_date.isoformat()}")
        signature = [
            (part.name, part.stat().st_size, part.stat().st_mtime_ns)
            for part in sorted(directory.iterdir())
            if part.is_file()
        ]
        digest = hashlib.sha256(
            repr((sorted(wanted), signature)).encode("utf-8")
        ).hexdigest()[:16]
        return self._extract_dir / dataset / f"{trade_date.isoformat()}.{digest}.jsonl"


def _row_key(row: Mapping[str, Any]) -> str | None:
    for column in _KEY_COLUMNS:
        value = row.get(column)
        if value:
            return str(value)
    return None


def _flatten(row: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten one level of nesting into dotted names.

    ``docs/state_space.md`` names ``surface_change_step.atm_iv_30d``, which is a
    nested object in the JSONL.  Flattening here means the state space asks for
    the field by the name the document uses, which is the only name a reader of
    that document knows.
    """
    flat: dict[str, Any] = {}
    for name, value in row.items():
        full = f"{prefix}{name}"
        if isinstance(value, Mapping):
            flat[full] = value
            flat.update(_flatten(value, prefix=f"{full}."))
        else:
            flat[full] = value
    return flat


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# Kept as an alias: several call sites in the jobs package refer to the source
# by what it reads rather than by where it reads from.
ParquetFeatureSource = DatasetFeatureSource
