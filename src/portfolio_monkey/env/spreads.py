"""The underlying's proportional trading cost ``k``, per name and per session.

``k`` is the relative **half**-spread, ``0.5 * (ask - bid) / mid``, because both
consumers charge one crossing: the Whalley-Wilmott band prices a correction at
``k * S`` per share, and the share fill moves the price by ``k * S``.  Storing
the full spread and halving it at two call sites is how the two drift apart.

Built by ``scripts/analysis/build_underlying_spreads.py``; see that module for
why PM comes from CRSP's closing NBBO, why AM is a scaled version of it rather
than a measurement, and why both are trailing medians.

**Why a session key and not just a date.**  The opening spread is 4-5x the
closing spread on the nine single names and 1.0x on SPY.  Collapsing the two
sessions onto one daily number would understate the AM cost by a factor that is
itself name-dependent, which is worse than a flat cost: it would look like a
per-name model while carrying a per-name error.

**Missing rows fall back, they do not fail.**  A hedge that raises on a data gap
turns a missing spread into a missing episode.  The fallback is the flat cost
the environment already assumed, and every lookup that used it is counted, so
"the table covered everything" is a positive statement rather than an absence of
complaints.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

__all__ = ["SpreadTable", "SpreadTableError"]


class SpreadTableError(RuntimeError):
    """Raised when the spread table cannot be read or is malformed."""


@dataclass(frozen=True, slots=True)
class _Key:
    day: date
    ticker: str


class SpreadTable:
    """Per-(date, ticker) AM and PM half-spreads, with counted fallbacks."""

    def __init__(
        self,
        rows: Mapping[tuple[date, str], tuple[float | None, float | None]],
        *,
        fallback: float,
        source: str = "",
    ) -> None:
        self._rows = dict(rows)
        self._fallback = fallback
        self._source = source
        self._hits = 0
        self._misses: dict[str, int] = {}

    @classmethod
    def from_csv(cls, path: str | Path, *, fallback: float) -> "SpreadTable":
        path = Path(path)
        if not path.exists():
            raise SpreadTableError(f"no spread table at {path}")
        rows: dict[tuple[date, str], tuple[float | None, float | None]] = {}
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"date", "ticker", "k_am", "k_pm"}
            if not required.issubset(reader.fieldnames or ()):
                raise SpreadTableError(
                    f"{path} needs columns {sorted(required)}, has {reader.fieldnames}"
                )
            for row in reader:
                key = (date.fromisoformat(row["date"][:10]), row["ticker"])
                rows[key] = (_maybe_float(row["k_am"]), _maybe_float(row["k_pm"]))
        if not rows:
            raise SpreadTableError(f"{path} is empty")
        return cls(rows, fallback=fallback, source=str(path))

    @classmethod
    def flat(cls, value: float) -> "SpreadTable":
        """A table that answers ``value`` for everything.

        This is what ``band_rule='whalley_wilmott'`` gets when no table is
        configured, and it is the control the sweep needs: it isolates the
        gamma scaling of the band from the spread scaling, because with a
        constant ``k`` only ``Gamma`` still moves the band.
        """
        return cls({}, fallback=value, source="flat")

    def half_spread(self, ticker: str, day: date, *, session: str) -> float:
        """``k`` for one name at one session.

        Missing *rows* fall back and are counted; an unrecognised *session*
        raises.  The two are different failures: a missing row is a data gap
        this table exists to survive, while an unknown session is a caller
        reading the wrong column, and falling back would hide it behind a
        number that looks entirely reasonable.
        """
        entry = self._rows.get((day, ticker))
        if entry is not None:
            value = entry[_session_index(session)]
            if value is not None and value > 0.0:
                self._hits += 1
                return value
        self._misses[ticker] = self._misses.get(ticker, 0) + 1
        return self._fallback

    def present(
        self, tickers: Iterable[str], days: Iterable[date], *, session: str
    ) -> dict[str, int]:
        """Per-name count of days that would answer from the table.

        The ``coverage`` property is the honest measure because it counts real
        lookups, but it is only available once the run has been paid for.  This
        is the same question asked before the chain is opened, so a table built
        for the wrong window or the wrong ticker spelling costs a second on the
        login node rather than an arm.  Per name, never per date: a table that
        is complete on nine names and empty on the tenth is 90% covered by date
        and has one name secretly running the flat fallback.
        """
        index = _session_index(session)
        days = list(days)
        counts: dict[str, int] = {}
        for ticker in tickers:
            counts[ticker] = sum(
                1
                for day in days
                if (entry := self._rows.get((day, ticker))) is not None
                and entry[index] is not None
                and entry[index] > 0.0
            )
        return counts

    @property
    def coverage(self) -> dict[str, object]:
        """Provenance for the manifest.

        Reported per *name*: a universe-wide hit count hides a name-shaped hole,
        and a name that always falls back is a name whose band is secretly the
        flat rule while the manifest says Whalley-Wilmott.
        """
        return {
            "source": self._source,
            "rows": len(self._rows),
            "hits": self._hits,
            "fallbacks": dict(sorted(self._misses.items())),
            "fallback_value": self._fallback,
        }


#: Which column each session name selects.  Both vocabularies are admitted
#: because both reach this table: the environment's grid says ``AM``/``PM``
#: (``GridPoint.session``, and it is a ``GridPoint`` the hedge is handed) while
#: the feature datasets say ``market_open``/``market_close``.  The canonical
#: translation lives in ``env/datasets.SESSION_TO_STEP``; this map is keyed by
#: both sides of it so that neither caller has to convert on the way in.
#:
#: The pair used to be ``0 if session == 'market_open' else 1``, which answered
#: the PM column for *every* unrecognised string -- and ``AM`` is unrecognised,
#: so the environment's own AM hedge would have been priced at the closing
#: spread, roughly a quarter of the true one, with no error anywhere.
_SESSION_COLUMN: Mapping[str, int] = {
    "AM": 0,
    "market_open": 0,
    "PM": 1,
    "market_close": 1,
}


def _session_index(session: str) -> int:
    try:
        return _SESSION_COLUMN[session]
    except KeyError:
        raise SpreadTableError(
            f"unknown session {session!r}; expected one of "
            f"{sorted(_SESSION_COLUMN)}"
        ) from None


def _maybe_float(raw: str | None) -> float | None:
    if raw is None or raw == "" or raw.lower() in {"nan", "none"}:
        return None
    return float(raw)
