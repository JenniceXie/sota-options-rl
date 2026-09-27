"""Per-contract NBBO for marking the open book, from Massive's REST API.

This exists because of one line in ``resolvers/market.py``: a leg with no quote
in the chain slice used to be carried at ``entry_price``.  That is not a mark.
A package carried at entry shows zero unrealized PnL for as long as it stays
unquoted, so the NAV series goes flat, the reward (log return) reads it as low
volatility rather than as missing data, and the error is *systematically*
signed — positions drift out of the chain when their quotes go wide or stale,
which is exactly when their value is moving most.

**Marking only.**  Nothing here may reach an execution path.  The chain remains
the single source of tradeable prices, with its build-time quality tiers and
its env-time recomputed ``true_age`` gate, and this module deliberately has
neither.  A quote good enough to value a position you already own is not the
same as a quote good enough to open one against, and conflating the two is how
a backtest fills at prices no one could have traded.

**Why the API and not the flat files.**  ``normalized/massive_option_quote_snapshots``
covers 64 of the 496 quote dates (2025-05-30 onward) and backfilling the rest
from ``quotes_v1`` is ~19 TB, because that product is ~107 GB/day.  The bounded
reverse query below returns the single last NBBO at or before an instant in
about 200 bytes:

    /v3/quotes/{contract}?timestamp.lte=<ns>&order=desc&sort=timestamp&limit=1

Marking an open book is a small, *known* set of contracts -- tens, not the
8,500-contract cross-section a chain build needs -- so the whole evaluation
window costs tens of thousands of requests rather than terabytes.

Two measured traps shaped the code:

* **Throughput saturates near 28 requests/second** and is flat from 16 to 64
  workers, so the ceiling is server-side rather than bandwidth or concurrency.
  ``prefetch`` therefore parallelises to hide latency, not to go faster than
  that, and the cache is what actually makes a repeated run cheap.
* HTTP/2 multiplexing against this host dies under concurrency
  (``PROTOCOL_ERROR``, every request in the batch failing at once) while single
  requests succeed.  ``urllib`` speaks HTTP/1.1 only, which is why this uses it
  rather than a modern client -- an incidental property worth stating, since
  "upgrade the HTTP client" would reintroduce the bug.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

__all__ = [
    "MarkQuotes",
    "NullMarkQuotes",
    "MassiveMarkQuotes",
    "MarkQuote",
    "MARK_QUOTE_SOURCE_VERSION",
    "occ_ticker",
]

MARK_QUOTE_SOURCE_VERSION = "massive_mark_quotes.v1"

#: Requests in flight.  Above ~16 the measured rate stops improving, so this is
#: chosen to reach the server-side ceiling without queueing behind it.
_DEFAULT_WORKERS = 16


class MarkQuote:
    """One NBBO, with the lag between the print and the instant asked for."""

    __slots__ = ("bid", "ask", "quote_time", "age_seconds")

    def __init__(self, bid: float, ask: float, quote_time: datetime, age_seconds: float) -> None:
        self.bid = bid
        self.ask = ask
        self.quote_time = quote_time
        self.age_seconds = age_seconds

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class MarkQuotes(Protocol):
    """What ``MarketResolver`` needs from a mark-price source."""

    def prefetch(self, contract_ids: Iterable[str], at: datetime) -> None:
        """Warm whatever ``quote`` will be asked for at this instant."""

    def quote(self, contract_id: str, at: datetime) -> MarkQuote | None:
        """The last NBBO at or before ``at``, or ``None`` if there is none."""


class NullMarkQuotes:
    """The default: no out-of-chain marking at all.

    Kept as a real object rather than an ``Optional`` so that the resolver has
    one code path.  A run configured without a mark source behaves exactly as it
    did before this module existed, which is what makes the two comparable.
    """

    #: No source, and it says so in the field the fingerprint hashes.
    source_version: str | None = None

    def prefetch(self, contract_ids: Iterable[str], at: datetime) -> None:
        return None

    def quote(self, contract_id: str, at: datetime) -> MarkQuote | None:
        return None


class MassiveMarkQuotes:
    """Bounded reverse NBBO lookups, cached on disk.

    The cache is append-only JSONL keyed by ``(contract_id, instant)``, and it
    stores *misses* as well as hits.  Without that, a contract with no quote
    before the instant -- which is a real and permanent fact about the tape --
    would be re-requested on every replay of the same window, and the reruns
    that matter here are cost sweeps over an identical grid.
    """

    #: What this source is, for ``EnvConfig.mark_quote_source`` and the
    #: manifest.  An attribute rather than a literal at each use site because
    #: the whole point of the field is that the config's claim and the object
    #: actually handed to the environment must be the same string; two literals
    #: can drift, and the drift is exactly the bug this closes.
    source_version: str | None = MARK_QUOTE_SOURCE_VERSION

    def __init__(
        self,
        *,
        cache_path: Path,
        api_key: str | None = None,
        base_url: str = "https://api.massive.com",
        timeout_seconds: float = 30.0,
        workers: int = _DEFAULT_WORKERS,
        offline: bool = False,
    ) -> None:
        self._cache_path = Path(cache_path)
        self._api_key = api_key or os.getenv("MASSIVE_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        self._workers = max(1, int(workers))
        # ``offline`` replays the cache and refuses to reach the network.  It is
        # what makes a cached run reproducible on a compute node with no egress,
        # and what keeps the tests from needing a key.
        self._offline = bool(offline)
        self._entries: dict[str, dict[str, Any] | None] = {}
        self._lock = threading.Lock()
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._first_error: str | None = None
        self._load_cache()

    # -- statistics ------------------------------------------------------

    @property
    def stats(self) -> Mapping[str, Any]:
        """Provenance for the manifest: how much of the book was really marked."""
        return {
            "cached_entries": len(self._entries),
            "requests": self._requests,
            "quotes_returned": self._hits,
            "no_quote_before_instant": self._misses,
            "request_errors": self._errors,
            "first_request_error": self._first_error or "",
            # Recorded because it changes what a zero means: offline, zero
            # requests is the cache doing its job, and online it is a book that
            # never left the chain.
            "offline": int(self._offline),
        }

    # -- interface -------------------------------------------------------

    def prefetch(self, contract_ids: Iterable[str], at: datetime) -> None:
        nanos = _epoch_nanos(at)
        wanted = [c for c in dict.fromkeys(contract_ids) if _key(c, nanos) not in self._entries]
        if not wanted or self._offline:
            return
        # One pool per call rather than a long-lived one: marking happens twice
        # a day against a handful of contracts, so pool creation is noise next
        # to the requests, and a pool that outlives the call would have to be
        # shut down by a caller that currently owns no lifecycle.
        with ThreadPoolExecutor(max_workers=min(self._workers, len(wanted))) as pool:
            list(pool.map(lambda c: self._fetch_into_cache(c, nanos), wanted))

    def quote(self, contract_id: str, at: datetime) -> MarkQuote | None:
        nanos = _epoch_nanos(at)
        key = _key(contract_id, nanos)
        if key not in self._entries:
            if self._offline:
                return None
            self._fetch_into_cache(contract_id, nanos)
        payload = self._entries.get(key)
        if not payload:
            return None
        quote_time = datetime.fromtimestamp(payload["t"] / 1e9, tz=timezone.utc)
        return MarkQuote(
            bid=float(payload["bid"]),
            ask=float(payload["ask"]),
            quote_time=quote_time,
            age_seconds=(nanos - int(payload["t"])) / 1e9,
        )

    # -- pieces ----------------------------------------------------------

    def _fetch_into_cache(self, contract_id: str, nanos: int) -> None:
        key = _key(contract_id, nanos)
        try:
            payload = self._request(contract_id, nanos)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            # A failed request is *not* recorded as a miss.  Caching it would
            # bake a transient outage into the ledger as a permanent absence of
            # market data, and the next run would never retry it.
            with self._lock:
                self._errors += 1
                # Keep the first one.  Without it a systematic failure -- every
                # id malformed, a dead key -- reports as a bare count that is
                # indistinguishable from a book that never left the chain, and
                # a run with 112/112 errors reads as clean in the manifest.
                if self._first_error is None:
                    self._first_error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            self._entries[key] = payload
            if payload:
                self._hits += 1
            else:
                self._misses += 1
            self._append(key, payload)

    def _request(self, contract_id: str, nanos: int) -> dict[str, Any] | None:
        if not self._api_key:
            raise ValueError("MASSIVE_API_KEY is required for live mark quotes")
        query = urlencode(
            {
                "timestamp.lte": str(nanos),
                "order": "desc",
                "sort": "timestamp",
                "limit": "1",
                "apiKey": self._api_key,
            }
        )
        url = f"{self._base_url}/v3/quotes/{quote(occ_ticker(contract_id), safe='')}?{query}"
        request = Request(url, headers={"Accept": "application/json"})
        with self._lock:
            self._requests += 1
        with urlopen(request, timeout=self._timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        results = body.get("results") or []
        if not results:
            return None
        row = results[0]
        bid = row.get("bid_price")
        ask = row.get("ask_price")
        # A one-sided or crossed book has no mid.  Recording it as a miss rather
        # than half a quote keeps the "no usable price" case single-valued, so
        # the resolver's fallback is reached for the same reason every time.
        if bid is None or ask is None or float(ask) <= float(bid):
            return None
        stamp = row.get("sip_timestamp") or row.get("participant_timestamp")
        if stamp is None:
            return None
        return {"bid": float(bid), "ask": float(ask), "t": int(stamp)}

    def _load_cache(self) -> None:
        if not self._cache_path.is_file():
            return
        with self._cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-append leaves one torn final line.  The
                    # prefix is still good, so skipping beats refusing to start.
                    continue
                self._entries[str(record["k"])] = record.get("v")

    def _append(self, key: str, payload: dict[str, Any] | None) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"k": key, "v": payload}, separators=(",", ":")) + "\n")


def occ_ticker(contract_id: str) -> str:
    """Translate the environment's contract id into Massive's OCC ticker.

    The chain carries legs as ``AAPL:2024-09-20:212.5:P``; Massive addresses
    them as ``O:AAPL240920P00212500``.  Passing the former straight through --
    which is what this module used to do -- returns ``400 Ticker was incorrectly
    formatted`` on *every* request, so the whole book silently fell back to
    being carried at entry price: the exact failure this module exists to stop.

    Ids already in Massive's form pass through, because ``env/chain.py`` reads
    ``contract_id`` or ``option_ticker``, and the latter is already an OCC
    symbol on some partitions.
    """
    if contract_id.startswith("O:"):
        return contract_id
    parts = contract_id.split(":")
    if len(parts) != 4:
        raise ValueError(f"cannot form an OCC ticker from {contract_id!r}")
    underlying, expiry, strike, right = parts
    code = right.strip().upper()[:1]
    if code not in ("C", "P"):
        raise ValueError(f"cannot form an OCC ticker from {contract_id!r}")
    day = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    # Strikes are thousandths in OCC, and ``round`` rather than ``int`` because
    # a strike like 212.5 arrives as a float whose product with 1000 can land at
    # 212499.99999999997 -- truncating that shifts the strike by a tenth of a
    # cent and addresses a contract that does not exist.
    thousandths = round(float(strike) * 1000)
    return f"O:{underlying.strip().upper()}{day}{code}{thousandths:08d}"


def _key(contract_id: str, nanos: int) -> str:
    return f"{contract_id}@{nanos}"


def _epoch_nanos(at: datetime) -> int:
    if at.tzinfo is None:
        raise ValueError("mark instants must be timezone-aware")
    return int(at.timestamp() * 1_000_000_000)
