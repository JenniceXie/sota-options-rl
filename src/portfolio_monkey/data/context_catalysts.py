"""Turn the free-text catalyst corpus into one wire-encodable row per name-day.

Three gates, applied in order, each of which records *why* it dropped a row so
the yield can be audited rather than asserted:

1. **Label standardisation.** ``catalyst_type`` is the only unenumerated field
   the summariser emits -- 933 distinct values over 4,080 rows, 590 of them
   occurring exactly once. The other two encoded fields (``direction`` and
   ``time_horizon``) are already closed sets. :func:`normalize_catalyst`
   collapses the free text onto ten codes so it costs one wire token and is
   comparable across names.

2. **On-name relevance.** ``related_symbols`` tagging is loose: an NVDA-tagged
   document on 2025-06-06 summarises *"Aehr Test Systems shares rose 15.9%"* and
   another summarises a CoreWeave IPO. Under an append-only prompt a summary
   inserted at step *t* is paid for at every later step of the episode, so an
   off-name row is worse than an empty one. :func:`mentions` requires the name to
   appear in the summary text itself, not merely in the tag list.

3. **Materiality.** A row whose direction and horizon are both ``unknown``
   carries no tradeable content. These rows are *not* caught by the claim
   validator -- 87% of them pass it, against 58% of the rest, because a summary
   that asserts nothing checkable cannot fail a grounding check. The two filters
   are complementary and the validator alone keeps exactly the wrong ones.

This module reads the corpus as built. It does not re-generate summaries and
makes no model calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

#: The closed catalyst taxonomy, in wire order. The description is what the
#: ``SYS`` block declares once per episode; the code is what each row carries.
CATALYST_CODES: tuple[tuple[str, str], ...] = (
    ("ea", "earnings, results, guidance"),
    ("dv", "dividend, buyback, capital return"),
    ("ma", "merger, acquisition, divestiture"),
    ("lg", "lawsuit, investigation, regulatory action"),
    ("gv", "governance, board, leadership, ownership"),
    ("pr", "product, operations, partnership, trial"),
    ("an", "analyst rating, price target, valuation"),
    ("cp", "debt or equity issuance, financing, split"),
    ("mo", "macro, policy, sector-wide forecast"),
    ("ot", "other"),
)

#: Substring rules applied to the normalised label, **first match wins**. Order
#: is load-bearing and is asserted in the tests: ``market_forecast`` must reach
#: ``mo`` before ``forecast`` sends it to ``an``, and ``share_buyback`` must
#: reach ``dv`` before ``share`` means anything else.
_CATALYST_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ea", ("earning", "quarterly_result", "financial_result", "guidance", "eps")),
    ("mo", ("macro", "policy", "tariff", "inflation", "fed", "interest_rate",
            "market_forecast", "market_growth", "market_sentiment",
            "industry_forecast", "economic", "sector", "volatility",
            "politic", "election", "geopolit")),
    ("dv", ("dividend", "buyback", "repurchase", "capital_return", "distribution")),
    ("ma", ("merger", "acquisition", "acqui", "divestiture", "takeover", "spin_off",
            "spinoff", "asset_sale", "disposition", "deal")),
    ("lg", ("lawsuit", "litigation", "legal", "class_action", "investigation",
            "regulat", "settlement", "antitrust", "probe", "subpoena", "fine",
            "compliance", "sanction")),
    ("gv", ("governance", "board", "leadership", "executive", "ceo", "cfo",
            "shareholder_vote", "proxy", "annual_meeting", "agm", "ownership",
            "control", "insider", "appointment", "resignation", "succession",
            "auditor", "reincorporation",
            # 13F-style positioning news: a fund taking or cutting a stake is
            # ownership, and it is the largest identifiable family left in the
            # `ot` residual once document types are removed.
            "hedge_fund", "institutional", "stake", "portfolio", "buffett",
            "13f", "divestment")),
    ("cp", ("debt", "issuance", "offering", "financing", "funding", "note",
            "credit", "loan", "stock_split", "reverse_split", "convertible",
            "investment", "capital_raise", "prospectus", "ipo", "listing",
            "delisting", "tender_offer", "option_exercise", "warrant")),
    ("pr", ("product", "launch", "clinical", "trial", "partnership", "contract",
            "project", "technology", "expansion", "facility", "supply",
            "production", "award", "collaboration", "customer", "licensing",
            "agreement")),
    ("an", ("analyst", "recommendation", "rating", "price_target", "valuation",
            "upgrade", "downgrade", "forecast", "outlook", "estimate",
            "analysis", "growth")),
)

#: Labels that name a *document type* rather than a catalyst. These are the head
#: of the ``ot`` residual -- ``filing`` (79), ``risk_factors`` (41), ``news``
#: (39), ``conference`` (25), ``prospectus``, ``presentation``, ``article`` --
#: and they mean the summariser answered "what kind of document is this" instead
#: of "what will move the stock". A row like that is not evidence that no
#: catalyst exists, only that none was identified; with one 40-token slot per
#: name-day the two are worth the same, so it is suppressed when the direction
#: is also non-directional.
DOCUMENT_TYPE_KEYWORDS: tuple[str, ...] = (
    "filing",
    "news",
    "article",
    "press_release",
    "presentation",
    "conference",
    "report",
    "document",
    "disclosure",
    "risk_factor",
    "forward_looking",
    "reference",
    "summary",
    "overview",
    "update",
    "commentary",
    "event",
    "general",
    "other",
    "misc",
)

#: Wire codes for the three fields that are already closed sets upstream.
DIRECTION_CODES: Mapping[str, str] = {
    "bullish": "+",
    "bearish": "-",
    "neutral": "0",
    "mixed": "?",
    "unknown": "?",
}
HORIZON_CODES: Mapping[str, str] = {
    "intraday": "0",
    "1_5d": "1",
    "6_21d": "2",
    "gt_21d": "3",
    "unknown": "?",
}
#: Prose the summary must contain for the row to count as on-name.
#:
#: This is deliberately **not** on ``UniverseSpec``. ``context_symbols()`` there
#: answers "which provider tags mean this name"; this answers "which words in
#: English prose mean this name". The second is a lexicon for an offline
#: retrieval filter and has no bearing on environment dynamics, so it must not
#: enter ``EnvConfig.fingerprint()`` and invalidate checkpoints when a synonym is
#: added.
#:
#: Tickers are matched case-sensitively and company names case-insensitively.
#: That asymmetry is the reason bare ``Meta`` is absent: matched
#: case-insensitively it fires on the English word ``meta``, so Meta Platforms is
#: reachable only through the ticker or an unambiguous property name.
NAME_TICKERS: Mapping[str, tuple[str, ...]] = {
    "AAPL": ("AAPL",),
    "AMZN": ("AMZN",),
    "GOOGL": ("GOOGL", "GOOG"),
    "META": ("META", "FB"),
    "MSFT": ("MSFT",),
    "MU": ("MU",),
    "NVDA": ("NVDA",),
    "PLTR": ("PLTR",),
    "TSLA": ("TSLA",),
    "SPY": ("SPY", "SPX", "VOO", "IVV"),
}
NAME_PROSE: Mapping[str, tuple[str, ...]] = {
    "AAPL": ("Apple", "iPhone", "iPad"),
    "AMZN": ("Amazon", "AWS"),
    "GOOGL": ("Alphabet", "Google", "YouTube", "Waymo"),
    "META": ("Meta Platforms", "Facebook", "Instagram", "WhatsApp"),
    "MSFT": ("Microsoft", "Azure", "Xbox"),
    "MU": ("Micron",),
    "NVDA": ("Nvidia",),
    "PLTR": ("Palantir",),
    "TSLA": ("Tesla",),
    "SPY": ("S&P 500", "S&P500", "the S&P"),
}


@dataclass(frozen=True)
class Catalyst:
    """One selected, standardised catalyst for one name on one day."""

    name: str
    catalyst: str
    direction: str
    horizon: str
    clause: str

    def row(self) -> str:
        """The shipped wire row, block letter first, per state_space.md 6.5.

        There is no flow field. ``flow_relationship`` is written to
        ``features/catalyst_assessments``, not to the summaries this module
        reads, so the column was ``.`` on every row ever rendered and the three
        legend codes explaining it were paid for once per episode to describe a
        value that could not occur. Joining assessments to recover it is
        possible but not worth it: the field is ``absent`` on 87% of the rows
        that do carry it.
        """
        return (
            f"N {self.name} {self.catalyst} {self.direction} "
            f"{self.horizon} {self.clause}"
        )


def normalize_catalyst(raw: object) -> str:
    """Map a free-text ``catalyst_type`` onto one of :data:`CATALYST_CODES`.

    Unrecognised labels become ``ot`` rather than raising. The residual rate is
    a reported statistic, not an error: a taxonomy that silently absorbs
    everything and a taxonomy that rejects everything are equally useless, and
    the only way to tell them apart is to count.
    """

    label = re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")
    if not label or label in ("unknown", "none", "no_selected_context"):
        return "ot"
    for code, keywords in _CATALYST_RULES:
        if any(keyword in label for keyword in keywords):
            return code
    return "ot"


def names_a_catalyst(raw: object) -> bool:
    """Did the summariser name an event, or just the kind of document it read?

    Checked only when :func:`normalize_catalyst` already fell through to ``ot``,
    so a label that reaches a real code is never second-guessed: ``8-K filing``
    is ``ot`` and a document type, while ``earnings`` is ``ea`` and is left
    alone even though ``report`` is a document-type keyword.
    """

    label = re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")
    if not label:
        return False
    if normalize_catalyst(raw) != "ot":
        return True
    return not any(keyword in label for keyword in DOCUMENT_TYPE_KEYWORDS)


def _word_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


_TICKER_PATTERNS = {
    name: tuple(_word_pattern(t) for t in terms)
    for name, terms in NAME_TICKERS.items()
}
_PROSE_PATTERNS = {
    name: tuple(re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in terms)
    for name, terms in NAME_PROSE.items()
}


def mentions(text: object, name: str) -> bool:
    """Does ``text`` actually talk about ``name``?

    ``SPY`` is the one slot where a literal mention is the wrong test for part of
    the corpus -- macro and policy news moves the index without naming it -- so
    callers pass such rows through :func:`select_catalyst`, which exempts
    ``mo``-coded catalysts for the index slot only.
    """

    body = str(text or "")
    if any(pattern.search(body) for pattern in _TICKER_PATTERNS.get(name, ())):
        return True
    return any(pattern.search(body) for pattern in _PROSE_PATTERNS.get(name, ()))


#: The index slot, which is the only name allowed to carry an unnamed catalyst.
INDEX_NAME = "SPY"

#: Horizons worth a wire row. ``gt_21d`` survives because a 182-day tenor cap
#: still leaves room for it; ``unknown`` does not.
MATERIAL_HORIZONS = frozenset({"intraday", "1_5d", "6_21d", "gt_21d"})

#: ``unmapped_*`` fires when the summary contains a number the model did not
#: declare a structured claim for. It is a *coverage* test over the claim
#: record, not a grounding test over the prose, and measured against the source
#: documents 94.6% of the numbers it flags are present in the source anyway
#: (84.3% as the verbatim token, a further 10.3% as the bare numeral; only 5.5%
#: are absent). See ``scripts/analysis/audit_context_quarantine_recovery.py``.
RECOVERABLE_CODE_PREFIXES = ("unmapped_",)

#: The wider encoding family: the claim record is malformed, unparseable or
#: incomplete. 42.1% of quarantined rows carry *only* these. Their prose is
#: plausibly fine but that has **not** been verified the way ``unmapped_*`` has,
#: which is why admitting them is opt-in rather than the default.
ENCODING_CODE_PREFIXES = (
    "unmapped_",
    "unparseable_",
    "invalid_",
    "missing_",
)
ENCODING_CODE_SUFFIXES = ("_source_text_missing", "_summary_text_missing")
ENCODING_CODES = frozenset({"structured_claims_missing", "numeric_role_unit_mismatch"})

#: What to do with a quarantined row. The pipeline as built is ``"strict"``.
QUARANTINE_POLICIES = ("strict", "recoverable", "encoding", "off")


def _violation_codes(row: Mapping[str, object]) -> list[str]:
    results = row.get("validation_results")
    if not isinstance(results, Mapping):
        return []
    violations = results.get("validation_violations")
    if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes)):
        return []
    return [
        str(v.get("code") or "")
        for v in violations
        if isinstance(v, Mapping)
    ]


def _is_encoding_code(code: str) -> bool:
    return (
        code.startswith(ENCODING_CODE_PREFIXES)
        or code.endswith(ENCODING_CODE_SUFFIXES)
        or code in ENCODING_CODES
    )


def admits(row: Mapping[str, object], policy: str = "recoverable") -> bool:
    """Does ``policy`` let this row past the claim validator?

    ``strict`` reproduces the pipeline: any violation discards the summary.
    ``recoverable`` additionally admits rows whose violations are all
    ``unmapped_*``. ``encoding`` admits the whole encoding family. ``off``
    ignores validation.
    """

    if policy not in QUARANTINE_POLICIES:
        raise ValueError(f"unknown quarantine policy {policy!r}")
    if policy == "off":
        return True
    results = row.get("validation_results")
    status = results.get("validation_status") if isinstance(results, Mapping) else None
    if status != "quarantined":
        return True
    codes = _violation_codes(row)
    if not codes:
        return False
    if policy == "strict":
        return False
    if policy == "recoverable":
        return all(code.startswith(RECOVERABLE_CODE_PREFIXES) for code in codes)
    return all(_is_encoding_code(code) for code in codes)


def truncate_clause(text: str, limit: int) -> str:
    """Collapse whitespace and cut to ``limit`` characters on a word boundary.

    Cutting mid-word produces clauses like ``"investors seeking highe\u2026"``,
    which spends tokens on a fragment the policy cannot read. The result is
    never longer than ``limit``, so a caller can multiply by the prose rate and
    get a true upper bound on the row.
    """

    clause = " ".join(text.split())
    if len(clause) <= limit:
        return clause
    if limit <= 1:
        return "\u2026"[:limit]
    cut = clause[: limit - 1].rstrip().rfind(" ")
    if cut <= 0:
        # Not even the first word fits. A fragment of one word is worse than
        # nothing, so emit the marker alone rather than half a token.
        return "\u2026"
    return clause[:cut].rstrip(" ,;:.-") + "\u2026"


def screen(
    row: Mapping[str, object],
    name: str,
    *,
    quarantine_policy: str = "recoverable",
) -> tuple[bool, str]:
    """Should this summary reach the prompt? Returns ``(keep, reason)``.

    ``reason`` is the drop cause when ``keep`` is false and ``"kept"`` otherwise,
    so a caller can tabulate attrition per gate instead of observing only the
    survivors.
    """

    summary = str(row.get("summary") or "").strip()
    if not summary:
        return False, "empty_summary"

    if not admits(row, quarantine_policy):
        return False, "quarantined"

    catalyst = normalize_catalyst(row.get("catalyst_type"))
    direction = str(row.get("direction") or "unknown")
    horizon = str(row.get("time_horizon") or "unknown")

    directional = direction in ("bullish", "bearish")

    if direction in ("unknown", "mixed") and horizon not in MATERIAL_HORIZONS:
        return False, "immaterial_direction_and_horizon"

    if not directional and not names_a_catalyst(row.get("catalyst_type")):
        return False, "no_catalyst_named"

    if not mentions(summary, name):
        if not (name == INDEX_NAME and catalyst == "mo"):
            return False, "off_name"

    return True, "kept"


def select_catalyst(
    rows: Iterable[Mapping[str, object]],
    name: str,
    *,
    clause_chars: int = 256,
    quarantine_policy: str = "recoverable",
) -> tuple[Catalyst | None, dict[str, int]]:
    """Pick at most one catalyst for ``name`` and report per-gate attrition.

    The budget in ``state_space.md`` 6.6.1 is one summary per name per day, so
    this **selects** rather than aggregates. That is not only a token argument:
    the deterministic aggregation in ``catalyst_assessments`` collapses three
    NVDA summaries labelled ``bullish/gt_21d``, ``bullish/1_5d`` and
    ``mixed/gt_21d`` into ``mixed_catalysts / mixed / unknown``, turning three
    informative labels into one uninformative one. Selecting keeps whichever
    label survived the gates intact.

    Ties break toward the narrower horizon, then the directional over the
    neutral, then the longer clause -- an intraday call is worth more to a policy
    deciding today than a quarter-out one.

    ``clause_chars`` is 256 rather than 6.6.1's recommended ~160 because that
    recommendation was sized against a 10-row-per-day worst case that the corpus
    does not produce: measured fill is 38.3% of (date, session, name) slots.
    The earlier 12.3% figure in this docstring was measured against a stale
    63-date pilot cache in the production context root; the 246-date corpus
    lives in the shadow root and is 17,550 universe rows, so any budget derived
    from the old number understated the block by roughly 3x. 256 is still where
    the corpus stops -- raising the cap to 400 moves the 30-date total by 6
    tokens -- but at 256 the block busts a 32,768 cap on two months of the
    246-date window at P=12, so the caller passes 160 for monthly episodes.
    """

    attrition: dict[str, int] = {}
    kept: list[Mapping[str, object]] = []
    for row in rows:
        keep, reason = screen(row, name, quarantine_policy=quarantine_policy)
        attrition[reason] = attrition.get(reason, 0) + 1
        if keep:
            kept.append(row)
    if not kept:
        return None, attrition

    order = {"intraday": 0, "1_5d": 1, "6_21d": 2, "gt_21d": 3, "unknown": 4}
    best = min(
        kept,
        key=lambda r: (
            order.get(str(r.get("time_horizon")), 4),
            0 if str(r.get("direction")) in ("bullish", "bearish") else 1,
            -len(str(r.get("summary") or "")),
        ),
    )
    clause = truncate_clause(str(best.get("summary") or ""), clause_chars)
    return (
        Catalyst(
            name=name,
            catalyst=normalize_catalyst(best.get("catalyst_type")),
            direction=DIRECTION_CODES.get(str(best.get("direction")), "?"),
            horizon=HORIZON_CODES.get(str(best.get("time_horizon")), "?"),
            clause=clause,
        ),
        attrition,
    )


def sys_block() -> str:
    """The enum tables the ``SYS`` prompt declares once per episode."""

    codes = "  ".join(f"{code} {label}" for code, label in CATALYST_CODES)
    return (
        "N sym cat dir hz <clause>\n"
        f"cat  {codes}\n"
        "dir  + bullish  - bearish  0 neutral  ? mixed or unknown\n"
        "hz   0 intraday  1 1-5d  2 6-21d  3 gt-21d  ? unknown\n"
    )


def catalyst_code_set() -> Sequence[str]:
    return tuple(code for code, _ in CATALYST_CODES)
