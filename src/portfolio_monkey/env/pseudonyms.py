"""De-identify the observation, so the teacher cannot recognise the window.

Remedies A (filter the corpus) and B (ask the model not to use what it knows)
both act after the fact or on trust. This one removes the *cue*. A model that
cannot tell which name it is looking at, or when, has nothing to recall --
which is why this is the only one of the three that does not depend on the
model's cooperation.

**It is a view, not a change to the environment.** The book, the ledger, the
resolvers and every dataset keep real tickers and real dates throughout. The
substitution happens at the two points where text crosses to and from the
policy: outbound on the rendered observation, inbound on the completion. So the
resolver still computes which expiry a ``8_30`` request lands on, while the
policy that asked for it never saw a date -- the asymmetry that makes this
safe to train on.

**Why the labels are redrawn every episode.** A fixed map is broken on arrival:
the ``M`` row carries iv, rv, skew and flow, and those are a fingerprint. One
name sitting at 48% iv with a +7 pt vol premium is recognisable to anyone who
traded 2024, and a model only has to link it once for a stable label to carry
that identification across the whole corpus. Redrawing per episode does not
make a single episode harder to de-anonymise -- it stops a successful
de-anonymisation from being *reusable*, which is the property that matters when
the corpus is the training set. Carried positions are relabelled into the new
draw, per the ruling of 2026-09-23.

**What this cannot hide, stated because the guard below cannot catch it.**
The alias table covers the ten names of the universe. News clauses are the
source's own words and routinely name *other* entities -- a supplier, a rival,
a politician, a strike -- and those date a step just as precisely as a
timestamp does. ``residual`` counts them rather than pretending they are gone.
Anyone reading a run generated through this layer should read that count first.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence

#: Every spelling of a universe name that has actually turned up in the news
#: corpus, plus the products that identify a company as surely as its name.
#: Over-masking is cheap here and under-masking is not, so a term goes in
#: whenever it is more identifying than it is ambiguous.
ALIASES: Mapping[str, tuple[str, ...]] = {
    "AAPL": ("AAPL", "Apple", "Apple Inc", "iPhone", "iPad", "Mac", "App Store"),
    "AMZN": ("AMZN", "Amazon", "Amazon.com", "AWS", "Amazon Web Services"),
    "GOOGL": ("GOOGL", "GOOG", "Google", "Alphabet", "YouTube", "Waymo"),
    "META": ("META", "Meta", "Meta Platforms", "Facebook", "Instagram", "WhatsApp"),
    "MSFT": ("MSFT", "Microsoft", "Azure", "Xbox", "Copilot"),
    "MU": ("MU", "Micron", "Micron Technology"),
    "NVDA": ("NVDA", "Nvidia", "NVIDIA", "GeForce", "DGX", "CUDA"),
    "PLTR": ("PLTR", "Palantir", "Palantir Technologies"),
    "TSLA": ("TSLA", "Tesla", "Musk", "Elon Musk", "Cybertruck", "Model Y"),
    # The index slot. ``SPY`` files nothing itself, so the identifying strings
    # are the index's, not the ETF's -- see the standing note that SPX news
    # counts as SPY context.
    "SPY": ("SPY", "SPX", "S&P 500", "S&P500", "S&P", "Standard & Poor's"),
}

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October"
    "|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

#: Applied in order. Each pattern removes a way of writing *when*, and the
#: order matters: the fuller forms have to go before the bare year, or
#: ``October 7, 2024`` degrades to ``October 7, <y>`` instead of vanishing.
_TIME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bep_(\d{4})_(\d{2})\b"), "ep"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<d>"),
    (re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"), "<d>"),
    (re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{4}}\b"), "<d>"),
    (re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"), "<d>"),
    (re.compile(rf"\b(?:{_MONTHS})\b"), "<d>"),
    (re.compile(r"\b(?:Q[1-4]|[1-4]Q)\s*(?:FY)?\s*\d{2,4}\b"), "<d>"),
    (re.compile(r"\b(?:19|20)\d{2}\b"), "<y>"),
)

#: Tokens that look like a proper noun but carry no date or identity. Kept
#: small on purpose: the point of ``residual`` is to over-report, so that the
#: number is an upper bound someone has to argue *down* rather than a
#: reassuring figure they have to argue up.
_RESIDUAL_STOPWORDS = frozenset(
    {
        "AM", "PM", "CEO", "CFO", "COO", "CTO", "AI", "EV", "IPO", "ETF", "US",
        "U.S.", "The", "A", "An", "And", "But", "For", "In", "On", "At", "It",
        "This", "That", "These", "Those", "He", "She", "They", "We", "I",
        "OK", "EPI", "T", "M", "P", "N", "R", "MKT", "ACCT", "POS", "RES",
    }
)

#: Multi-word names are joined by spaces and tabs only, never by a newline.
#: ``\s`` would span the line break and report ``PM\nEPI`` as an entity, which
#: inflates a count whose whole job is to be believable.
_PROPER = re.compile(r"\b[A-Z][A-Za-z&.']{1,}(?:[ \t]+[A-Z][A-Za-z&.']{1,})*\b")


def third_friday(year: int, month: int) -> date:
    """The standard monthly expiry: the third Friday of the month."""
    first = date(year, month, 1)
    # ``weekday()`` is Monday 0 .. Friday 4, so this walks forward to the first
    # Friday and then two weeks on.
    return first.replace(day=1 + ((4 - first.weekday()) % 7) + 14)


def dte_to_monthly(day: date) -> int:
    """Calendar days from ``day`` to the next standard monthly expiry.

    Rolls to the following month once this month's expiry has passed, which is
    what makes the series a sawtooth: it counts down to 0 on the third Friday
    and jumps back to roughly a month the next session. That shape carries the
    two things a date was being used for -- where in the option cycle this step
    sits, and how much time the tradeable tenors have -- and carries nothing
    about *which* cycle it is.
    """
    expiry = third_friday(day.year, day.month)
    if expiry < day:
        year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        expiry = third_friday(year, month)
    return (expiry - day).days


@dataclass(frozen=True)
class Pseudonyms:
    """A per-episode bijection between real tickers and opaque labels.

    Labels are always ``U01``..``U10``: the *set* is constant so the policy can
    learn the encoding, while the *assignment* is redrawn per episode so no
    single identification stays useful. They are three characters and begin
    with a letter that is not a block letter (``M P A N T R``), not a verb
    (``O C X H``), and not a family or orientation code, so a label can never
    be misread as grammar.
    """

    episode_id: str
    to_label: Mapping[str, str]
    to_ticker: Mapping[str, str]

    @classmethod
    def draw(cls, episode_id: str, universe: Sequence[str], seed: int = 0) -> "Pseudonyms":
        """Deterministic from ``(seed, episode_id)``, so a run is auditable.

        Seeding off the episode id rather than a counter means the draw does
        not depend on how many episodes ran before it: replaying one episode in
        isolation reproduces the same map, which is what makes a disagreement
        between two runs a real finding rather than a bookkeeping artifact.
        """
        order = list(universe)
        random.Random(f"{seed}:{episode_id}").shuffle(order)
        to_label = {ticker: f"U{i + 1:02d}" for i, ticker in enumerate(order)}
        return cls(
            episode_id=episode_id,
            to_label=to_label,
            to_ticker={label: ticker for ticker, label in to_label.items()},
        )

    def sort_key(self, ticker: str) -> str:
        """Order rows by label, not by ticker.

        Relabelling alone is not enough while the rows stay in universe order:
        the label would be scrambled but the *position* would still spell out
        the real sequence, and one fixed row order across the corpus is as
        reusable an identification as one fixed label.
        """
        return self.to_label.get(ticker, "U99")

    # -- outbound --------------------------------------------------------

    def mask(self, text: str) -> str:
        """Rewrite one rendered block for the policy.

        Names first, then time. Longest alias first, so ``Meta Platforms``
        cannot be half-consumed by ``Meta`` and leave ``U04 Platforms``.
        """
        pairs: list[tuple[str, str]] = []
        for ticker, aliases in ALIASES.items():
            label = self.to_label.get(ticker)
            if label is None:
                continue
            pairs.extend((alias, label) for alias in aliases)
        for alias, label in sorted(pairs, key=lambda kv: -len(kv[0])):
            text = re.sub(rf"\b{re.escape(alias)}\b", label, text, flags=re.IGNORECASE)
        for pattern, replacement in _TIME_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def mask_clause(self, clause: str) -> str:
        """``mask`` plus: every remaining proper noun becomes ``<ent>``.

        Ruling of 2026-09-23, option (a). Kept separate from ``mask`` and
        applied **only to the news clause**, because the clause is the sole
        free prose in the observation and a proper-noun sweep let loose on the
        whole text would eat the grammar -- ``EPI``, ``OK``, ``E_LIMIT``,
        ``FORCE_CLOSE_DTE`` are all capitalised and all load-bearing.

        **One opaque token for every entity, not ``<ent1>``/``<ent2>``.**
        Numbering them would preserve that two different companies were named,
        which is worth something inside a single clause -- but a numbering
        stable across the corpus is precisely a new identifier, and the model
        only has to solve ``<ent7> = the one that keeps appearing with the
        chipmaker`` once. The within-clause distinction is not worth handing
        back a re-identification channel.

        **What survives, and it is not small.** Sector language is not a proper
        noun: "memory chips", "dockworkers' strike", "the election" all read
        through untouched, and each of them dates a step. Measured on 1,011
        real ``N`` rows, only 21.7% were free of a reachable proper noun before
        this sweep; after it the named entities are gone but the prose is not
        neutral. Treat this as reducing the leak, never as closing it.
        """
        masked = self.mask(clause)
        labels = set(self.to_ticker)

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token in _RESIDUAL_STOPWORDS or token in labels:
                return token
            return "<ent>"

        masked = _PROPER.sub(replace, masked)
        # ``<ent> <ent> <ent>`` from a three-word name says how long the name
        # was, which is a weak fingerprint and pure token cost besides.
        return re.sub(r"(?:<ent>)(?:\s+<ent>)+", "<ent>", masked)

    # -- inbound ---------------------------------------------------------

    def unmask(self, text: str) -> str:
        """Turn a completion's labels back into tickers for the resolver.

        This is the half that lets the resolver do arithmetic the policy could
        not: the order arrives naming ``U03`` and a tenor bucket, and leaves
        here naming a real underlying, which the contract resolver then prices
        against a real chain on a real date.
        """
        for label, ticker in self.to_ticker.items():
            text = re.sub(rf"\b{re.escape(label)}\b", ticker, text)
        return text

    # -- guard -----------------------------------------------------------

    def residual(self, text: str) -> dict[str, list[str]]:
        """What still identifies, split by whether this layer claims to fix it.

        ``must_be_empty`` is the layer's own contract: a universe ticker, a
        universe alias, or any way of writing a date. Anything here is a defect
        and the caller should refuse the text.

        ``unreachable`` is everything else that reads like a proper noun -- a
        non-universe company, a person, a country. The alias table cannot cover
        these and this class does not claim to. They are returned so the count
        can be reported next to the corpus instead of discovered later.
        """
        must: list[str] = []
        for ticker, aliases in ALIASES.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
                    must.append(alias)
        for pattern, _ in _TIME_PATTERNS:
            must.extend(m.group(0) for m in pattern.finditer(text))

        labels = set(self.to_ticker)
        unreachable = [
            token
            for token in _PROPER.findall(text)
            if token not in _RESIDUAL_STOPWORDS
            and token not in labels
            and not token.startswith("U0")
        ]
        return {"must_be_empty": must, "unreachable": sorted(set(unreachable))}


def redraw_for(episode_ids: Iterable[str], universe: Sequence[str], seed: int = 0) -> dict[str, Pseudonyms]:
    """One draw per episode id, which is the unit the ruling names."""
    return {eid: Pseudonyms.draw(eid, universe, seed) for eid in episode_ids}


def anonymize(text: str, pseudo: Pseudonyms, *, trade_date: date | None = None) -> str:
    """Rewrite a rendered block row by row, respecting each row's schema.

    Applied to the finished text rather than threaded through the six renderers
    that build it. That is a deliberate trade. Row-wise rewriting depends on
    the rendered format, which is more brittle than editing each renderer --
    but the rows are self-labelling by design (the ``system_block`` calls the
    leading block letter "the only cue that says which schema line to read the
    row against"), and doing it in one place means a block added later cannot
    quietly skip the layer. A renderer-by-renderer version fails silently; this
    one fails in a single function that ``residual`` can then check whole.

    ``trade_date`` supplies the sawtooth for the ``T`` header. Without it the
    date is masked to ``<d>``, which is safe but throws away the time-series
    information the instruction asked to keep.
    """
    out: list[str] = []
    market: list[str] = []

    def flush() -> None:
        # ``M`` rows are emitted in universe order. Relabelling alone would
        # scramble the label while the row *position* still spelled out the
        # real sequence, so the rows are reordered by label before they land.
        out.extend(sorted(market, key=lambda r: r.split()[1]))
        market.clear()

    for line in text.splitlines():
        fields = line.split()
        if not fields:
            flush()
            out.append(line)
            continue
        tag = fields[0]

        if tag == "T" and len(fields) >= 3:
            flush()
            stamp = f"m{dte_to_monthly(trade_date)}" if trade_date else "<d>"
            out.append(" ".join([fields[0], fields[1], stamp, *fields[3:]]))
        elif tag == "M" and len(fields) >= 2:
            fields[1] = pseudo.to_label.get(fields[1], fields[1])
            market.append(" ".join(fields))
        elif tag == "N" and len(fields) >= 5:
            flush()
            head = [
                fields[0],
                pseudo.to_label.get(fields[1], fields[1]),
                *fields[2:5],
            ]
            clause = " ".join(fields[5:])
            out.append(" ".join(head + ([pseudo.mask_clause(clause)] if clause else [])))
        elif tag == "P" and len(fields) >= 3:
            flush()
            fields[2] = pseudo.to_label.get(fields[2], fields[2])
            out.append(pseudo.mask(" ".join(fields)))
        elif tag == "spot":
            # Dropped per the ruling of 2026-09-23: a price level names a stock
            # in one glance, and orders are delta-coordinate so nothing
            # downstream can act on it.
            #
            # The renderer that emitted this row was deleted the same day --
            # it read ``EpisodeContext.metadata["spot"]``, which nothing ever
            # populated, so no observation has ever carried a level. The branch
            # stays because this layer is the last thing between a renderer and
            # the policy, and it should refuse a price line whether or not one
            # exists today. The ruling's other half needs no code: the ``M``
            # row's leading ``ret`` cell already is the session return per name.
            flush()
        else:
            flush()
            out.append(pseudo.mask(line))
    flush()
    return "\n".join(out)


def session_return_row(returns: Mapping[str, float], pseudo: Pseudonyms) -> str:
    """``ret U07 -15 U08 -40 ...`` -- last session's move per name, as a row.

    **Not wired into the observation, and the reason is a finding rather than
    an omission.** This was written to satisfy the second half of the ruling of
    2026-09-23 -- "drop spot and replace it with the tickers returns from the
    last trading session to current" -- and on reading the renderer the
    replacement turned out to be already present: the ``M`` row's leading
    ``ret`` cell is ``spot_return_step``, the session return for that name, in
    these same 0.1% units. Emitting this line in the episode header would put
    the same ten numbers in the prompt twice, since step 0's ``M`` block
    follows the header immediately in the same context.

    Kept because the function is the honest way to *show* that redundancy, and
    because the one case where a header row would not be redundant is real but
    unbuilt: a header emitted without a following step. Do not wire it without
    a ruling; a duplicated field teaches the policy that two encodings exist.
    """
    cells = sorted(
        (pseudo.to_label[t], value) for t, value in returns.items() if t in pseudo.to_label
    )
    return "ret " + " ".join(f"{label} {round(value * 1000):+d}" for label, value in cells)
