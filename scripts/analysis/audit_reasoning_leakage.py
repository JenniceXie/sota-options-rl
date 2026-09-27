"""Audit a teacher's saved reasoning for future information.

The standing agreement is *"save all the reasoning of kimi-k3, and verify that
the reasoning does not use future information at the point in time."*  This
script is the verification half.  It answers a narrower question than its name
suggests, and the narrowing is the point:

**The input side is already proved, and this script does not re-prove it.**
Every feature row rendered into an observation carries its own
``available_time`` and is checked against the decision instant by
``statespace.assert_point_in_time``, which *raises* -- a run that finished is a
run in which no row postdated its own decision.  So the auditor reads the
manifest for a point-in-time traceback and reports the step count, rather than
recomputing a gate whose failure mode is a crash.

**What is left is the model's own memory**, and that is genuinely unguarded.
The teacher's weights were trained past 2024-11, so it can know that NVDA's
November print beat, and nothing in the environment stops it writing that down
in a trace stamped 2024-10-08.  That is the leak this script looks for.

Split into exposure, bias and impact, and reported per underlying, because a
leak is almost never uniform: one name with a memorable quarter contaminates a
handful of steps and a date-level rollup averages it into nothing.

**Every detector reports candidates, not verdicts, and the distinction is
load-bearing.**  There is no mechanical test for "the model knew".  A number
that first appears in a later observation may be arithmetic the model did on
the present one; a November date in an October trace is usually an expiry, not
a prophecy.  So each detector is tuned to be *cheap to overturn by reading*,
the flagged text is printed with its context, and the headline number is an
upper bound on exposure rather than a count of leaks.  Saying otherwise would
make the audit the thing it exists to prevent: a confident claim with no
mechanism under it.

Usage::

    python scripts/analysis/audit_reasoning_leakage.py \\
        --run s01=runs/sft_k3_w3_s01 --run s02=runs/sft_k3_w3_s02 \\
        --show 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

# --- what counts as a hit ----------------------------------------------------

#: Numbers below this many digits are not evidence of anything.  The wire is
#: dense with small integers -- every ``dte``, every delta coordinate, every
#: scaled feature -- so a 2-digit match between a trace and a future
#: observation happens by chance many times per step.  At 4 digits the
#: collision rate against a 43-step window is low enough that a hit is worth
#: reading, which is the only bar a candidate has to clear.
MIN_LEAK_DIGITS = 4

#: Words that make a nearby future date a *tenor*, not a claim.  An option
#: trace is inherently forward-looking: "roll the Nov 15 vertical" names a
#: contract that exists on the ladder today.  Without this split the detector
#: flags essentially every step and stops discriminating.
TENOR_WORDS = (
    "dte", "expiry", "expiration", "expire", "exp ", "tenor", "roll",
    "strike", "leg", "front", "back", "otm", "itm", "atm", "cycle",
)

#: Hindsight verbs.  Matched against the trace, then required to sit near an
#: underlying so that generic market prose does not fire.  This detector has the
#: worst precision of the three and is kept because it is the only one that can
#: catch a leak carrying no number and no date -- "we know MU disappoints here".
OUTCOME_WORDS = (
    r"beat", r"missed?", r"reported", r"announced", r"surged", r"plunged",
    r"rallied", r"crashed", r"tanked", r"soared", r"turn(?:ed|s) out",
    r"in hindsight", r"as we know", r"i know that", r"actually (?:went|fell|rose)",
    r"will (?:beat|miss|report|announce|drop|rally|surge|crash)",
    r"guidance (?:was|came)", r"came in (?:above|below|at)",
)
_OUTCOME_RE = re.compile(r"\b(" + "|".join(OUTCOME_WORDS) + r")\b", re.IGNORECASE)

#: The recall register: phrases that mark a claim as *remembered* rather than
#: read.  This is the only detector here with real precision, and the reason is
#: that it does not try to catch a leak by its content -- it catches the model
#: announcing its own source.  "let's recall historical prices", "in reality",
#: "? Actual" are not things a trace says about a number on its wire.
#:
#: Kept separate from ``outcome_language`` rather than folded into it because
#: the two want opposite readings.  D3's count is an upper bound to be argued
#: down; D4's is a floor, and each hit is a leak until someone reads it and
#: says otherwise.  Averaging them would destroy the only number worth acting
#: on.
RECALL_WORDS = (
    r"in reality", r"in hindsight", r"as (?:i|we) (?:recall|know)",
    r"let'?s recall", r"from memory", r"i remember",
    r"actually (?:went|rose|fell|hit|rallied|dropped)",
    r"\? ?Actual\b", r"historically\)",
    r"by (?:mid|late|early|end)-[A-Z][a-z]+ 20\d\d",
    r"(?:knowing|given) what (?:happened|came)",
)
_RECALL_RE = re.compile("(" + "|".join(RECALL_WORDS) + ")", re.IGNORECASE)

#: Numbers *claimed by a trace*.  The lookarounds keep the fractional tail of a
#: decimal from being read as an integer in its own right: ``0.125`` must not
#: yield ``125``.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*)(?!\d)(?!\.\d)(?![^\W\d])")

#: Numbers *available in an observation*, deliberately more permissive than the
#: one above.  The asymmetry is the whole correctness argument for D1, so it is
#: not an oversight:
#:
#: A false positive here is a number the model legitimately read off the wire
#: and the auditor calls a prophecy.  The first version of this script shared
#: one regex for both sides and flagged ``2025`` at step 0 -- a token sitting in
#: that very step's news line as *"could drive a rally in 2025."*, invisible to
#: the index only because a sentence-final period tripped the trailing
#: lookahead.  Every D1 candidate it printed was that bug.
#:
#: So the observation side takes every digit run, including the ones buried in
#: decimals and identifiers.  That costs sensitivity -- a genuine leak whose
#: digits happen to occur inside some unrelated number goes unseen -- and the
#: trade is taken knowingly: this detector's output is read by a human, and a
#: list that is mostly noise does not get read at all.
_OBSERVED_NUMBER_RE = re.compile(r"\d[\d,]*")

#: The unit scales the system block declares, as the zeros an integer grows by.
#:
#: The wire is quantised -- ``$1000 | nlv cash bp``, ``$100 | rlz unrlz`` -- and
#: the prompt shows the model how to undo it (*"rlz +52 = +$5,200"*).  So a
#: trace that reads ``A ... -115 ...`` and writes ``-11500`` has done the
#: arithmetic it was told to do, on the observation in front of it.
#:
#: Without this, D1 cannot see the link: ``115`` and ``11500`` are different
#: tokens, so the expansion collides with some *later* observation by chance
#: and gets reported as foreknowledge.  On the first real run every surviving
#: D1 candidate was exactly this -- ``-115``, ``-26``, ``-151`` and a ``227.50``
#: entry credit, all restated in dollars.  A detector whose entire output is
#: the model obeying the prompt is not measuring leakage.
_DECLARED_SCALES = ("0", "00", "000")
_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "jan feb mar apr may jun jul aug sep oct nov dec".split()
    )
}
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b",
    re.IGNORECASE,
)


@dataclass
class Hit:
    """One candidate, with everything a reader needs to overturn it."""

    detector: str
    step: int
    step_date: date
    token: str
    underlying: str | None
    context: str
    #: Where the same token shows up in the future, for ``future_number``.
    first_seen_step: int | None = None


@dataclass
class RunAudit:
    label: str
    steps: int = 0
    traced: int = 0
    hits: list[Hit] = field(default_factory=list)
    point_in_time_failure: str | None = None
    #: ``underlying -> pnl`` for positions opened on a flagged step.
    flagged_pnl: dict[str, float] = field(default_factory=dict)
    #: The same, restricted to steps carrying a recall-register hit.
    hard_pnl: dict[str, float] = field(default_factory=dict)
    total_pnl: float = 0.0
    flagged_steps: set[int] = field(default_factory=set)
    verbs_flagged: Counter = field(default_factory=Counter)
    verbs_clean: Counter = field(default_factory=Counter)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_date(text: str) -> date:
    return datetime.fromisoformat(text).date()


def _context(text: str, start: int, end: int, width: int = 70) -> str:
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return " ".join(text[lo:hi].split())


def _nearest_underlying(text: str, position: int, universe: Sequence[str]) -> str | None:
    """The ticker whose mention sits closest to ``position``, within a sentence.

    Attribution has to be per name -- a leak on one memorable quarter is the
    shape this is looking for, and a run-level count would average it away --
    but a trace names several tickers per paragraph, so "closest mention" is
    the honest resolution rather than "any mention in the step".
    """
    best: tuple[int, str] | None = None
    for ticker in universe:
        for match in re.finditer(rf"\b{re.escape(ticker)}\b", text):
            distance = abs(match.start() - position)
            if distance <= 200 and (best is None or distance < best[0]):
                best = (distance, ticker)
    return best[1] if best else None


def _numbers(text: str) -> Iterator[tuple[str, int, int]]:
    """Numeric tokens a trace can be held to, with their offsets."""
    for match in _NUMBER_RE.finditer(text):
        token = match.group(1).replace(",", "")
        if len(token.lstrip("0")) >= MIN_LEAK_DIGITS:
            yield token, match.start(), match.end()


def _observed_numbers(text: str) -> Iterator[str]:
    """Every numeric token an observation makes knowable.

    Yields the digit runs *and* their long suffixes, because the wire and the
    trace do not agree on where a number starts: an observation carrying a
    strike of ``12500`` is a perfectly good source for a trace that writes
    ``2500``, and treating those as different tokens invents a leak out of a
    formatting difference.  Suffixes only -- a prefix would let ``115000``
    excuse a trace that named ``11500``, which is a different quantity.
    """
    for match in _OBSERVED_NUMBER_RE.finditer(text):
        raw = match.group(0).replace(",", "")
        for token in (raw, *(raw + zeros for zeros in _DECLARED_SCALES)):
            for start in range(len(token)):
                suffix = token[start:]
                if len(suffix.lstrip("0")) >= MIN_LEAK_DIGITS:
                    yield suffix


#: How much of an outcome verb has to match for the wire to have said it too.
#:
#: The news block is prose, so it inflects: the trace writes *"PLTR beat"* over
#: a line reading *"revenue and earnings beating expectations"*, and *"NVDA
#: rallied"* over *"a rally in"*.  Comparing whole words would miss both and
#: report the model for reading its own feed.  Four characters merges
#: beat/beating, report/reported, rally/rallied, surge/surged without merging
#: anything that matters.
_STEM = 4


def _observed_claims(text: str, universe: Sequence[str]) -> dict[str, str]:
    """Per ticker, the observation text that mentions it, lowercased.

    A trace is entitled to everything on its own wire.  The news block is
    point-in-time by construction -- each row carries an ``available_time`` the
    runner asserts against -- so a hindsight verb the *observation* already
    applied to a name is context, not memory, and D3 must not report it.  On the
    first real run this was most of D3: the feed said *"Palantir reported Q3
    2024 results ... beating expectations"* and the auditor flagged the trace
    for saying PLTR beat.
    """
    claims: dict[str, list[str]] = {ticker: [] for ticker in universe}
    for line in text.splitlines():
        lowered = line.lower()
        for ticker in universe:
            if re.search(rf"\b{re.escape(ticker)}\b", line):
                claims[ticker].append(lowered)
    return {ticker: " ".join(lines) for ticker, lines in claims.items()}


def _dates_in(text: str, year_hint: int) -> Iterator[tuple[date, int, int]]:
    for match in _ISO_RE.finditer(text):
        y, m, d = (int(g) for g in match.groups())
        try:
            yield date(y, m, d), match.start(), match.end()
        except ValueError:
            continue
    for match in _MONTH_RE.finditer(text):
        month = _MONTHS[match.group(1)[:3].lower()]
        try:
            yield date(year_hint, month, int(match.group(2))), match.start(), match.end()
        except ValueError:
            continue


def audit_run(label: str, root: Path, universe: Sequence[str]) -> RunAudit:
    rows = sorted(
        read_jsonl(root / "decisions.jsonl"),
        key=lambda r: int(r.get("step_index") or 0),
    )
    audit = RunAudit(label=label, steps=len(rows))
    if not rows:
        return audit

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        traceback = manifest.get("traceback") or ""
        if "PointInTimeViolation" in str(traceback):
            audit.point_in_time_failure = str(traceback)[:400]

    dates = [_as_date(str(r["step_ts"])) for r in rows]
    observations = [str(r.get("observation") or "") for r in rows]

    # The set of numeric tokens each observation carries, and the earliest step
    # at which each token is *knowable*.  Built once: the per-step question is
    # "was this token available at or before now", which is a lookup.
    first_step_of: dict[str, int] = {}
    for index, text in enumerate(observations):
        for token in _observed_numbers(text):
            first_step_of.setdefault(token, index)

    #: Tokens this trace has already been flagged for.  With
    #: ``--reasoning-replay-turns 3`` the model is shown its own last three
    #: traces, so a number it invented at step 5 is on its context window at
    #: step 6 and restating it is not a second leak.  Flagging only the first
    #: statement keeps one contaminated step from smearing across the next
    #: three and inflating exposure.
    claimed: set[str] = set()

    for index, row in enumerate(rows):
        reasoning = str(row.get("reasoning") or "")
        if not reasoning.strip():
            continue
        audit.traced += 1
        today = dates[index]
        observation = observations[index]
        claims = _observed_claims(observation, universe)
        #: Dates this step's own wire names -- an earnings date on the ``ea``
        #: row, an election the news block is already bracing for.  A scheduled
        #: event is knowable in advance, and saying so is not a prophecy.
        on_the_wire = {when for when, _s, _e in _dates_in(observation, today.year)}

        # D1 -- a number the trace states before any observation carried it.
        for token, start, end in _numbers(reasoning):
            first = first_step_of.get(token)
            if first is None or first <= index:
                continue
            if token in claimed:
                continue
            claimed.add(token)
            audit.hits.append(
                Hit(
                    "future_number",
                    index,
                    today,
                    token,
                    _nearest_underlying(reasoning, start, universe),
                    _context(reasoning, start, end),
                    first_seen_step=first,
                )
            )

        # D2 -- a date after today that is not obviously a tenor.
        for when, start, end in _dates_in(reasoning, today.year):
            if when <= today or when in on_the_wire:
                continue
            window = reasoning[max(0, start - 60) : end + 60].lower()
            if any(word in window for word in TENOR_WORDS):
                continue
            audit.hits.append(
                Hit(
                    "future_date",
                    index,
                    today,
                    when.isoformat(),
                    _nearest_underlying(reasoning, start, universe),
                    _context(reasoning, start, end),
                )
            )

        # D4 -- the model naming its own source as memory.  No wire exemption:
        # an observation never says "let's recall", so there is nothing to
        # excuse a trace that does.
        for match in _RECALL_RE.finditer(reasoning):
            audit.hits.append(
                Hit(
                    "recall_register",
                    index,
                    today,
                    match.group(0).lower().strip(),
                    _nearest_underlying(reasoning, match.start(), universe),
                    _context(reasoning, match.start(), match.end(), width=110),
                )
            )

        # D3 -- hindsight language near a ticker.
        for match in _OUTCOME_RE.finditer(reasoning):
            ticker = _nearest_underlying(reasoning, match.start(), universe)
            if ticker is None:
                continue
            if match.group(0).lower()[:_STEM] in claims.get(ticker, ""):
                continue
            audit.hits.append(
                Hit(
                    "outcome_language",
                    index,
                    today,
                    match.group(0).lower(),
                    ticker,
                    _context(reasoning, match.start(), match.end()),
                )
            )

    audit.flagged_steps = {hit.step for hit in audit.hits}
    for index, row in enumerate(rows):
        verbs = Counter(
            line.split()[0]
            for line in str(row.get("completion") or "").splitlines()
            if line.split()
        )
        target = audit.verbs_flagged if index in audit.flagged_steps else audit.verbs_clean
        target.update(verbs)

    # Impact: the PnL of positions *opened* on a flagged step.  Opened and not
    # held, because a leak can only act through a decision, and the decision a
    # contaminated trace produced is the open.
    flagged_ts = {dates[i].isoformat() for i in audit.flagged_steps}
    hard_ts = {dates[i].isoformat() for i in _hard_steps(audit)}
    pnl: dict[str, float] = defaultdict(float)
    hard: dict[str, float] = defaultdict(float)
    for row in read_jsonl(root / "position_steps.jsonl"):
        step_pnl = float(row.get("realized_pnl_step") or 0.0) + float(
            row.get("mtm_pnl_step") or 0.0
        )
        audit.total_pnl += step_pnl
        opened = str(row.get("opened_ts") or "")[:10]
        if opened in flagged_ts:
            pnl[str(row.get("underlying"))] += step_pnl
        if opened in hard_ts:
            hard[str(row.get("underlying"))] += step_pnl
    audit.flagged_pnl = dict(pnl)
    audit.hard_pnl = dict(hard)
    return audit


def _hard_steps(audit: RunAudit) -> set[int]:
    """Steps carrying a recall-register hit -- the leaks, not the candidates."""
    return {hit.step for hit in audit.hits if hit.detector == "recall_register"}


def _report(audits: Sequence[RunAudit], universe: Sequence[str], show: int) -> None:
    print("=" * 78)
    print("EXPOSURE -- candidate steps, per run")
    print("=" * 78)
    print(f"{'run':<10}{'steps':>7}{'traced':>8}{'num':>7}{'date':>7}{'lang':>7}"
          f"{'RECALL':>8}{'steps':>7}{'%':>7}{'PIT':>6}")
    for audit in audits:
        counts = Counter(hit.detector for hit in audit.hits)
        hard = _hard_steps(audit)
        share = 100.0 * len(hard) / audit.traced if audit.traced else 0.0
        print(
            f"{audit.label:<10}{audit.steps:>7}{audit.traced:>8}"
            f"{counts['future_number']:>7}{counts['future_date']:>7}"
            f"{counts['outcome_language']:>7}{counts['recall_register']:>8}"
            f"{len(hard):>7}{share:>6.1f}%"
            f"{'FAIL' if audit.point_in_time_failure else 'ok':>6}"
        )
    print()
    print("PIT = the point-in-time gate on the *inputs*. 'ok' means the run finished")
    print("without a PointInTimeViolation, which is an assertion the runner raises on,")
    print("not a statistic. Everything else is about the *reasoning*.")
    print()
    print("num/date/lang are CANDIDATES and are upper bounds -- read them before")
    print("believing them. On the first real run they were dominated by the model")
    print("doing the unit arithmetic the prompt asked for and restating its own news")
    print("feed; the exemptions for both are in this script and tested.")
    print()
    print("RECALL is different and is the number to act on: the trace naming memory")
    print("as its source ('let's recall', 'in reality'). Treat each as a leak until")
    print("read and overturned. 'steps' counts the distinct steps carrying one.")

    print()
    print("=" * 78)
    print("EXPOSURE -- per underlying (flagged steps naming that name)")
    print("=" * 78)
    header = f"{'run':<10}" + "".join(f"{t:>8}" for t in universe)
    print(header)
    for audit in audits:
        per_name: Counter = Counter()
        for hit in audit.hits:
            if hit.underlying:
                per_name[hit.underlying] += 1
        print(f"{audit.label:<10}" + "".join(f"{per_name[t]:>8}" for t in universe))

    print()
    print("=" * 78)
    print("BIAS -- action mix on flagged vs clean steps")
    print("=" * 78)
    print(f"{'run':<10}{'':<8}" + "".join(f"{v:>7}" for v in ("O", "C", "X", "H")))
    for audit in audits:
        for name, counter in (("flagged", audit.verbs_flagged), ("clean", audit.verbs_clean)):
            total = sum(counter.values()) or 1
            print(
                f"{audit.label:<10}{name:<8}"
                + "".join(f"{100.0 * counter[v] / total:>6.0f}%" for v in ("O", "C", "X", "H"))
            )

    print()
    print("=" * 78)
    print("IMPACT -- PnL of positions opened on a flagged step")
    print("=" * 78)
    print(f"{'run':<10}{'recall $':>13}{'share':>8}{'candidate $':>14}{'share':>8}"
          f"{'total $':>13}")
    for audit in audits:
        flagged = sum(audit.flagged_pnl.values())
        hard = sum(audit.hard_pnl.values())
        total = audit.total_pnl

        def _share(value: float) -> str:
            return f"{100.0 * value / total:>7.1f}%" if total else "     n/a"

        print(
            f"{audit.label:<10}{hard:>13,.0f}{_share(hard)}"
            f"{flagged:>14,.0f}{_share(flagged)}{total:>13,.0f}"
        )
    print()
    print("Share is signed and can exceed 100% or go negative -- the denominator is a")
    print("net of winners and losers, not a magnitude. Read it as attribution, not as")
    print("a fraction of risk.")

    if show:
        print()
        print("=" * 78)
        print(f"CANDIDATES -- up to {show} per detector per run, read these")
        print("=" * 78)
        for audit in audits:
            for detector in (
                "recall_register", "future_number", "future_date", "outcome_language",
            ):
                chosen = [h for h in audit.hits if h.detector == detector][:show]
                if not chosen:
                    continue
                print(f"\n-- {audit.label} / {detector}")
                for hit in chosen:
                    seen = (
                        f" first in obs at step {hit.first_seen_step}"
                        if hit.first_seen_step is not None
                        else ""
                    )
                    print(f"   step {hit.step:>3} {hit.step_date} "
                          f"[{hit.underlying or '-'}] {hit.token!r}{seen}")
                    print(f"      ...{hit.context}...")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--show", type=int, default=3,
                        help="candidates to print per detector per run (0 to skip)")
    parser.add_argument(
        "--universe",
        default="AAPL,AMZN,GOOGL,META,MSFT,MU,NVDA,PLTR,TSLA,SPY",
        help="tickers used to attribute a candidate to a name",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    universe = tuple(t.strip() for t in args.universe.split(",") if t.strip())
    audits = []
    for spec in args.run:
        label, _, directory = spec.partition("=")
        audits.append(audit_run(label or directory, Path(directory), universe))

    _report(audits, universe, args.show)

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                [
                    {
                        "run": a.label,
                        "steps": a.steps,
                        "traced": a.traced,
                        "flagged_steps": sorted(a.flagged_steps),
                        "point_in_time_failure": a.point_in_time_failure,
                        "hits": [
                            {
                                "detector": h.detector,
                                "step": h.step,
                                "date": h.step_date.isoformat(),
                                "token": h.token,
                                "underlying": h.underlying,
                                "context": h.context,
                                "first_seen_step": h.first_seen_step,
                            }
                            for h in a.hits
                        ],
                        "flagged_pnl": a.flagged_pnl,
                        "total_pnl": a.total_pnl,
                    }
                    for a in audits
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
