"""Build the supervised frame for the Band D baselines, without leaking.

One row is one ``(run, trade_date, underlying)`` decision opportunity.  The
features are the market state the policy actually saw at that step; the label is
a strategy family or ``NONE``.

Five leakage guards, each of which exists because the obvious implementation
gets it wrong:

**Features come off the *quote* turn, never the *act* turn.**  A decision is two
turns.  ``turn='quote'`` sees the T/M/A/P market state and asks for prices;
``turn='act'`` sees *the returned prices* (``R OK Q ... m43242 c676``) and emits
the orders.  Building features from the act turn would hand the model the
realized premium and cost of the very packages it is choosing between -- and on
``runner.v1`` it would also hand it real tickers, which appear in 44% of those
rows.  Steps with no quote round carry the market state on the act turn instead,
so both are accepted, but an act-turn observation carrying ``R`` rows is refused
rather than parsed.

**No date ever becomes a feature.**  The observation is already date-blind: its
header is ``T 17 m7 PM``, a within-episode counter, not ``2024-09-24``.  The real
date is read from ``decisions.jsonl``'s ``step_ts`` and used *only* to assign a
row to a split.  A month dummy or a step index would be a calendar identity, and
the test window is a different calendar period, so such a feature could only
memorize.

**No account or position state.**  The A and P blocks are path-dependent: at a
fixed step, 4 redraws give 4 distinct A blocks and 1 identical M block.  Keeping
them would make the frame's size look like evidence -- and it would make the two
label sources incomparable, because an oracle label has no book attached to it.
The cost is stated rather than hidden: the teacher *did* see its book, so the
teacher branch is a lower bound on imitability.

**Pooling redraws does not add market information.**  The M block is
byte-identical across redraws, so 142 runs over 63 dates contain **630 distinct
market states**, not 89,460.  ``split_of`` therefore keys on the date, and
``date_folds`` blocks by contiguous date runs: a row-level shuffle would put a
byte-identical feature vector in both folds and report a validation score that
means nothing.

**The test window is refused by default.**  ``load_split`` raises on ``eval``
unless ``allow_test=True``, which exists so the final scoring pass can be written
deliberately and greppably.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env.policy.rules import read_observation
from ..env.pseudonyms import Pseudonyms
from ..env.statespace import Observation

#: The ten ``M``-row fields, in wire order.  Imported semantics, restated names:
#: ``rules._MARKET_COLUMNS`` owns the scales, and ``read_observation`` has
#: already divided them out by the time we see a value.
MARKET_FIELDS: tuple[str, ...] = (
    "ret", "rv", "iv", "dv", "w", "ts", "sk", "bf", "fi", "doi",
)

#: The nine wire family codes, plus the abstain class.  ``NONE`` is a real
#: decision and by far the most common one -- 68% of decision rows open nothing
#: -- so it is a class, not a dropped row.  Dropping it would train a model that
#: has never been taught to stand still and then score it on a book it churns.
FAMILIES: tuple[str, ...] = (
    "ol", "dv", "cv", "dg", "bf", "ls", "lg", "ib", "ic",
)
ABSTAIN = "NONE"
LABELS: tuple[str, ...] = FAMILIES + (ABSTAIN,)

#: Split name -> the config file that defines it.  Read from the repo rather
#: than hardcoded, because the windows were redesigned once already and a
#: literal here would have survived the redesign silently.
SPLIT_FILES: Mapping[str, str] = {
    "train": "configs/dates_sft.txt",
    "val": "configs/dates_rl.txt",
    "test": "configs/dates_eval.txt",
}


class LeakageError(RuntimeError):
    """Raised when a guard fires.  Never caught inside this module."""


@dataclass(frozen=True, slots=True)
class Row:
    """One decision opportunity.  ``label`` is ``None`` for unlabelled rows."""

    run: str
    episode_id: str
    trade_date: str
    step_index: int
    underlying: str
    features: Mapping[str, float | None]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class Frame:
    rows: tuple[Row, ...]
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(sorted({r.trade_date for r in self.rows}))

    @property
    def distinct_states(self) -> int:
        """How many distinct ``(date, underlying)`` cells the rows cover.

        Reported because it, and not ``len(self)``, is the sample size that
        bounds generalization: redraws multiply rows and not information.
        """
        return len({(r.trade_date, r.underlying) for r in self.rows})


def read_dates(repo: Path, split: str) -> tuple[str, ...]:
    path = repo / SPLIT_FILES[split]
    return tuple(
        line.strip() for line in path.read_text().splitlines() if line.strip()
    )


def split_map(repo: Path) -> dict[str, str]:
    """``trade_date -> split``, and it refuses to build if the windows overlap.

    The windows were overlapping once (63 of 63 dates shared between the SFT and
    eval files), which is the single worst thing that can be true of this
    mapping, so it is checked on every load rather than trusted.
    """
    out: dict[str, str] = {}
    for name in SPLIT_FILES:
        for day in read_dates(repo, name):
            if day in out:
                raise LeakageError(
                    f"{day} is in both {out[day]!r} and {name!r}; the windows "
                    "must be disjoint or every score below is contaminated"
                )
            out[day] = name
    return out


def _market_block_present(text: str) -> bool:
    return any(line.startswith("M ") for line in text.splitlines())


def _market_only(text: str) -> str:
    """Keep the ``T`` header and the ``M`` rows; drop everything else.

    Not merely a convenience.  ``read_observation`` decodes A and P rows too, and
    on this corpus it **raises** when it reaches them: the anonymizer's
    date/year scrubber has replaced numeric financial cells that happened to look
    like a year with the literal ``<y>``, e.g.
    ``P p09 U05 ib n 14 30 -154900 -<y> +19`` -- an unrealized PnL of about
    -$2,0xx, gone.  Measured over the 142 shipped runs: **1,219 of 53,154 P rows
    (2.3%, all 142 runs) and 361 of 8,946 A rows (4.0%, 120 runs)** carry a
    redaction.  **M rows: zero.**

    So the market block is the only part of this corpus that is numerically
    intact, which is a second, independent reason the feature set is market-only.
    Filtering here rather than widening ``_decode`` to tolerate ``<y>`` is
    deliberate: tolerating it would silently turn a redacted PnL into a
    ``None`` that imputation would then fill with a plausible number.
    """
    return "\n".join(
        line for line in text.splitlines()
        if line.startswith(("T ", "M "))
    )


def _has_quote_results(text: str) -> bool:
    """True only for *quote* receipts, ``R <status> Q <name> ...``.

    ``R`` is the receipt prefix for every verb, so a blanket ``startswith("R ")``
    is wrong and was: the market-state observation legitimately carries
    ``R OK O U06 ib n 8_30 ... m-43240 c989`` and ``R OK C p21``, which report
    what the *previous* step's own orders filled at.  That is decision-time
    feedback a policy is entitled to.  What must never be in a feature source is
    ``R OK Q AMZN ls n 8_30 ... m43242 c676`` -- the priced candidates for the
    decision being made now, which is the answer wearing a receipt's clothes.
    """
    for line in text.splitlines():
        cells = line.split()
        if len(cells) >= 3 and cells[0] == "R" and cells[2] == "Q":
            return True
    return False


def _cross_section(
    per_name: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    """Add a within-step cross-sectional rank for each field.

    Legitimate at decision time: it is computed from one observation, the same
    one the policy read.  It is also the only way the published cross-sectional
    sorts this baseline sits beside (a vol-spread sort, an idiosyncratic-vol
    sort) can be expressed at all -- they rank names against each other, and a
    model shown one name's ten numbers in isolation cannot represent them.
    """
    out: dict[str, dict[str, float | None]] = {
        name: dict(vals) for name, vals in per_name.items()
    }
    for field_name in MARKET_FIELDS:
        vals = [
            (name, v[field_name])
            for name, v in per_name.items()
            if v.get(field_name) is not None
        ]
        vals.sort(key=lambda kv: kv[1])  # type: ignore[arg-type,return-value]
        n = len(vals)
        for rank, (name, _) in enumerate(vals):
            # Midrank in [0, 1]; a single valid name gets 0.5 rather than a
            # divide-by-zero or a 0.0 that would read as "lowest".
            out[name][f"{field_name}_xrank"] = (
                0.5 if n == 1 else rank / (n - 1)
            )
        for name, v in per_name.items():
            out[name].setdefault(f"{field_name}_xrank", None)
    return out


def feature_names() -> tuple[str, ...]:
    return tuple(MARKET_FIELDS) + tuple(f"{f}_xrank" for f in MARKET_FIELDS)


def load_run(
    run_dir: Path,
    *,
    universe: Sequence[str] | None = None,
    pseudonym_seed: int = 0,
    to_real_tickers: bool = True,
) -> list[Row]:
    """Read one run directory into unlabelled-then-labelled rows.

    ``to_real_tickers`` maps the anonymized ``U01..U10`` back through
    ``Pseudonyms.draw(episode_id, universe, seed)``, which is deterministic and
    was validated against the ``runner.v1`` rows that still carry real names --
    five of five orders matched on name, family and tenor.  The mapping is keyed
    by *episode*, not by run, so the same code means the same ticker in all 142
    runs, which is what makes them poolable and what lets an oracle label
    computed in ticker space join to a teacher label recorded in code space.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if universe is None:
        raw = manifest.get("universe") or ""
        universe = [t for t in str(raw).replace(",", " ").split() if t]
    if to_real_tickers and not universe:
        raise LeakageError(
            f"{run_dir.name}: cannot de-anonymize without a universe; the "
            "manifest's 'universe' key is empty"
        )

    rows = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    quote_obs: dict[int, str] = {}
    acts: list[dict] = []
    for r in rows:
        if r.get("error"):
            # A dead step looks exactly like a healthy one downstream, so it is
            # dropped here where the error field is still visible.
            continue
        if r.get("turn") == "quote":
            quote_obs[r["step_index"]] = r["observation"]
        elif r.get("turn") == "act":
            acts.append(r)

    maps: dict[str, Pseudonyms] = {}
    out: list[Row] = []
    for act in acts:
        step = act["step_index"]
        episode = act["episode_id"]
        source = quote_obs.get(step, act["observation"])
        if not _market_block_present(source):
            continue
        if _has_quote_results(source):
            raise LeakageError(
                f"{run_dir.name} step {step}: the observation chosen for "
                "features carries R (quote-result) rows, which would leak the "
                "premium and cost of the candidates being chosen between"
            )
        state = read_observation(Observation(text=_market_only(source)))
        per_name = {name: dict(row.values) for name, row in state.market.items()}
        enriched = _cross_section(per_name)

        opened: dict[str, str] = {}
        for line in (act.get("completion") or "").splitlines():
            cells = line.split()
            if len(cells) >= 3 and cells[0] == "O" and cells[2] in FAMILIES:
                # First open wins: two opens on one name in one step is a real
                # (rare) completion, and picking the first keeps the label
                # single-valued without silently averaging two intents.
                opened.setdefault(cells[1], cells[2])

        if to_real_tickers:
            if episode not in maps:
                maps[episode] = Pseudonyms.draw(episode, list(universe), pseudonym_seed)
            code_to_ticker = maps[episode].to_ticker
        else:
            code_to_ticker = {}

        date = str(act["step_ts"])[:10]
        for code, values in enriched.items():
            name = code_to_ticker.get(code, code) if to_real_tickers else code
            out.append(
                Row(
                    run=run_dir.name,
                    episode_id=episode,
                    trade_date=date,
                    step_index=step,
                    underlying=name,
                    features={k: values.get(k) for k in feature_names()},
                    label=opened.get(code, ABSTAIN),
                )
            )
    return out


def load_teacher_frame(
    corpus_root: Path,
    *,
    repo: Path,
    split: str = "train",
    allow_test: bool = False,
    runs: Iterable[str] | None = None,
    pseudonym_seed: int = 0,
) -> Frame:
    """The 142 gate-survivor runs, restricted to one split's dates.

    ``corpus_root`` should be the *frozen* ``runs_main_sample/`` shipped inside
    the export package, not the live ``runs/`` tree: the live tree is mid
    campaign and mutable, and these 142 directories are the exact inputs the SFT
    student trained on.
    """
    if split == "test" and not allow_test:
        raise LeakageError(
            "refusing to load the test split. The eval window is held out; pass "
            "allow_test=True only in a final scoring pass, and never in "
            "preprocessing, hyperparameter search or model selection."
        )
    wanted = set(read_dates(repo, split))
    names = (
        sorted(runs)
        if runs is not None
        else sorted(
            p.name for p in corpus_root.iterdir()
            if p.is_dir() and (p / "decisions.jsonl").is_file()
        )
    )
    collected: list[Row] = []
    for name in names:
        for row in load_run(
            corpus_root / name, pseudonym_seed=pseudonym_seed
        ):
            if row.trade_date in wanted:
                collected.append(row)
    return Frame(rows=tuple(collected), feature_names=feature_names())


#: Wire code for each full family name, so an oracle row (which carries
#: ``family='outright'``) and a teacher row (which carries ``ol``) land on the
#: same label. Derived from the parser's own table rather than restated.
def _code_for_family() -> dict[str, str]:
    from ..env.actions import FAMILY_CODES

    return {name: code for code, name in FAMILY_CODES.items()}


#: How a candidate is scored when picking the best one for a cell.  These are
#: not interchangeable and the choice is a claim about what the baseline
#: measures -- see the measured degeneracy of each in ``STATUS.md``:
#:
#: ``profit``        realized after-cost PnL.  Picks ``outright`` in 96.9% of
#:                   SFT cells, because with hindsight an uncapped directional
#:                   position always beats a defined-risk one.
#: ``profit_per_risk`` the same numerator over ``max_loss``.  The denominator
#:                   is what stops unbounded upside winning by construction.
SCORERS: Mapping[str, Any] = {
    "profit": lambda r: r["profit"],
    "profit_per_risk": (
        lambda r: r["profit"] / r["max_loss"] if r.get("max_loss") else None
    ),
}


def load_oracle_labels(
    dump: Path, *, scorer: str = "profit", with_orientation: bool = False
) -> dict[tuple[str, str], str]:
    """``(trade_date, underlying) -> label`` from an oracle candidate dump.

    ``with_orientation`` decides whether the label is a family (``ol``) or a
    family and a direction (``ol b``).  It is not a cosmetic choice: for
    ``outright`` the orientation *is* the entire bet, so a family-only label
    hands the resolver a package it cannot build, and the measured distribution
    collapses onto one class.

    A cell whose best candidate does not clear zero is labelled ``NONE``.  That
    is only reachable because the dump is written upstream of the oracle's
    ``profit > 0`` filter; from the report's numbers it would be unrepresentable.
    """
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; expected one of {sorted(SCORERS)}")
    score = SCORERS[scorer]
    codes = _code_for_family()
    best: dict[tuple[str, str], tuple[float, str]] = {}
    with dump.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            value = score(row)
            if value is None:
                continue
            key = (row["trade_date"], row["underlying"])
            code = codes.get(row["family"], row["family"])
            label = (
                f"{code} {row['head'].split()[-1]}" if with_orientation else code
            )
            if key not in best or value > best[key][0]:
                best[key] = (value, label)
    return {k: (lab if v > 0 else ABSTAIN) for k, (v, lab) in best.items()}


def relabel(frame: Frame, labels: Mapping[tuple[str, str], str]) -> Frame:
    """Swap in a different label set, keeping the features untouched.

    Rows with no entry in ``labels`` are dropped rather than defaulted: an
    absent oracle label means the oracle never reached that cell, which is not
    the same statement as "the best action there was to do nothing", and
    conflating them would put a fabricated abstain into the training set.
    """
    kept = tuple(
        Row(
            run=r.run,
            episode_id=r.episode_id,
            trade_date=r.trade_date,
            step_index=r.step_index,
            underlying=r.underlying,
            features=r.features,
            label=labels[(r.trade_date, r.underlying)],
        )
        for r in frame.rows
        if (r.trade_date, r.underlying) in labels
    )
    return Frame(rows=kept, feature_names=frame.feature_names)


def load_feature_frame(
    run_dir: Path, *, repo: Path, split: str, allow_test: bool = False
) -> Frame:
    """Features from a single ``--policy hold`` run, with no labels attached.

    The oracle labels a ``(date, underlying)`` cell on windows where no policy
    was ever run, so the features cannot come from a policy's transcript there.
    They come from a hold run instead, which renders the identical state through
    the identical code path and trades nothing.

    Verified equivalent rather than assumed: over the 63 SFT dates, the
    de-anonymized ``M`` blocks of ``feat_sft`` and of teacher run
    ``sft_astra2_text_r0008`` are **identical on 63 of 63 dates**.  A raw text
    comparison fails, because a hold run is not anonymized and so orders its
    rows by real ticker while the teacher orders them by pseudonym -- the values
    are the same, and only the names and the row order differ.

    A hold run also has an empty book, so its ``A``/``P`` blocks are trivial.
    That costs nothing here because no feature reads them.
    """
    if split == "test" and not allow_test:
        raise LeakageError(
            "refusing to build a test-split frame without allow_test=True. "
            "Features on the test window are legitimate -- a model must see "
            "them to predict -- but the flag makes the final scoring pass "
            "explicit and greppable."
        )
    wanted = set(read_dates(repo, split))
    return Frame(
        rows=tuple(
            r for r in load_run(run_dir, to_real_tickers=False)
            if r.trade_date in wanted
        ),
        feature_names=feature_names(),
    )


def date_folds(dates: Sequence[str], n_folds: int = 3) -> list[tuple[list[str], list[str]]]:
    """Contiguous date blocks, for hyperparameter search inside one window.

    Contiguous rather than interleaved: adjacent trading days share overlapping
    option positions and highly autocorrelated volatility state, so an
    interleaved fold is a near-duplicate of its training set.  Returned as
    ``(train_dates, holdout_dates)`` so a caller cannot accidentally score on
    the rows it fitted.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ordered = sorted(set(dates))
    if len(ordered) < n_folds:
        raise ValueError(f"{len(ordered)} dates cannot make {n_folds} folds")
    size, out = len(ordered) / n_folds, []
    for k in range(n_folds):
        lo, hi = int(round(k * size)), int(round((k + 1) * size))
        hold = ordered[lo:hi]
        out.append(([d for d in ordered if d not in set(hold)], hold))
    return out


def label_counts(frame: Frame) -> dict[str, int]:
    out = {name: 0 for name in LABELS}
    for row in frame.rows:
        if row.label is not None:
            out[row.label] = out.get(row.label, 0) + 1
    return out
