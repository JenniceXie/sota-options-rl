"""Build NAV panels from a run's ledger (``env.ledger`` T2 → ``eval.schema``).

This is the join between the environment and the report, and it is deliberately
a *translation* and not a computation.  Every number here already exists in
``nav_panel.jsonl``; nothing is derived, imputed or filled.  If a quantity the
report wants was not measured by the environment, it is left ``None`` and named
in the provenance notes, because ``eval.schema`` treats ``None`` as "N/A" and
treats a number as a measurement.

Three translations are not identity, and each one is a decision:

**Timestamp to (trade_date, session).**  The ledger stamps every mark point with
a full ISO timestamp; the schema keys on a date and a session label.  The
mapping is exact — the environment already wrote the session — so this is a
reshape, not a bucketing.

**Levels to flows, on the grid actually reported.**  ``mtm_pnl`` and
``realized_pnl`` are *cumulative* in the ledger (they come off ``BookView``,
which reports the book's running totals), whereas ``rf_accrual``,
``cost_half_spread``, ``cost_fees`` and ``notional_traded`` are already
per-step.  A panel mixing the two would make ``Σ cost`` meaningful and
``Σ realized`` nonsense, so the cumulative pair is differenced and the whole
panel is flows — which is what section 5's cash identity is written in.

The differencing happens against the previous *kept* row, and the per-step
columns are summed over the marks in between.  Doing it the other way — reading
the flows off the full mark grid and then filtering — silently drops everything
that happened on an AM mark, so a PM panel would under-report interest and
costs.  Interest is the one that bites: cash accrues on both marks of the day,
and a PM panel that sampled instead of summing would report half the ``r_f``
the book actually earned.

**``vega_90`` is withheld.**  Section 5 names the column and
``portfolio_trajectory_rl_plan.md`` defines it as the root-time normalization
``vega * sqrt(90/T)``, which the environment does not apply: the ledger's
``dollar_vega`` is the raw sum over positions at their own tenors.  Writing the
un-normalized number into a column named ``vega_90`` would be a mislabel, and
normalizing it here would require a per-position tenor that T2 does not carry.
It is left ``None`` with a note; the unnormalized series is in T1 for anyone who
wants it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from portfolio_monkey.eval.schema import (
    NavPanel,
    NavRow,
    PanelProvenance,
    build_panel,
)

__all__ = [
    "LEDGER_BUILDER_VERSION",
    "LedgerCoverageError",
    "DecisionQuality",
    "read_manifest",
    "read_nav_rows",
    "build_ledger_panel",
    "decision_quality",
]

LEDGER_BUILDER_VERSION = "from_ledger.v1"

#: Ledger columns that are running totals rather than per-step flows.
CUMULATIVE_COLUMNS = ("mtm_pnl", "realized_pnl")

#: Ledger columns that are already per-step.  On a coarser reported grid these
#: are *summed* over the marks between kept rows rather than sampled, so the
#: panel's column totals equal the run's regardless of which grid is reported.
FLOW_COLUMNS = ("rf_accrual", "cost_half_spread", "cost_fees", "notional_traded")


class LedgerCoverageError(ValueError):
    """The ledger cannot produce a panel over the requested window."""


def read_manifest(root: Path | str) -> dict[str, Any]:
    """Return ``manifest.json``, or ``{}`` if the run did not write one.

    Absence is tolerated because a crashed run still has a usable ledger, and
    refusing to read it would throw away the steps it did complete.  It is
    recorded as a note so the panel says so.
    """
    path = Path(root) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_nav_rows(root: Path | str) -> Iterator[dict[str, Any]]:
    """Stream ``nav_panel.jsonl`` in write order."""
    path = Path(root) / "nav_panel.jsonl"
    if not path.exists():
        raise LedgerCoverageError(f"{path} does not exist; the run wrote no T2 table")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_ledger_panel(
    root: Path | str,
    *,
    grid: str = "PM",
    start: date | None = None,
    end: date | None = None,
    arm: str | None = None,
    track: str | None = None,
    universe: str = "",
    cost_level: str = "modelled",
    notes: Sequence[str] = (),
) -> NavPanel:
    """Read one run directory into a validated :class:`NavPanel`.

    ``grid`` is ``"PM"`` (the metric grid of protocol section 3) or ``"AM+PM"``.
    Choosing PM is not a data reduction that can bias the cumulative return —
    log returns telescope — but it *does* shallow the drawdown, which is why the
    choice is recorded in the provenance rather than assumed.

    ``arm`` and ``track`` override the labels the run was written under.  They
    are for relabelling a re-run, not for merging: a directory holds exactly one
    ``(arm, track)`` and this raises if it holds more.
    """
    if grid not in ("PM", "AM+PM"):
        raise ValueError(f"grid must be 'PM' or 'AM+PM'; got {grid!r}")

    manifest = read_manifest(root)
    raw = list(read_nav_rows(root))
    if not raw:
        raise LedgerCoverageError(f"{root}: nav_panel.jsonl is empty")

    written = {(row["arm"], row["track"]) for row in raw}
    if len(written) != 1:
        raise LedgerCoverageError(
            f"{root}: ledger holds {len(written)} (arm, track) pairs {sorted(written)}; "
            "a panel is one arm on one track"
        )
    ledger_arm, ledger_track = written.pop()

    kept: list[NavRow] = []
    dropped_session = 0
    dropped_window = 0
    #: Per-step columns from marks that were dropped for the reported grid.
    #: They roll forward into the next kept row rather than vanishing.
    pending = {name: 0.0 for name in FLOW_COLUMNS}
    #: Cumulative totals as of the last *reported* mark.  Advancing this on a
    #: mark that is dropped for the grid would throw away that mark's increment:
    #: a PM-only panel would report the PM-minus-AM move and call it the day.
    level = {name: 0.0 for name in CUMULATIVE_COLUMNS}

    for row in raw:
        stamp = datetime.fromisoformat(row["step_ts"])
        in_window = (start is None or stamp.date() >= start) and (
            end is None or stamp.date() <= end
        )
        on_grid = grid == "AM+PM" or row["session"] == "PM"

        for name in FLOW_COLUMNS:
            pending[name] += float(row.get(name) or 0.0)

        if not in_window:
            # Outside the window its costs are not this panel's to report, but
            # its *level* still moves the baseline, so the first row inside the
            # window measures the increment into the window and not the whole
            # history before it.
            dropped_window += 1
            pending = {name: 0.0 for name in FLOW_COLUMNS}
            for name in CUMULATIVE_COLUMNS:
                level[name] = float(row.get(name) or 0.0)
            continue
        if not on_grid:
            # Baseline deliberately untouched: this mark's move is reported on
            # the next kept row, together with its costs.
            dropped_session += 1
            continue

        flow = {name: float(row.get(name) or 0.0) - level[name] for name in CUMULATIVE_COLUMNS}
        for name in CUMULATIVE_COLUMNS:
            level[name] = float(row.get(name) or 0.0)

        kept.append(
            NavRow(
                arm=arm or ledger_arm,
                track=track or ledger_track,
                trade_date=stamp.date(),
                session=row["session"],
                nlv=float(row["nlv"]),
                cash=_optional(row, "cash"),
                rf_accrual=pending["rf_accrual"],
                mtm_pnl=flow["mtm_pnl"],
                realized_pnl=flow["realized_pnl"],
                cost_half_spread=pending["cost_half_spread"],
                cost_fees=pending["cost_fees"],
                net_dollar_delta=_optional(row, "net_dollar_delta"),
                gross_dollar_delta=_optional(row, "gross_dollar_delta"),
                dollar_gamma=_optional(row, "dollar_gamma"),
                vega_90=None,
                notional_traded=pending["notional_traded"],
                n_positions=_optional_int(row, "n_positions"),
                stale_mark_share=_optional(row, "stale_mark_share"),
            )
        )
        pending = {name: 0.0 for name in FLOW_COLUMNS}

    if len(kept) < 2:
        raise LedgerCoverageError(
            f"{root}: {len(kept)} rows on the {grid} grid"
            + (f" within [{start}, {end}]" if start or end else "")
            + f" (ledger has {len(raw)}); a panel needs at least two marks"
        )

    return build_panel(
        kept,
        PanelProvenance(
            source=str(Path(root)),
            grid=grid,
            universe=universe or str(manifest.get("universe", "")),
            cost_level=cost_level,
            notes=_notes(
                manifest,
                extra=notes,
                raw_rows=len(raw),
                kept_rows=len(kept),
                dropped_session=dropped_session,
                dropped_window=dropped_window,
            ),
        ),
    )


def _optional(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    return None if value is None else float(value)


def _optional_int(row: Mapping[str, Any], name: str) -> int | None:
    value = row.get(name)
    return None if value is None else int(value)


def _notes(
    manifest: Mapping[str, Any],
    *,
    extra: Sequence[str],
    raw_rows: int,
    kept_rows: int,
    dropped_session: int,
    dropped_window: int,
) -> tuple[str, ...]:
    notes = [f"built by {LEDGER_BUILDER_VERSION}"]
    if manifest:
        for key in ("ledger_version", "env_config_version", "state_space_id", "policy_model"):
            if manifest.get(key):
                notes.append(f"{key}={manifest[key]}")
        resolvers = manifest.get("resolvers")
        if isinstance(resolvers, Mapping):
            notes.append(
                "resolvers=" + ",".join(f"{k}:{v}" for k, v in sorted(resolvers.items()))
            )
        terminated = manifest.get("terminated")
        if terminated:
            notes.append(f"run terminated: {terminated}")
    else:
        notes.append("no manifest.json; run may not have completed")

    notes.append(
        "mtm_pnl and realized_pnl differenced from cumulative ledger levels; "
        "rf_accrual and cost columns summed over marks between reported rows"
    )
    notes.append(
        "vega_90 withheld: ledger records unnormalized dollar vega, "
        "root-time normalization not applied"
    )
    if dropped_session:
        notes.append(f"dropped {dropped_session} AM mark(s) for the PM metric grid")
    if dropped_window:
        notes.append(f"dropped {dropped_window} mark(s) outside the requested window")
    notes.append(f"{kept_rows} of {raw_rows} ledger marks retained")
    notes.extend(extra)
    return tuple(notes)


# ---------------------------------------------------------------------------
# run quality
# ---------------------------------------------------------------------------


@dataclass
class DecisionQuality:
    """How well the policy spoke, independent of how well it traded.

    PnL alone cannot distinguish a policy that chose to hold from one whose
    orders were all rejected as ungrammatical, and the two call for opposite
    responses: the first is a result, the second is a prompt bug.  This is
    reported alongside every LLM arm for that reason.

    The counts come from the ``results`` the environment produced rather than
    from any parse of the completion, because ``results`` is what actually
    happened.  ``by_status`` is kept whole rather than bucketed into
    "good"/"bad": ``E_SYNTAX`` (the model cannot write the grammar),
    ``E_NO_CHAIN`` (our data is missing) and ``E_SIZE_CASH`` (the model asked
    for more than the book can fund) are three different problems with three
    different fixes, and collapsing them hides which one a run has.

    **Quote turns are counted apart from acting turns**, and the split is not
    cosmetic.  A ``Q`` turn's ``results`` are prices, and every one of them
    reads ``status: OK``: folded into the same counters they would inflate
    ``orders_attempted``, drive ``fill_rate`` toward 1.0, and make the headline
    say the policy's orders almost always filled when what almost always
    succeeded was asking a question.  So order counting reads acting turns
    only, while cost and reliability -- tokens, latency, provider errors,
    truncation -- read every turn, because the run really did pay for both.
    A truncated quote turn in particular is a step whose decision was made
    blind, and hiding it would hide the worst failure this path can have.
    """

    decisions: int = 0
    #: Proposal turns, and the candidates they priced.  Zero on a one-turn arm.
    quote_turns: int = 0
    quotes_answered: int = 0
    abstentions: int = 0
    orders_attempted: int = 0
    orders_filled: int = 0
    risk_closes: int = 0
    provider_errors: int = 0
    truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_seconds: float = 0.0
    by_status: Counter[str] = dataclass_field(default_factory=Counter)

    @property
    def turns(self) -> int:
        """Every completion the run paid for, quote turns included."""
        return self.decisions + self.quote_turns

    @property
    def abstain_rate(self) -> float:
        """Decision points at which the policy placed no order at all."""
        return self.abstentions / self.decisions if self.decisions else 0.0

    @property
    def fill_rate(self) -> float:
        """Orders that became positions, per order the policy wrote."""
        return self.orders_filled / self.orders_attempted if self.orders_attempted else 0.0

    @property
    def provider_error_rate(self) -> float:
        return self.provider_errors / self.turns if self.turns else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Prompt tokens served from the provider's prefix cache.

        This is the number that says whether the append-only context of ``R7``
        is working.  A run near zero here is rewriting its prefix somewhere and
        paying full price on every step.
        """
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "quote_turns": self.quote_turns,
            "quotes_answered": self.quotes_answered,
            "abstain_rate": self.abstain_rate,
            "orders_attempted": self.orders_attempted,
            "fill_rate": self.fill_rate,
            "risk_closes": self.risk_closes,
            "provider_error_rate": self.provider_error_rate,
            # Per turn, not per decision: a two-turn arm with one truncated
            # quote per step would otherwise report a rate above 1.
            "truncation_rate": self.truncated / self.turns if self.turns else 0.0,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "mean_latency_seconds": (
                self.latency_seconds / self.turns if self.turns else 0.0
            ),
            "by_status": dict(sorted(self.by_status.items())),
        }


def decision_quality(root: Path | str) -> DecisionQuality:
    """Summarize ``decisions.jsonl``.

    Returns an all-zero summary for a run with no decision table, which is the
    right answer for the hold control: it made no calls, so its rates are not
    unknown, they are zero.
    """
    quality = DecisionQuality()
    path = Path(root) / "decisions.jsonl"
    if not path.exists():
        return quality

    for row in _read_jsonl(path):
        # Absent on every ledger written before the quote turn existed, and
        # those runs had exactly one turn per step -- so the default is the
        # truth about them, not a guess.
        is_quote = str(row.get("turn") or "act") == "quote"
        if is_quote:
            quality.quote_turns += 1
            quality.quotes_answered += len(row.get("results") or ())
        else:
            quality.decisions += 1
            attempted = 0
            for result in row.get("results") or ():
                if not isinstance(result, Mapping):
                    continue
                status = str(result.get("status") or "")
                quality.by_status[status] += 1
                # ``order == "-"`` marks a close the risk controls took on their
                # own; counting it as an order the policy wrote would credit the
                # policy with activity it did not choose.
                if result.get("order") == "-":
                    quality.risk_closes += 1
                    continue
                attempted += 1
                if status == "OK":
                    quality.orders_filled += 1
            quality.orders_attempted += attempted
            if attempted == 0:
                quality.abstentions += 1

        if row.get("error"):
            quality.provider_errors += 1
        if row.get("finish_reason") == "length":
            quality.truncated += 1
        quality.prompt_tokens += int(row.get("prompt_tokens") or 0)
        quality.completion_tokens += int(row.get("completion_tokens") or 0)
        quality.latency_seconds += float(row.get("latency_seconds") or 0.0)
        extra = row.get("extra")
        if isinstance(extra, Mapping):
            quality.cached_tokens += int(extra.get("cached_tokens") or 0)
    return quality


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
