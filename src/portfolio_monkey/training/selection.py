"""Which recorded runs enter a sample, and -- just as important -- why the rest did not.

A FILTER OBJECT, NOT A LIST OF RUN NAMES.  The 126 trajectories that passed the
``V + ASR>=1 + MDD<=10%`` gate came out of the first thousand draws.  A second
thousand is on the queue, a third is plausible, and each ablation arm is drawn
separately.  Freezing the sample as a list of directory names would mean the
next batch enters by hand-editing, which is how two arms end up selected under
quietly different rules.  So the sample is a :class:`SelectionSpec` -- thresholds
and scopes as data -- and the run list is whatever that spec admits *today*.

EVERY THRESHOLD IS A PARAMETER AND EVERY ONE OF THEM DEFAULTS TO "ABSENT".
``SelectionSpec()`` with no arguments selects nothing away except unreadable
directories.  The ruled main-sample gate is a named constructor
(:func:`main_sample_spec`), not a default, because a default threshold is a
threshold nobody chose: it would be applied to the RL corpus, the eval sample
and every ablation without anyone deciding it should be.

REJECTION IS REPORTED PER GATE, NOT AS A COUNT.  ``998 -> 126`` says a filter
bit hard; it does not say which clause bit.  :meth:`Selection.breakdown` gives
both readings that matter: how many runs each gate rejects *applied alone*
(which is comparable across arms) and how many it rejects *in sequence* (which
sums to the total).  The two differ whenever gates correlate, and on this sample
they correlate strongly -- frictions and Sharpe are not independent.

TWO GATES THAT ARE NOT ABOUT PERFORMANCE AND MUST NOT BE FORGOTTEN:

*   ``summary.json`` must exist.  It is written as the last act of a run, so its
    absence is the signature of a run that was killed, not of a run that did
    badly.  Measured 2026-09-24: of 547 ``sft_astra_text_r1[0-9]{3}`` directories
    left behind by the cancelled campaign, 148 are torsos with no summary and
    399 are complete runs.  A glob over the directory names cannot tell those
    apart; this gate can.

*   ``runner_version`` splits the sample even though ``env_fingerprint`` does
    not.  Before ``runner.v2`` the quote round rendered its price block straight
    off the state space, bypassing the environment's mask, so a de-identified
    arm put real tickers in front of the policy -- 537 of 537 quote turns,
    measured.  The mask lives outside ``EnvConfig``, so the fingerprint is
    byte-identical across the fix: as of 2026-09-24 there are 1,398 complete
    ``sft_astra*`` runs, all ``runner.v1``, all on fingerprint ``556df22b``.
    Nothing but this field separates a leaked draw from a clean one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

__all__ = [
    "Gate",
    "RunRecord",
    "Selection",
    "SelectionSpec",
    "gates_for",
    "main_sample_spec",
    "read_run",
    "scan_runs",
    "select",
]


class SelectionError(ValueError):
    """A selection was asked for something it cannot honestly answer."""


# --------------------------------------------------------------------------
# one run, read cheaply
# --------------------------------------------------------------------------

#: An arm name ends in the redraw index the submit script dealt it, zero padded
#: to a fixed four.  The width is fixed at the *writer*, so it is matched here
#: rather than reconstructed: ``seq -w`` pads to its own upper bound, and a
#: campaign submitted in two ranges therefore contains both ``r8`` and ``r0008``
#: unless the writer pins the width.  Anchored at the end, and the prefix is
#: stripped before this runs, because ``sed 's/.*_r//'`` is greedy and an arm
#: called ``sft_astra2_text_r0007`` would otherwise yield ``0007`` from the
#: wrong ``r``.
_INDEX_TAIL = re.compile(r"r(\d+)$")


@dataclass(frozen=True)
class RunRecord:
    """What a run directory says about itself, before any metric is computed.

    Everything here comes from ``manifest.json`` and ``summary.json`` plus one
    pass over ``decisions.jsonl``, and the pass is only there for
    ``peak_tokens``: the summary reports token *totals*, and a run that blew the
    context budget on one turn is invisible in a total.
    """

    run_dir: Path
    arm: str
    #: The redraw index parsed out of the arm name, or ``None`` when the name
    #: does not end in one.  Never used to *split* a queue -- arms reuse index
    #: ranges across prefixes -- only to sub-range within a prefix already fixed.
    index: int | None
    runner_version: str
    ledger_version: str
    state_space_id: str
    env_fingerprint: str
    policy_model: str
    quote_channel: str
    anonymize: bool
    textual_context: bool
    complete: bool
    failure: str | None
    terminated: str | None
    episodes: int
    decisions: int
    provider_error_rate: float
    truncation_rate: float
    peak_tokens: int
    status_counts: Mapping[str, int]
    wall_seconds: float
    window: tuple[str, str] | None
    log_return: float | None
    #: Level and risk metrics, attached later by whatever computes them.  Kept
    #: as an open mapping rather than as fields because the gate set is expected
    #: to grow, and because this module must not import the eval stack: scanning
    #: ten thousand directories to find out which two hundred are candidates has
    #: to stay cheap enough to do casually.
    metrics: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def rejected_order_rate(self) -> float:
        """Share of order results the environment refused, over all statuses.

        Not an error rate in the provider sense: these are the policy's own
        malformed or inadmissible orders, and a run that spends 20% of its
        turns being refused is a run whose completions teach the student to
        write refusals.
        """
        total = sum(self.status_counts.values())
        if not total:
            return 0.0
        rejected = sum(v for k, v in self.status_counts.items() if k.startswith("E_"))
        return rejected / total

    @property
    def real_name_leaks(self) -> int:
        """Orders refused for naming a real underlying under de-identification.

        Nonzero means the policy had the map -- which, before ``runner.v2``, it
        sometimes did, handed to it by the quote round.
        """
        return int(self.status_counts.get("E_REAL_NAME", 0))

    def with_metrics(self, metrics: Mapping[str, float | None]) -> RunRecord:
        return replace(self, metrics={**self.metrics, **metrics})

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["run_dir"] = str(self.run_dir)
        out["status_counts"] = dict(self.status_counts)
        out["metrics"] = dict(self.metrics)
        out["rejected_order_rate"] = self.rejected_order_rate
        out["real_name_leaks"] = self.real_name_leaks
        return out


def _index_of(arm: str, prefixes: Sequence[str]) -> int | None:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if arm.startswith(prefix):
            match = _INDEX_TAIL.search(arm[len(prefix) :])
            return int(match.group(1)) if match else None
    match = _INDEX_TAIL.search(arm)
    return int(match.group(1)) if match else None


def read_run(run_dir: Path | str, *, prefixes: Sequence[str] = ()) -> RunRecord | None:
    """One directory, or ``None`` if it is not a finished run.

    ``None`` rather than an exception, and rather than a record with
    ``complete=False``, for exactly one case: there is no ``manifest.json``.
    That directory has no arm name, no fingerprint and no episode list, so there
    is nothing to report *about* -- it is a run in flight or a run killed before
    it wrote anything, and counting it as a rejection would put live jobs in the
    reject column of a campaign report.  A directory that has a manifest but no
    summary is a different animal: it ran and died, it is worth counting, and it
    comes back with ``complete=False``.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    summary_path = run_dir / "summary.json"
    summary: Mapping[str, object] = {}
    complete = False
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            complete = True
        except (OSError, json.JSONDecodeError):
            complete = False

    quality = summary.get("decision_quality") or {}
    if not isinstance(quality, Mapping):
        quality = {}
    status_counts = quality.get("by_status") or {}
    if not isinstance(status_counts, Mapping):
        status_counts = {}

    quotes = manifest.get("quotes") or {}
    window = manifest.get("window")

    arm = str(manifest.get("arm", run_dir.name))
    return RunRecord(
        run_dir=run_dir,
        arm=arm,
        index=_index_of(arm, prefixes),
        runner_version=str(manifest.get("runner_version", "")),
        ledger_version=str(manifest.get("ledger_version", "")),
        state_space_id=str(manifest.get("state_space_id", "")),
        env_fingerprint=str(manifest.get("env_fingerprint", "")),
        policy_model=str(manifest.get("policy_model", "")),
        quote_channel=str(quotes.get("channel", "")) if isinstance(quotes, Mapping) else "",
        anonymize=bool(manifest.get("anonymize", False)),
        textual_context=bool(manifest.get("textual_context", False)),
        complete=complete,
        failure=_text_or_none(manifest.get("failure")),
        terminated=_text_or_none(manifest.get("terminated")),
        episodes=len(manifest.get("episodes") or ()),
        decisions=int(quality.get("decisions") or 0),
        provider_error_rate=float(quality.get("provider_error_rate") or 0.0),
        truncation_rate=float(quality.get("truncation_rate") or 0.0),
        peak_tokens=_peak_tokens(run_dir),
        status_counts={str(k): int(v) for k, v in status_counts.items()},
        wall_seconds=float(manifest.get("wall_seconds") or 0.0),
        window=(str(window[0]), str(window[1]))
        if isinstance(window, Sequence) and not isinstance(window, str) and len(window) == 2
        else None,
        log_return=_float_or_none(manifest.get("log_return")),
    )


def _text_or_none(value: object) -> str | None:
    return None if value in (None, "", False) else str(value)


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _peak_tokens(run_dir: Path) -> int:
    """Largest prompt+completion any single turn of this run occupied.

    The one field worth a full pass over ``decisions.jsonl``.  The summary
    reports the *sum* over turns, and a sum cannot answer the question the
    context budget asks, which is whether any one turn came near the cap.  The
    astra sample peaked at 32,763 against a 32,768 budget -- five tokens of
    headroom on the worst turn, invisible in a 1.2M total.
    """
    path = run_dir / "decisions.jsonl"
    if not path.exists():
        return 0
    peak = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            used = int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
            peak = max(peak, used)
    return peak


def scan_runs(
    root: Path | str,
    *,
    prefixes: Sequence[str] = (),
    workers: int = 1,
) -> list[RunRecord]:
    """Every run under ``root`` whose arm name starts with one of ``prefixes``.

    ``prefixes`` is how a campaign is scoped, and it is deliberately the *only*
    name-based scope offered at this level.  An index range is available in
    :class:`SelectionSpec`, but it applies within a prefix, never across: two
    campaigns reuse the same ``r0001..r1000`` indices, so an index range is
    meaningless until the prefix has already fixed which campaign is meant.

    The prefix is matched against the arm name from the manifest and against the
    directory name, and both must agree, because they can disagree: the submit
    script names the job and the arm together, but a hand-run job can be given
    an ``--arm`` that does not match where it wrote.
    """
    root = Path(root)
    if not root.is_dir():
        raise SelectionError(f"{root}: not a directory")
    candidates = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and (not prefixes or any(d.name.startswith(p) for p in prefixes))
    )
    if workers > 1 and len(candidates) > workers:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            found = list(pool.map(_read_one, ((d, tuple(prefixes)) for d in candidates), chunksize=8))
    else:
        found = [read_run(d, prefixes=prefixes) for d in candidates]
    records = [r for r in found if r is not None]
    if prefixes:
        records = [r for r in records if any(r.arm.startswith(p) for p in prefixes)]
    return records


def _read_one(args: tuple[Path, tuple[str, ...]]) -> RunRecord | None:
    run_dir, prefixes = args
    return read_run(run_dir, prefixes=prefixes)


# --------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """One named clause of a filter.

    Named because the name is the output.  A filter that returns a count has
    told you a sample got smaller; a filter that returns which gate rejected
    which run has told you whether the sample is smaller for a reason you meant.
    """

    name: str
    test: Callable[[RunRecord], bool]

    def __call__(self, record: RunRecord) -> bool:
        return self.test(record)


@dataclass(frozen=True)
class SelectionSpec:
    """The sample, as data.

    Every field is ``None``/empty by default, and ``None`` means *no gate* --
    not "use the usual value".  The distinction matters because this object is
    serialized into the corpus it produces: reading ``min_sharpe: null`` off a
    manifest tells you the sample was unfiltered on Sharpe, whereas a default of
    1.0 would have told you nothing about whether anyone chose it.
    """

    #: Arm-name prefixes that scope the campaign.  Empty admits every arm.
    prefixes: tuple[str, ...] = ()
    #: Inclusive redraw-index bounds, applied *within* the prefixes above.
    min_index: int | None = None
    max_index: int | None = None

    # -- provenance: what the run has to have been ------------------------
    env_fingerprint: str | None = None
    runner_version: str | None = None
    state_space_id: str | None = None
    policy_model: str | None = None
    quote_channel: str | None = None
    require_anonymize: bool | None = None
    require_textual_context: bool | None = None
    window: tuple[str, str] | None = None

    # -- validity: what the run has to have done --------------------------
    require_complete: bool = True
    expected_episodes: int | None = None
    forbid_failure: bool = True
    forbid_terminated: bool = True
    max_provider_error_rate: float | None = None
    max_truncation_rate: float | None = None
    max_peak_tokens: int | None = None
    max_rejected_order_rate: float | None = None
    forbid_real_name_leak: bool = False
    min_wall_seconds: float | None = None
    min_decisions: int | None = None

    # -- performance: metric gates, keyed by whatever filled ``metrics`` ---
    #: ``{"asr": 1.0}`` keeps runs whose ``metrics["asr"] >= 1.0``.  A run whose
    #: metrics do not carry the key is REJECTED, not admitted: a missing metric
    #: is an unanswered question, and admitting on an unanswered question is how
    #: an unmeasurable run slips into a sample that claims to be measured.
    metric_minimums: Mapping[str, float] = field(default_factory=dict)
    #: ``{"mdd_pm": 0.10}`` keeps runs whose ``metrics["mdd_pm"] <= 0.10``.
    metric_maximums: Mapping[str, float] = field(default_factory=dict)

    #: Free-text note carried into the corpus provenance, e.g. the date and the
    #: ruling that set these numbers.
    label: str = ""

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["metric_minimums"] = dict(self.metric_minimums)
        out["metric_maximums"] = dict(self.metric_maximums)
        return out


def main_sample_spec(
    *,
    prefixes: Sequence[str],
    runner_version: str | None = None,
    min_sharpe: float = 0.75,
    max_drawdown: float = 0.15,
    max_peak_tokens: int = 32_768,
    max_rejected_order_rate: float = 0.06,
    expected_episodes: int = 3,
    label: str = "",
) -> SelectionSpec:
    """The ruled ``V + ASR>=0.75 + MDD<=15%`` gate, reproduced clause for clause.

    ``V`` is the union of two validity predicates that lived in separate scratch
    scripts and were applied one after the other, so the 126 were gated by both:
    the structural half (three episodes, no provider error, a wall time that
    rules out a run that died at the first request) and the quality half (peak
    tokens within budget, no ``E_REAL_NAME``, rejected-order rate at or under
    6%).  Written out here as one spec so the next batch cannot be gated by only
    half of it.

    Every threshold is an argument with the ruled value as its default, because
    the numbers are a ruling and not a law, and a second thousand under a changed
    environment is entitled to be asked whether they still cut where they were
    meant to.  Changing one here changes it in exactly one place, and the changed
    value is what lands in the provenance.

    THE DEFAULTS MOVED 2026-09-24, user ruling, verbatim: "ASR >= 0.75, MDD <=
    15%".  The previous ``ASR>=1 + MDD<=10%`` was set against the first (leaking,
    ``runner.v1``) thousand; priced against the 995 valid ``sft_astra2_text_*``
    draws it keeps 98 runs, and the new pair keeps 142 -- +45% corpus for -1.27pp
    of mean TR and -0.22 of mean ASR, with the 38 marginal runs showing no extra
    leverage (AVOL 22.4% vs 21.9%) and no extra friction (12.26% vs 12.11%).

    ``max_drawdown`` IS INOPERATIVE AT THIS PAIR and is kept as a guard, not as a
    filter.  The worst PM drawdown among the ``asr >= 0.75`` runs is 13.86%, so
    every cap at or above 14% keeps all 142; on that sample the clause removes
    nothing and the sample size is set entirely by ``min_sharpe``.  It stays
    because a future draw is not bound by that measurement -- but read a report
    that credits the gate's selectivity to the drawdown clause with suspicion.

    ``mdd_pm`` IS THE PM-GRID DRAWDOWN (``ArmMetrics.max_drawdown_daily``), never
    the AM+PM one.  Both exist and they are the same statistic at two sampling
    rates, but AM rows carry a ~97% ``stale_mark_share`` against ~0.3% at PM, so
    the finer grid interleaves a marked-to-market point with a marked-to-stale
    one and reports roughly 1.5x the drawdown as an artefact of the mark
    convention.  Gating on the AM+PM figure would fail 10 of these 142.

    ``runner_version`` has no default on purpose.  Pinning it is almost always
    right -- ``runner.v1`` and ``runner.v2`` differ in what the policy was shown
    -- but *which* version a given sample wants is a ruling, and a default would
    make it silently.
    """
    return SelectionSpec(
        prefixes=tuple(prefixes),
        runner_version=runner_version,
        require_complete=True,
        expected_episodes=expected_episodes,
        forbid_failure=True,
        forbid_terminated=True,
        max_provider_error_rate=0.0,
        max_peak_tokens=max_peak_tokens,
        max_rejected_order_rate=max_rejected_order_rate,
        forbid_real_name_leak=True,
        min_wall_seconds=120.0,
        metric_minimums={"asr": min_sharpe},
        metric_maximums={"mdd_pm": max_drawdown},
        label=label,
    )


def gates_for(spec: SelectionSpec) -> tuple[Gate, ...]:
    """The spec, expanded into named clauses in the order they are applied.

    Order is cheapest-and-most-structural first.  That is not an optimization:
    the sequential breakdown reads as a funnel, and a funnel whose first stage
    is "Sharpe" would report a torso as a performance failure.
    """
    gates: list[Gate] = []

    if spec.prefixes:
        gates.append(
            Gate(
                f"arm starts with {'|'.join(spec.prefixes)}",
                lambda r, p=tuple(spec.prefixes): any(r.arm.startswith(x) for x in p),
            )
        )
    if spec.min_index is not None:
        gates.append(Gate(f"index >= {spec.min_index}",
                          lambda r, v=spec.min_index: r.index is not None and r.index >= v))
    if spec.max_index is not None:
        gates.append(Gate(f"index <= {spec.max_index}",
                          lambda r, v=spec.max_index: r.index is not None and r.index <= v))

    if spec.require_complete:
        gates.append(Gate("summary.json present (run finished)", lambda r: r.complete))
    for attr, value in (
        ("runner_version", spec.runner_version),
        ("env_fingerprint", spec.env_fingerprint),
        ("state_space_id", spec.state_space_id),
        ("policy_model", spec.policy_model),
        ("quote_channel", spec.quote_channel),
    ):
        if value is not None:
            gates.append(
                Gate(f"{attr} == {value}", lambda r, a=attr, v=value: getattr(r, a) == v)
            )
    if spec.require_anonymize is not None:
        gates.append(Gate(f"anonymize is {spec.require_anonymize}",
                          lambda r, v=spec.require_anonymize: r.anonymize is v))
    if spec.require_textual_context is not None:
        gates.append(Gate(f"textual_context is {spec.require_textual_context}",
                          lambda r, v=spec.require_textual_context: r.textual_context is v))
    if spec.window is not None:
        gates.append(Gate(f"window == {spec.window[0]}..{spec.window[1]}",
                          lambda r, v=spec.window: r.window == v))

    if spec.forbid_failure:
        gates.append(Gate("no recorded failure", lambda r: r.failure is None))
    if spec.forbid_terminated:
        gates.append(Gate("no episode terminated early", lambda r: r.terminated is None))
    if spec.expected_episodes is not None:
        gates.append(Gate(f"episodes == {spec.expected_episodes}",
                          lambda r, v=spec.expected_episodes: r.episodes == v))
    if spec.min_decisions is not None:
        gates.append(Gate(f"decisions >= {spec.min_decisions}",
                          lambda r, v=spec.min_decisions: r.decisions >= v))
    if spec.max_provider_error_rate is not None:
        gates.append(Gate(f"provider_error_rate <= {spec.max_provider_error_rate}",
                          lambda r, v=spec.max_provider_error_rate: r.provider_error_rate <= v))
    if spec.max_truncation_rate is not None:
        gates.append(Gate(f"truncation_rate <= {spec.max_truncation_rate}",
                          lambda r, v=spec.max_truncation_rate: r.truncation_rate <= v))
    if spec.min_wall_seconds is not None:
        gates.append(Gate(f"wall_seconds >= {spec.min_wall_seconds}",
                          lambda r, v=spec.min_wall_seconds: r.wall_seconds >= v))
    if spec.max_peak_tokens is not None:
        gates.append(Gate(f"peak_tokens <= {spec.max_peak_tokens}",
                          lambda r, v=spec.max_peak_tokens: r.peak_tokens <= v))
    if spec.max_rejected_order_rate is not None:
        gates.append(Gate(f"rejected_order_rate <= {spec.max_rejected_order_rate}",
                          lambda r, v=spec.max_rejected_order_rate: r.rejected_order_rate <= v))
    if spec.forbid_real_name_leak:
        gates.append(Gate("no E_REAL_NAME order", lambda r: r.real_name_leaks == 0))

    for key, floor in sorted(spec.metric_minimums.items()):
        gates.append(Gate(f"{key} >= {floor}", lambda r, k=key, v=floor: _at_least(r, k, v)))
    for key, ceiling in sorted(spec.metric_maximums.items()):
        gates.append(Gate(f"{key} <= {ceiling}", lambda r, k=key, v=ceiling: _at_most(r, k, v)))
    return tuple(gates)


def _at_least(record: RunRecord, key: str, floor: float) -> bool:
    value = record.metrics.get(key)
    return value is not None and value >= floor


def _at_most(record: RunRecord, key: str, ceiling: float) -> bool:
    value = record.metrics.get(key)
    return value is not None and value <= ceiling


# --------------------------------------------------------------------------
# applying it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """The admitted runs, the rejected ones, and which clause did the rejecting."""

    spec: SelectionSpec
    kept: tuple[RunRecord, ...]
    #: arm -> the first gate the run failed.  First, not all, because the gates
    #: are a funnel: a torso has no metrics, so reporting it as failing "asr >= 1"
    #: as well would double-count it into the performance column.
    rejected: Mapping[str, str]
    #: Every gate name in application order, including ones nothing failed --
    #: a gate that rejected nothing is a finding, and it disappears if the
    #: report is built from the rejections alone.
    gate_names: tuple[str, ...]

    @property
    def considered(self) -> int:
        return len(self.kept) + len(self.rejected)

    def arms(self) -> tuple[str, ...]:
        return tuple(r.arm for r in self.kept)

    def breakdown(self, records: Sequence[RunRecord]) -> list[tuple[str, int, int]]:
        """``(gate, rejected_alone, rejected_in_sequence)`` for each gate.

        The two columns answer different questions and neither substitutes for
        the other.  *Alone* is how many runs this clause would reject if it were
        the only clause -- comparable across arms, and the right number to quote
        when asking whether two arms fail for the same reasons.  *In sequence*
        is how many reached this gate and died at it -- sums to the rejection
        total, and the right number for a funnel.  Where the two diverge, the
        gates overlap; on the astra sample they diverge a lot, because a run
        that blows the token budget also tends to be a run that gets refused.
        """
        gates = gates_for(self.spec)
        alone = {g.name: sum(1 for r in records if not g(r)) for g in gates}
        sequential: dict[str, int] = {g.name: 0 for g in gates}
        for record in records:
            for gate in gates:
                if not gate(record):
                    sequential[gate.name] += 1
                    break
        return [(g.name, alone[g.name], sequential[g.name]) for g in gates]

    def report(self, records: Sequence[RunRecord]) -> str:
        lines = [
            f"selection {self.spec.label or '(unlabelled)'}: "
            f"{len(self.kept)} kept of {self.considered} considered",
            f"{'gate':<44}{'alone':>10}{'in sequence':>14}",
        ]
        for name, alone, sequential in self.breakdown(records):
            lines.append(f"{name:<44}{alone:>10}{sequential:>14}")
        return "\n".join(lines)


def select(records: Iterable[RunRecord], spec: SelectionSpec) -> Selection:
    """Apply ``spec``, keeping the reason each rejected run was rejected."""
    gates = gates_for(spec)
    kept: list[RunRecord] = []
    rejected: dict[str, str] = {}
    for record in records:
        for gate in gates:
            if not gate(record):
                rejected[record.arm] = gate.name
                break
        else:
            kept.append(record)
    return Selection(
        spec=spec,
        kept=tuple(kept),
        rejected=rejected,
        gate_names=tuple(g.name for g in gates),
    )
