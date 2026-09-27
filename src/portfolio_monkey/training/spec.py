"""The experiment matrix as data: every arm in Section 5 is a row, not a branch.

WHY THE MATRIX IS DATA.  Section 5 names six baseline bands, three system arms,
three diagnostics, four primary ablations and five secondary ones, and several
of those are swept over transaction-cost multipliers and training seeds.  Written
as code, each arm becomes an ``if`` somewhere, and the set of arms that were
actually run becomes a property of which branches were taken -- unrecoverable
from the output.  Written as data, the matrix is a value: it can be printed,
diffed against the paper, serialized beside the results, and *extended from a
file* without touching this module, which is the specific thing asked for.

THE ONE DISTINCTION THIS MODULE EXISTS TO KEEP STRAIGHT is contribution versus
adaptation, because it is the distinction Section 5 says is "routinely
conflated".  A **contribution** arm freezes an existing decision stream and
replays it under a changed environment; it has a ``frozen_from`` and no training.
An **adaptation** arm retrains the policy inside the changed environment; it has
a ``training`` stage and no ``frozen_from``.  An arm marked both is *two rows*
here, never one row reported twice, because the two have different provenance and
Section 5 requires they never be averaged together.  :func:`validate` refuses a
matrix in which a row claims one and carries the machinery of the other.

WHAT THIS MODULE DOES NOT DO.  It does not implement a single baseline.  A
GARCH(1,1) forecaster and a value-based RL agent are real work and they live
where the work is; here they are a ``PolicySpec`` with a name and parameters.
The separation is what lets the matrix be validated on a laptop with no data:
:func:`validate` can prove that two arms differ, that an ablation actually
ablates, and that no two rows are the same experiment under two names, all
without running anything.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..env.spec import ConfigError, EnvConfig

__all__ = [
    "Arm",
    "ExperimentMatrix",
    "Finding",
    "MatrixError",
    "PolicySpec",
    "Sweep",
    "apply_overrides",
    "findings",
    "load_matrix",
    "paper_matrix",
    "resolve_env",
    "runnable",
    "validate",
]


class MatrixError(ValueError):
    """The experiment matrix describes something that cannot be run as stated."""


# Roles decide whether a row is ranked in Table 1.  Section 5 excludes three
# groups "for reasons of construction rather than of performance", and the
# exclusion has to travel with the arm rather than be re-derived at table time,
# because re-deriving it is how a ceiling ends up ranked first.
ROLES = ("baseline", "system", "control", "ceiling", "reference")

#: A contribution arm freezes decisions and changes the environment under them;
#: an adaptation arm retrains inside the changed environment.  ``""`` is a row
#: that is not an ablation at all.
EVIDENCE = ("", "contribution", "adaptation")


@dataclass(frozen=True)
class PolicySpec:
    """How an arm's decisions are produced.

    ``kind`` is the dispatch key a runner uses to build the thing; ``params`` is
    whatever that kind needs.  Deliberately untyped past that point: a GARCH
    arm's ``(p, q)`` and an LLM arm's decoding temperature have nothing in
    common, and forcing them into one schema would mean editing this file to add
    a baseline, which is exactly what the matrix-as-data split is for.
    """

    kind: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    #: For trained arms: which checkpoint to load.  Left empty in the matrix and
    #: filled at launch, because a matrix that names checkpoints is a matrix
    #: that expires.
    checkpoint: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["params"] = dict(self.params)
        return out


@dataclass(frozen=True)
class Arm:
    """One row of one table.

    ``env_overrides`` are dotted paths into :class:`EnvConfig` -- ``"hedge.enabled"``,
    ``"cost.half_spread_multiplier"`` -- applied to a *base* config supplied at
    resolve time.  Dotted rather than a nested dict so that a row reads as the
    single sentence the paper writes it as, and validated against the dataclass
    rather than against a list kept here, so a field added to ``EnvConfig``
    becomes overridable the same day.
    """

    arm_id: str
    band: str
    label: str
    policy: PolicySpec
    role: str = "baseline"
    ranked: bool = True
    env_overrides: Mapping[str, Any] = field(default_factory=dict)
    #: ``"A1"``..``"A4"`` for the primary ablations, ``"S1"``.. for the secondary
    #: ones, ``""`` for a row that is not an ablation.
    ablation: str = ""
    evidence: str = ""
    #: The arm whose recorded decisions this row replays.  Contribution arms
    #: only, and required for them: a contribution claim with nothing frozen is
    #: an adaptation that forgot to retrain.
    frozen_from: str = ""
    #: ``"base"``, ``"sft"``, ``"sft+rl"``, or ``""`` for an arm that is not a
    #: trained language policy.  Part of the arm's identity, not decoration: two
    #: rows with the same environment and the same policy spec are the same
    #: experiment *unless* they differ here.
    training: str = ""
    #: Another arm whose run this row reports.  Several table rows are the
    #: unablated system seen from a different table -- A1's "full implied
    #: distribution" and A3's "proposed hedge band" are the paper's own system,
    #: and A4 is the "Ours" block under a different metric set.  Saying so is
    #: better than either duplicating the run or dropping the row: the launcher
    #: skips a row with ``reuses``, and the table builder still prints it.
    reuses: str = ""
    #: Training seeds to average over.  Empty for arms with no training.
    seeds: tuple[int, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["policy"] = self.policy.as_dict()
        out["env_overrides"] = dict(self.env_overrides)
        out["seeds"] = list(self.seeds)
        return out


@dataclass(frozen=True)
class Sweep:
    """A parameter varied across every arm it names, expanding one row into many.

    Section 5 sweeps the half-spread multiplier over ``{0, 0.25, 0.5, 1.0}`` and
    reports ablations as a mean over training seeds.  Both are cartesian
    expansions of the same shape, so they are one mechanism rather than two, and
    the expanded arm carries the swept value in its id -- ``..._hs0.25`` -- so
    that a directory listing is still a readable account of what was run.

    ``applies_to`` is a tuple of arm ids, or empty for "every arm".  Restricting
    it matters: sweeping cost over the Cboe reference series would fabricate
    variants of a published index.
    """

    key: str
    values: tuple[Any, ...]
    applies_to: tuple[str, ...] = ()
    #: How the value enters the arm.  ``"env"`` writes it as an
    #: ``env_overrides`` dotted path; ``"policy"`` writes it into
    #: ``policy.params``; ``"seed"`` writes it into ``seeds``.
    target: str = "env"
    #: Short tag used in the expanded arm id.  Defaults to the key's last segment.
    tag: str = ""

    def tag_for(self, value: Any) -> str:
        stem = self.tag or self.key.rsplit(".", 1)[-1]
        return f"{stem}{value}"


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------


def apply_overrides(base: EnvConfig, overrides: Mapping[str, Any]) -> EnvConfig:
    """``base`` with the dotted paths in ``overrides`` replaced.

    Routed through ``EnvConfig.as_dict``/``from_dict`` rather than through
    ``dataclasses.replace`` on nested objects, for one reason: ``from_dict``
    already refuses a key the dataclass does not declare, and that refusal is
    the only thing standing between a typo and a silent no-op.  An ablation arm
    whose override key is misspelled does not fail -- it runs the unablated
    environment under the ablation's name, and every number it produces is
    filed under the wrong row.
    """
    payload: dict[str, Any] = json.loads(json.dumps(base.as_dict()))
    for path, value in overrides.items():
        parts = path.split(".")
        cursor: Any = payload
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise MatrixError(
                    f"env override {path!r}: {part!r} is not a section of EnvConfig"
                )
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise MatrixError(
                f"env override {path!r}: {leaf!r} is not a field of "
                f"{'.'.join(parts[:-1]) or 'EnvConfig'}. Nothing would change, and "
                "the arm would run the unablated environment under the ablation's name."
            )
        cursor[leaf] = value
    try:
        return EnvConfig.from_dict(payload)
    except ConfigError as exc:
        raise MatrixError(f"overrides {dict(overrides)!r} do not decode: {exc}") from exc


def resolve_env(arm: Arm, base: EnvConfig) -> EnvConfig:
    """The environment this arm runs in."""
    return apply_overrides(base, arm.env_overrides)


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentMatrix:
    """An ordered set of arms, plus the sweeps that expand them."""

    arms: tuple[Arm, ...]
    sweeps: tuple[Sweep, ...] = ()
    label: str = ""

    def __iter__(self) -> Iterator[Arm]:
        return iter(self.arms)

    def __len__(self) -> int:
        return len(self.arms)

    def by_id(self, arm_id: str) -> Arm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise MatrixError(f"no arm {arm_id!r} in matrix {self.label or '(unlabelled)'}")

    def select(
        self,
        *,
        bands: Sequence[str] = (),
        ablations: Sequence[str] = (),
        roles: Sequence[str] = (),
        evidence: Sequence[str] = (),
        ranked: bool | None = None,
    ) -> ExperimentMatrix:
        """A sub-matrix, so a campaign can run Band A today and Band D in March."""
        def keep(arm: Arm) -> bool:
            return (
                (not bands or arm.band in bands)
                and (not ablations or arm.ablation in ablations)
                and (not roles or arm.role in roles)
                and (not evidence or arm.evidence in evidence)
                and (ranked is None or arm.ranked is ranked)
            )

        return replace(self, arms=tuple(a for a in self.arms if keep(a)))

    def expand(self) -> ExperimentMatrix:
        """Every arm crossed with every sweep that applies to it.

        An arm no sweep touches comes through unchanged and keeps its id, so
        expanding a matrix with no sweeps is the identity -- which means callers
        can expand unconditionally and a sweep can be added later without
        renaming anything that already ran.
        """
        out: list[Arm] = []
        for arm in self.arms:
            applicable = [
                s for s in self.sweeps if not s.applies_to or arm.arm_id in s.applies_to
            ]
            if not applicable:
                out.append(arm)
                continue
            for combo in itertools.product(*(s.values for s in applicable)):
                out.append(_apply_sweep(arm, applicable, combo))
        return replace(self, arms=tuple(out))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "arms": [a.as_dict() for a in self.arms],
            "sweeps": [asdict(s) for s in self.sweeps],
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def _apply_sweep(arm: Arm, sweeps: Sequence[Sweep], combo: Sequence[Any]) -> Arm:
    overrides = dict(arm.env_overrides)
    params = dict(arm.policy.params)
    seeds = arm.seeds
    tags: list[str] = []
    for sweep, value in zip(sweeps, combo, strict=True):
        tags.append(sweep.tag_for(value))
        if sweep.target == "env":
            overrides[sweep.key] = value
        elif sweep.target == "policy":
            params[sweep.key] = value
        elif sweep.target == "seed":
            seeds = (int(value),)
        else:
            raise MatrixError(f"sweep {sweep.key!r}: unknown target {sweep.target!r}")
    return replace(
        arm,
        arm_id="_".join([arm.arm_id, *tags]),
        env_overrides=overrides,
        policy=replace(arm.policy, params=params),
        seeds=seeds,
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One thing wrong with a matrix, or one thing not yet buildable in it."""

    arm_id: str
    kind: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.kind}] {self.arm_id}: {self.detail}"


#: Kinds of finding.  ``unimplemented`` is the one that is not an error: the
#: paper declares ablations whose environment knob does not exist yet, and a
#: matrix that refused to load until every one of them was built would make the
#: declared experiment list unusable as a plan.  The others are errors, and
#: every one of them is a way the matrix would run and produce numbers filed
#: under a row that does not describe them.
FINDING_KINDS = (
    "duplicate_id",
    "bad_role",
    "bad_evidence",
    "unimplemented",
    "no_op_override",
    "evidence_mismatch",
    "missing_source",
    "collision",
)

#: Findings that do not stop a campaign.  Separated from the rest so that
#: ``validate`` has a defensible default rather than a judgement call at each
#: call site.
TOLERATED = ("unimplemented",)


def findings(matrix: ExperimentMatrix, base: EnvConfig) -> list[Finding]:
    """Everything wrong with ``matrix``, all of it, in one pass.

    All of it, rather than raising on the first: a matrix is edited in batches,
    and a validator that reports one problem per run turns a five-minute fix
    into five runs.  More importantly, the *set* of findings is the useful
    object -- "these six arms need an EnvConfig field that does not exist" is a
    work item, whereas the first of them raised as an exception is a puzzle.

    Six error kinds and one tolerated kind, each corresponding to a way a matrix
    has actually gone wrong rather than to a hypothetical:

    ``duplicate_id``    Two rows, one output directory.
    ``unimplemented``   An override names a field ``EnvConfig`` does not declare.
                        Tolerated: the paper declares ablations ahead of the
                        environment work they need, and saying so is the point.
                        RULED 2026-09-24: for the ten arms that name
                        ``flags.iv_detail``, ``flags.suppress_portfolio_state``,
                        ``flags.suppress_history`` or
                        ``flags.textual_context_tier``, that environment work
                        will not happen -- they are permanently out of scope,
                        not pending.  This finding is the *correct* output for
                        them and a planner must refuse the job rather than run
                        it; do not "fix" it by adding the fields.  See
                        ``docs/experiments.md`` section 6.
    ``no_op_override``  Every path spells a real field and the resolved config
                        fingerprints identically to the base.  This is the
                        nastiest one, because it *runs*: the arm reports the
                        unablated environment under the ablation's name.
    ``evidence_mismatch`` A contribution row that trains, or freezes nothing; an
                        adaptation row that replays, or names no stage.  Section 5
                        says these two are "routinely conflated" and must never be
                        averaged, so the machinery has to match the claim.
    ``missing_source``  A contribution freezes an arm not in the matrix.
    ``collision``       Same resolved environment and same policy under two
                        names, so which one gets reported is arbitrary.
    """
    found: list[Finding] = []
    seen: set[str] = set()
    for arm in matrix.arms:
        if arm.arm_id in seen:
            found.append(Finding(arm.arm_id, "duplicate_id", "two rows share this id"))
        seen.add(arm.arm_id)
        if arm.role not in ROLES:
            found.append(Finding(arm.arm_id, "bad_role", f"{arm.role!r} not one of {ROLES}"))
        if arm.evidence not in EVIDENCE:
            found.append(
                Finding(arm.arm_id, "bad_evidence", f"{arm.evidence!r} not one of {EVIDENCE}")
            )

    base_fingerprint = base.fingerprint()
    resolved: dict[str, str] = {}
    for arm in matrix.arms:
        try:
            resolved[arm.arm_id] = resolve_env(arm, base).fingerprint()
        except MatrixError as exc:
            found.append(Finding(arm.arm_id, "unimplemented", str(exc)))
            continue
        # A row that declares ``reuses`` is *supposed* to resolve to the base:
        # it is the unablated system entered in an ablation table, and the
        # overrides on it are a restatement of what the base already does, kept
        # so the row reads as a complete specification rather than as a blank.
        if arm.env_overrides and not arm.reuses and resolved[arm.arm_id] == base_fingerprint:
            found.append(
                Finding(
                    arm.arm_id,
                    "no_op_override",
                    f"{dict(arm.env_overrides)!r} resolves to the base environment; every "
                    "path spells a real field, so this is not a typo -- it sets values the "
                    "base already holds, and the arm would report the unablated environment "
                    "under its own name",
                )
            )

    ids = {a.arm_id for a in matrix.arms}
    for arm in matrix.arms:
        if arm.reuses and arm.reuses not in ids:
            found.append(Finding(
                arm.arm_id, "missing_source",
                f"reports the run of {arm.reuses!r}, which is not in the matrix"))
        if arm.evidence == "contribution":
            if not arm.frozen_from:
                found.append(Finding(
                    arm.arm_id, "evidence_mismatch",
                    "claims a component contribution but freezes nothing; a contribution "
                    "replays decisions the unablated system already made, and without them "
                    "it is an adaptation that forgot to retrain"))
            elif arm.frozen_from not in ids:
                found.append(Finding(
                    arm.arm_id, "missing_source",
                    f"freezes {arm.frozen_from!r}, which is not in the matrix"))
            if arm.training:
                found.append(Finding(
                    arm.arm_id, "evidence_mismatch",
                    f"claims a contribution and also trains ({arm.training!r}); the two "
                    "answer different questions and must never be averaged"))
        if arm.evidence == "adaptation":
            if arm.frozen_from:
                found.append(Finding(
                    arm.arm_id, "evidence_mismatch",
                    f"claims an adaptation but replays {arm.frozen_from!r}; an adaptation "
                    "retrains inside the modified environment, and replaying frozen "
                    "decisions measures the contribution instead"))
            if not arm.training:
                found.append(Finding(
                    arm.arm_id, "evidence_mismatch",
                    "claims an adaptation but names no training stage"))

    # The signature is environment + policy + TRAINING STAGE.  Leaving the stage
    # out was the first version, and it reported the three "Ours" rows as one
    # experiment: base, SFT and SFT+RL share an environment and a ``PolicySpec``,
    # and differ only in what was done to the weights.  A row that declares
    # ``reuses`` is skipped rather than compared, because it is not a run.
    signatures: dict[tuple[str, str, str], str] = {}
    for arm in matrix.arms:
        if arm.arm_id not in resolved:
            continue  # unimplemented; its environment is not known
        if arm.reuses:
            continue
        signature = (
            resolved[arm.arm_id],
            json.dumps(arm.policy.as_dict(), sort_keys=True),
            arm.training,
        )
        if signature in signatures:
            found.append(Finding(
                arm.arm_id, "collision",
                f"resolves to the same environment and the same policy as "
                f"{signatures[signature]!r}, so they are one experiment under two names "
                "and whichever is reported would be arbitrary"))
        else:
            signatures[signature] = arm.arm_id
    return found


def validate(
    matrix: ExperimentMatrix, base: EnvConfig, *, tolerate: Sequence[str] = TOLERATED
) -> list[Finding]:
    """Raise on any finding whose kind is not tolerated; return the tolerated ones.

    Returning the tolerated findings rather than discarding them is deliberate.
    ``unimplemented`` is the list of environment knobs the declared experiment
    matrix is waiting on, and a caller that never sees it will not build them.
    """
    found = findings(matrix, base)
    fatal = [f for f in found if f.kind not in tolerate]
    if fatal:
        raise MatrixError(
            f"{len(fatal)} problem(s) in matrix {matrix.label or '(unlabelled)'}:\n  "
            + "\n  ".join(str(f) for f in fatal)
        )
    return [f for f in found if f.kind in tolerate]


def runnable(matrix: ExperimentMatrix, base: EnvConfig) -> ExperimentMatrix:
    """The sub-matrix every arm of which can be built against ``base`` today.

    The complement of the ``unimplemented`` findings.  Kept separate from
    :func:`validate` because "is this matrix coherent" and "how much of it can I
    launch this afternoon" are different questions, and answering the second by
    quietly dropping rows inside the first is how an arm goes missing from a
    campaign without anybody noticing.
    """
    blocked = {f.arm_id for f in findings(matrix, base) if f.kind == "unimplemented"}
    return replace(matrix, arms=tuple(a for a in matrix.arms if a.arm_id not in blocked))


# --------------------------------------------------------------------------
# the paper's matrix
# --------------------------------------------------------------------------


def paper_base_config(**overrides: Any) -> EnvConfig:
    """The unablated environment every arm is a modification of.

    VALIDATING AGAINST ``EnvConfig()`` IS WRONG, and wrong in a way that looks
    right.  The dataclass defaults are ``size_rule="nav_fraction"`` and
    ``hedge.enabled=False``; the campaign runs scenario sizing with the hedge on
    the five volatility families.  Check the matrix against the defaults and two
    genuine ablations -- A2's equal-capital arm and A3's no-hedge arm -- come
    back as no-ops, because they set values the *defaults* already hold.  The
    no-op check is base-relative by nature, so the base has to be the real one.

    Stated as a function rather than as a constant because it is not a constant:
    an arm's base is whatever the campaign ran, and a campaign that changes the
    window or the universe passes it here.
    """
    payload: dict[str, Any] = {
        "size.size_rule": "scenario",
        "size.target_scenario_risk": 0.005,
        "size.vol_shock_relative": 0.10,
        "size.nav_fraction": 0.10,
        "size.max_positions": 12,
        "size.max_positions_per_underlying": 3,
        "hedge.enabled": True,
        "hedge.band_rule": "fixed",
        "hedge.delta_band": 0.005,
        "hedge.hedged_families": [
            "butterfly", "long_straddle", "long_strangle",
            "iron_butterfly", "iron_condor",
        ],
        "flags.anonymize": True,
        "max_context_tokens": 32_768,
        "quotes_enabled": True,
        "quote_channel": "text",
        "max_quotes_per_name": 6,
        "grid.decision_sessions": ["PM"],
    }
    payload.update(overrides)
    return apply_overrides(EnvConfig(), payload)


def paper_matrix(
    *,
    seeds: Sequence[int] = (0, 1, 2),
    cost_multipliers: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    student: str = "Qwen3-4B",
    teacher: str = "us.openai.gpt-6-astra",
) -> ExperimentMatrix:
    """Section 5, transcribed.

    Labels are the paper's table rows verbatim so that a generated table and the
    typeset one can be diffed line by line rather than compared by eye.  Nothing
    here is a default anybody should silently inherit: ``seeds``,
    ``cost_multipliers`` and the model names are arguments, and a campaign that
    runs one seed states that by passing one seed rather than by editing a
    constant.

    THE CONTRIBUTION ARMS ALL FREEZE ``sys_sft_rl``.  Section 5's contribution
    definition is "freezes the policy or the trade decisions and changes one
    component downstream of them", and the only decision stream worth freezing
    is the finished system's.  Freezing a baseline's instead would measure what
    the component is worth to a policy that is not the paper's.
    """
    arms: list[Arm] = []

    def add(**kwargs: Any) -> None:
        arms.append(Arm(**kwargs))

    # -- Band A: rule-based overlays, no state input ------------------------
    #
    # REFERENCE ROWS, NOT RANKED (user ruling, 2026-09-24: *"band a and c are
    # completely rule-based / don't need the llm or rl. keep them as reference
    # rows"*).  Three of the four cannot be expressed in the action space at
    # all, and the reason is structural rather than unfinished work:
    #
    #   ``env/actions.py:115``     nine families, none of them a covered call, a
    #                              collar, a put-write or a short straddle;
    #   ``resolvers/contract.py``  ``_TOPOLOGY`` pairs every short leg with a
    #                              long leg of the same right, so a naked short
    #                              cannot be resolved whatever the DSL says;
    #   ``env/actions.py:396``     the ``H`` verb is *Hold*.  No verb trades
    #                              stock; shares exist only as hedge fills.
    #
    # So ``covered_call`` and ``collar`` (a policy-opened equity leg) and
    # ``put_write`` (a naked short put) are computed *outside* the environment,
    # the way Band F's published indices are, and read as magnitude context.
    # ``iron_condor`` is expressible -- ``ic`` is a family -- and is kept here
    # anyway so the band is one kind of row throughout rather than a mixture of
    # in-env and external series under one heading.
    #
    # Being unranked also drops them from the cost sweep, because ``applies_to``
    # is built from ``ranked``.  That is the right consequence rather than an
    # accident: a series computed outside the environment never passes through
    # ``ExecutionModel``, so ``cost.half_spread_multiplier`` would not reach it.
    for name, label in (
        ("covered_call", "Covered call"),
        ("put_write", "Put-write"),
        ("iron_condor", "Iron condor"),
        ("collar", "Collar"),
    ):
        add(
            arm_id=f"a_{name}",
            band="A",
            label=label,
            role="reference",
            ranked=False,
            policy=PolicySpec(kind="rule", name=name),
            notes="fixed structure on a fixed schedule; establishes what a passive "
                  "overlay is worth. Reference: computed outside the action space, "
                  "which admits no equity leg and no naked short",
        )

    # -- Band B: forecast-driven selection ---------------------------------
    add(arm_id="b_mr", band="B", label="MR",
        policy=PolicySpec(kind="forecast", name="mean_reversion"),
        notes="signal on the underlying; orientation follows the sign")
    add(arm_id="b_mom", band="B", label="MOM",
        policy=PolicySpec(kind="forecast", name="tsmom",
                          params={"citation": "moskowitz2012tsmom"}),
        notes="time-series momentum; orientation follows the sign")
    add(arm_id="b_garch", band="B", label="GARCH",
        policy=PolicySpec(kind="forecast", name="garch",
                          params={"p": 1, "q": 1, "citation": "bollerslev1986garch"}),
        notes="long gamma when the forecast exceeds the implied level, short gamma when it does not")
    add(arm_id="b_deepvol", band="B", label="DeepVol",
        policy=PolicySpec(kind="forecast", name="deepvol",
                          params={"citation": "deepvol2024"}),
        notes="dilated causal convolution forecaster, same comparison against implied")

    # -- Band C: published option-return signals ---------------------------
    #
    # Reference rows for the same reason, and with one extra wrinkle worth
    # stating: all three are published as SHORT-volatility strategies.  The
    # straddle is short outright; the two sorts are long-short portfolios whose
    # short leg is the whole result in the source papers.  Respecifying them
    # long-only to fit the action space would not be the published signal, so
    # they are reproduced as published, outside the environment, and never
    # differenced against a ranked row.
    add(arm_id="c_short_straddle", band="C", label=r"Short straddle, $\Delta$-hedged",
        role="reference", ranked=False,
        policy=PolicySpec(kind="signal", name="delta_hedged_short_straddle",
                          params={"citation": "coval2001expected"}),
        notes="reference: naked short, which no family admits")
    add(arm_id="c_iv_hv", band="C", label="IV/HV sort",
        role="reference", ranked=False,
        policy=PolicySpec(kind="signal", name="iv_hv_ratio_sort",
                          params={"citation": "goyal2009crosssection"}),
        notes="reference: the published sort is long-short and the short leg carries it")
    add(arm_id="c_ivol", band="C", label="Idiosyncratic-vol sort",
        role="reference", ranked=False,
        policy=PolicySpec(kind="signal", name="idiosyncratic_vol_sort",
                          params={"citation": "cao2013crosssection"}),
        notes="reference: same, and the sort is formed on a quantity the state "
              "space does not carry")

    # -- Band D: supervised predictors on the same state -------------------
    # The target is per-candidate realized after-cost profit at a FIXED horizon,
    # never at the maximizing exit: a label taken at the best realized exit is
    # clairvoyant, and the predictor it produces is one the environment cannot
    # reproduce.  Carried as a parameter so the choice is visible in the matrix.
    for name, label in (("linear", "Linear"), ("gbdt", "GBDT"), ("mlp", "MLP"), ("lstm", "LSTM")):
        add(arm_id=f"d_{name}", band="D", label=label,
            policy=PolicySpec(kind="supervised", name=name,
                              params={"target": "realized_after_cost_pnl",
                                      "label": "value_regression",
                                      "horizon": "fixed"}))

    # -- Band E: non-linguistic RL -----------------------------------------
    add(arm_id="e_rl", band="E", label="RL",
        policy=PolicySpec(kind="value_rl", name="dqn",
                          params={"action_space": "flat_index(family,orientation,tenor)",
                                  "inputs": "numeric_state_only"}),
        training="rl", seeds=tuple(seeds),
        notes="same reward, same state, same resolvers; bounds from below what the "
              "language interface is worth with post-training held fixed")

    # -- Ours ---------------------------------------------------------------
    add(arm_id="sys_base", band="ours", label="Qwen", role="system",
        policy=PolicySpec(kind="llm", name=student), training="base")
    add(arm_id="sys_sft", band="ours", label=r"Qwen $+$ SFT", role="system",
        policy=PolicySpec(kind="llm", name=student), training="sft", seeds=tuple(seeds))
    add(arm_id="sys_sft_rl", band="ours", label=r"Qwen $+$ SFT $+$ RL", role="system",
        policy=PolicySpec(kind="llm", name=student), training="sft+rl", seeds=tuple(seeds))

    # -- Controls and ceiling, not ranked ----------------------------------
    add(arm_id="ctl_random", band="controls", label="Random control (matched)",
        role="control", ranked=False,
        policy=PolicySpec(kind="random", name="matched",
                          params={"match": ["n_opens", "family_distribution", "size_distribution"],
                                  "matched_to": "sys_sft_rl"}))
    add(arm_id="ctl_teacher", band="controls", label="Teacher, zero-shot",
        role="control", ranked=False,
        policy=PolicySpec(kind="llm", name=teacher), training="base",
        notes="what the cold-start signal is worth before any post-training")
    add(arm_id="ctl_oracle", band="controls", label="Oracle",
        role="ceiling", ranked=False,
        policy=PolicySpec(kind="oracle", name="clairvoyant"),
        notes="sees the realized path; diagnostic only -- an arm far below a low "
              "ceiling is a policy problem, a low ceiling is an environment problem")

    # -- Band F: Cboe reference series, not ranked, never swept ------------
    for code, label in (("BXM", "BXM (buy-write)"), ("PUT", "PUT (put-write)"),
                        ("CLL", "CLL (collar)"), ("CNDR", "CNDR (condor)"),
                        ("BFLY", "BFLY (butterfly)")):
        add(arm_id=f"f_{code.lower()}", band="F", label=label,
            role="reference", ranked=False,
            policy=PolicySpec(kind="external", name=code,
                              params={"citation": "cboe2026benchmarks"}),
            notes="index options, differing capital base, published daily; context for "
                  "magnitude only and never differenced on a sub-daily grid")

    # ---------------------------------------------------------------------
    # A1: option-implied distribution information (contribution and adaptation)
    # ---------------------------------------------------------------------
    # The four arms are nested inputs on identical states.  The third -- skew and
    # tails removed, volatility level retained -- is the one that speaks to
    # strategy SHAPE rather than to volatility timing, and it is the arm whose
    # null result would leave the paper's central mechanism unsupported.
    a1 = (
        ("full", "Full implied distribution", {}),
        ("summary", r"ATM IV $+$ smile summaries", {"flags.iv_detail": "atm_plus_smile"}),
        ("no_shape", "No skew or tail features", {"flags.iv_detail": "level_only"}),
        ("none", "No option-implied input", {"flags.iv_detail": "none"}),
    )
    for name, label, overrides in a1:
        base_id = f"a1_{name}"
        # The "full" arm is the unablated system in both columns, so it is a
        # table row over a run that already exists rather than two more runs.
        reuse = "sys_sft_rl" if not overrides else ""
        add(arm_id=f"{base_id}_c", band="ablation", label=label, role="baseline",
            ablation="A1", evidence="contribution", frozen_from="sys_sft_rl",
            env_overrides=overrides, reuses=reuse,
            policy=PolicySpec(kind="replay", name=base_id))
        add(arm_id=f"{base_id}_a", band="ablation", label=label, role="baseline",
            ablation="A1", evidence="adaptation", training="sft+rl", seeds=tuple(seeds),
            env_overrides=overrides, reuses=reuse,
            policy=PolicySpec(kind="llm", name=student),
            notes="a retrained policy may recover the same return through a different "
                  "family mix, which is itself the finding")

    # ---------------------------------------------------------------------
    # A2: sizing resolver -- contribution only, selection frozen by construction
    # ---------------------------------------------------------------------
    # NESTED BY CONSTRUCTION: the scenario arm is the premium arm plus exactly
    # one limit, so the difference isolates the scenario term instead of
    # confounding it with a change of scale.  Two disjoint rules calibrated at
    # arbitrary budgets would not have that property, which is why the override
    # changes `size_rule` and nothing else about the capital base.
    for name, label, rule in (
        ("proposed", "Proposed sizing", "scenario"),
        ("equal_capital", "Equal capital", "nav_fraction"),
        ("equal_risk", "Equal risk", "full"),
    ):
        add(arm_id=f"a2_{name}", band="ablation", label=label,
            ablation="A2", evidence="contribution", frozen_from="sys_sft_rl",
            reuses="sys_sft_rl" if name == "proposed" else "",
            env_overrides={"size.size_rule": rule},
            policy=PolicySpec(kind="replay", name=f"a2_{name}"),
            notes="report return beside drawdown, utilization and realized exposures, "
                  "plus binding-limit shares, so a capital-constrained arm is visible as one")

    # ---------------------------------------------------------------------
    # A3: hedging rule (contribution and adaptation)
    # ---------------------------------------------------------------------
    a3 = (
        ("band", "Proposed hedge band", {"hedge.enabled": True, "hedge.band_rule": "fixed"}),
        ("daily_full", "Daily full delta hedge",
         {"hedge.enabled": True, "hedge.band_rule": "fixed", "hedge.delta_band": 0.0}),
        ("none", "No hedge", {"hedge.enabled": False}),
    )
    for name, label, overrides in a3:
        # As with A1, the proposed band IS the system; it is a row, not a run.
        reuse = "sys_sft_rl" if name == "band" else ""
        add(arm_id=f"a3_{name}_c", band="ablation", label=label,
            ablation="A3", evidence="contribution", frozen_from="sys_sft_rl",
            env_overrides=overrides, reuses=reuse,
            policy=PolicySpec(kind="replay", name=f"a3_{name}"),
            notes="decompose into option PnL, hedge PnL and transaction cost -- that is "
                  "what separates capturing realized movement from retaining exposure "
                  "and both from merely trading less")
        add(arm_id=f"a3_{name}_a", band="ablation", label=label,
            ablation="A3", evidence="adaptation", training="sft+rl", seeds=tuple(seeds),
            env_overrides=overrides, reuses=reuse,
            policy=PolicySpec(kind="llm", name=student),
            notes="a policy that knows it will not be hedged should select different "
                  "families; whether it does is the test of whether portfolio state is used")

    # ---------------------------------------------------------------------
    # A4: training stage -- adaptation by definition
    # ---------------------------------------------------------------------
    # These are the same three systems as the "Ours" block, entered again under
    # A4 because Table 2 reports a different metric set (reward dispersion,
    # invalid-order rate, and the change in the family/orientation/tenor mix).
    # They are NOT separate runs, and ``reuses`` is how that is said.
    for name, label, stage in (("base", "Base model", "base"),
                               ("sft", r"$+$ SFT", "sft"),
                               ("sft_rl", r"$+$ SFT $+$ RL", "sft+rl")):
        add(arm_id=f"a4_{name}", band="ablation", label=label, role="system",
            ablation="A4", evidence="adaptation", training=stage,
            seeds=() if stage == "base" else tuple(seeds),
            reuses=f"sys_{name}",
            policy=PolicySpec(kind="llm", name=student),
            notes="post-training can raise return by concentrating into a single family; "
                  "the family/orientation/tenor mix is reported so that is not read as "
                  "improved selection")

    # ---------------------------------------------------------------------
    # Secondary ablations (appendix)
    # ---------------------------------------------------------------------
    for name, label, overrides in (
        ("full_menu", "Full nine-family menu", {}),
        ("directional", "Directional families only",
         {"admitted_families": ["outright", "debit_vertical", "credit_vertical",
                                "defined_risk_reversal"]}),
        ("volatility", "Volatility families only",
         {"admitted_families": ["long_straddle", "long_strangle", "iron_butterfly",
                                "iron_condor", "butterfly"]}),
        ("restricted", "Restricted standard menu",
         {"admitted_families": ["long_straddle", "iron_condor"]}),
    ):
        add(arm_id=f"s1_{name}", band="ablation", label=label,
            ablation="S1", evidence="contribution", frozen_from="sys_sft_rl",
            env_overrides=overrides,
            policy=PolicySpec(kind="replay", name=f"s1_{name}"),
            notes="speaks most directly to prior RL work on option trading, which "
                  "occupies the single-family single-tenor corner of this grid")

    add(arm_id="s2_no_portfolio", band="ablation", label="No existing-position information",
        ablation="S2", evidence="adaptation", training="sft+rl", seeds=tuple(seeds),
        env_overrides={"flags.suppress_portfolio_state": True},
        policy=PolicySpec(kind="llm", name=student),
        notes="tests whether positions are coordinated rather than accumulated")

    add(arm_id="s3_no_history", band="ablation", label="Current snapshot only",
        ablation="S3", evidence="adaptation", training="sft+rl", seeds=tuple(seeds),
        env_overrides={"flags.suppress_history": True},
        policy=PolicySpec(kind="llm", name=student),
        notes="tests whether changes in market conditions matter beyond the current level")

    for name, label, overrides in (
        ("none", "No textual context", {"flags.suppress_textual_context": True}),
        ("headline", "Headline and metadata only", {"flags.textual_context_tier": "headline"}),
        ("validated", "Validated structured summaries", {"flags.textual_context_tier": "full"}),
    ):
        add(arm_id=f"s4_{name}", band="ablation", label=label,
            ablation="S4", evidence="adaptation", training="sft+rl", seeds=tuple(seeds),
            env_overrides=overrides,
            policy=PolicySpec(kind="llm", name=student),
            notes="controlled rather than observational, because context is not randomly "
                  "assigned across states")

    sweeps = (
        # Every ranked arm faces the same cost sweep.  Bands A, C and F are
        # excluded by ``applies_to``, which is built from ``ranked``: none of
        # them runs inside the environment, so ``half_spread_multiplier`` never
        # reaches their fills, and sweeping a published index would fabricate
        # variants of a series somebody else computed.
        Sweep(
            key="cost.half_spread_multiplier",
            values=tuple(cost_multipliers),
            applies_to=tuple(a.arm_id for a in arms if a.ranked),
            tag="hs",
        ),
    )
    return ExperimentMatrix(arms=tuple(arms), sweeps=sweeps, label="paper.section5")


# --------------------------------------------------------------------------
# loading a matrix from a file
# --------------------------------------------------------------------------


def load_matrix(path: Path | str, *, base: ExperimentMatrix | None = None) -> ExperimentMatrix:
    """A matrix from JSON, optionally patching one already in hand.

    THE POINT OF THE PATCH MODE.  Adding a variation should not require editing
    this file, and neither should it require restating forty arms to change one.
    So a file may carry ``"arms"`` (rows, which replace by id or append) and
    ``"drop"`` (ids to remove), and what comes back is ``base`` with those
    applied.  Without ``base`` the file is the whole matrix.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = {a.arm_id: a for a in (base.arms if base else ())}
    order = [a.arm_id for a in (base.arms if base else ())]
    for raw in payload.get("arms", ()):
        arm = _arm_from_dict(raw)
        if arm.arm_id not in arms:
            order.append(arm.arm_id)
        arms[arm.arm_id] = arm
    for arm_id in payload.get("drop", ()):
        if arm_id not in arms:
            raise MatrixError(
                f"{path}: cannot drop {arm_id!r}, it is not in the matrix. A drop that "
                "silently matches nothing leaves the arm in the campaign."
            )
        del arms[arm_id]
        order.remove(arm_id)
    sweeps = tuple(
        Sweep(
            key=str(s["key"]),
            values=tuple(s["values"]),
            applies_to=tuple(s.get("applies_to", ())),
            target=str(s.get("target", "env")),
            tag=str(s.get("tag", "")),
        )
        for s in payload.get("sweeps", ())
    ) or (base.sweeps if base else ())
    return ExperimentMatrix(
        arms=tuple(arms[i] for i in order),
        sweeps=sweeps,
        label=str(payload.get("label", base.label if base else "")),
    )


def _arm_from_dict(raw: Mapping[str, Any]) -> Arm:
    policy = raw.get("policy") or {}
    known = {f for f in Arm.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise MatrixError(
            f"arm {raw.get('arm_id', '?')!r}: {', '.join(unknown)} is not a field of Arm. "
            "Ignoring it would drop a property somebody wrote down on purpose."
        )
    return Arm(
        arm_id=str(raw["arm_id"]),
        band=str(raw["band"]),
        label=str(raw.get("label", raw["arm_id"])),
        policy=PolicySpec(
            kind=str(policy.get("kind", "")),
            name=str(policy.get("name", "")),
            params=dict(policy.get("params", {})),
            checkpoint=str(policy.get("checkpoint", "")),
        ),
        role=str(raw.get("role", "baseline")),
        ranked=bool(raw.get("ranked", True)),
        env_overrides=dict(raw.get("env_overrides", {})),
        ablation=str(raw.get("ablation", "")),
        evidence=str(raw.get("evidence", "")),
        frozen_from=str(raw.get("frozen_from", "")),
        training=str(raw.get("training", "")),
        reuses=str(raw.get("reuses", "")),
        seeds=tuple(int(s) for s in raw.get("seeds", ())),
        notes=str(raw.get("notes", "")),
    )
