"""Trade the per-package classifier: score every package, take the best few.

The model predicts one thing only -- ``P(this package's realized after-cost PnL
is positive)`` -- and everything else is the same deterministic machinery every
rule arm uses.  Strikes come from ``CoordinateBounds.defaults`` because the order
is a five-token open; the package count comes from ``SizeResolver``; the exit is
a fixed age and days-to-expiry rule.  Nothing here predicts a coordinate, a size
or an exit.

**Selection is by rank, not by probability level.**  Candidates are sorted by
``P`` and filled greedily into the environment's own caps.  Ranking is invariant
to miscalibration -- only the ordering has to be right, which is what the AUC
those models were selected on actually measures.  A threshold would make the
arm's turnover a function of how well calibrated the weakest head is.
``threshold`` therefore does one job: decide when to abstain entirely.

**The label is optimistic and the arm cannot be.**  The oracle scored each
package at its *best* exit, chosen with hindsight, so ``P`` answers "could this
package have made money if exited well" and not "will it make money under my
exit rule".  A fixed exit rule cannot recover that, so this arm's PnL is
expected to sit below what its AUC would suggest, and the gap belongs to the
exit rule rather than to the family choice.  Stated here because the two are
easy to conflate and only one of them is what the baseline is testing.

**It runs anonymized.**  Unlike the per-ticker rule baselines, nothing here looks
a name up: a package is scored from the market row in front of it, and the head
is name-independent.  So ``--anonymize`` is safe and is not refused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env.policy import PolicyResponse
from ..env.policy.rules import ObservedState, read_observation
from ..env.spec import EnvConfig
from ..env.statespace import Observation
from .packages import TENOR_INDEX


#: Never open a package the model thinks is more likely to lose than win.
#: Applied *before* the lift ranking, because lift alone would buy a head with a
#: 0.10 base rate at P=0.25 -- three times its base and still a 1-in-4 shot.
ABSOLUTE_FLOOR = 0.5


class ModelArtifactError(RuntimeError):
    """The fitted artifact is absent, stale, or disagrees with this code."""


@dataclass(frozen=True, slots=True)
class FittedPackageModels:
    """Per-head classifiers plus the exact feature contract they were fitted on."""

    models: Mapping[str, Any]
    feature_names: tuple[str, ...]
    threshold: float
    tenors: tuple[str, ...]
    provenance: Mapping[str, Any]
    #: Each head's unconditional positive rate on the training window.
    #: Load-bearing, not diagnostic.
    #:
    #: Measured 2026-09-26, and this is why it exists: a single global cut on raw
    #: ``P`` is a **base-rate sort**.  At tau=0.70 the arm traded exactly the four
    #: highest-base-rate heads -- ``cv b`` .836, ``ol b`` .796, ``dg b`` .778,
    #: ``dv b`` .740, whose AUCs are 0.504-0.553, i.e. chance -- and never once
    #: traded any of the five heads with real signal (``bf r`` .642, ``lg n``
    #: .628, ``ib n`` .623, ``ic n`` .616, ``bf b`` .595), because their base
    #: rates sit below 0.58 so their probabilities never reach 0.70.  The result
    #: was +32.93% of levered bullish beta earned by the packages the model
    #: cannot predict, with Sharpe 0.345 against a standard error of 1.432.
    base_rates: Mapping[str, float] = ()  # type: ignore[assignment]

    @classmethod
    def load(cls, path: Path) -> "FittedPackageModels":
        import pickle

        meta_path = path / "meta.json"
        if not meta_path.is_file():
            raise ModelArtifactError(
                f"{meta_path} is missing. Fit the models first with "
                "scripts/analysis/fit_package_models.py; this policy will not "
                "invent a classifier or a threshold at run time."
            )
        meta = json.loads(meta_path.read_text())
        models: dict[str, Any] = {}
        for head in meta["heads"]:
            blob = path / f"{head.replace(' ', '_')}.pkl"
            if not blob.is_file():
                raise ModelArtifactError(f"{blob} is missing but {head!r} is declared")
            models[head] = pickle.loads(blob.read_bytes())
        base = meta.get("base_rates")
        if not base:
            raise ModelArtifactError(
                f"{meta_path} has no 'base_rates' block. Refit with "
                "scripts/analysis/fit_package_models.py: without per-head base "
                "rates this policy would rank on raw P(positive), which across "
                "heads whose base rates run from 0.10 to 0.84 is a base-rate "
                "sort, and it traded only the no-signal families when it was."
            )
        missing = [h for h in meta["heads"] if h not in base]
        if missing:
            raise ModelArtifactError(f"base_rates absent for {missing}")
        return cls(
            models=models,
            feature_names=tuple(meta["feature_names"]),
            threshold=float(meta["threshold"]),
            tenors=tuple(meta["tenors"]),
            provenance=meta,
            base_rates={h: float(base[h]) for h in meta["heads"]},
        )


def _market_only(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if line.startswith(("T ", "M "))
    )


def _cross_section(
    per_name: Mapping[str, Mapping[str, float | None]], fields: Sequence[str]
) -> dict[str, dict[str, float | None]]:
    out = {name: dict(vals) for name, vals in per_name.items()}
    for field in fields:
        vals = sorted(
            ((n, v[field]) for n, v in per_name.items() if v.get(field) is not None),
            key=lambda kv: kv[1],
        )
        n = len(vals)
        for rank, (name, _) in enumerate(vals):
            out[name][f"{field}_xrank"] = 0.5 if n == 1 else rank / (n - 1)
        for name in per_name:
            out[name].setdefault(f"{field}_xrank", None)
    return out


class PackageModelPolicy:
    """Score every (name, head, tenor); open the best few the caps allow."""

    def __init__(
        self,
        fitted: FittedPackageModels,
        config: EnvConfig,
        *,
        hold_steps: int = 4,
        min_dte: int = 3,
    ) -> None:
        self.fitted = fitted
        self.config = config
        self.name = "package_model"
        self.hold_steps = hold_steps
        self.min_dte = min_dte
        self._opened_at: dict[str, int] = {}
        self._base_fields = tuple(
            f for f in fitted.feature_names
            if not f.endswith("_xrank") and f != "tenor_index"
        )
        if fitted.feature_names[-1] != "tenor_index":
            raise ModelArtifactError(
                "the artifact's last feature must be 'tenor_index'; the policy "
                f"appends it last and the order must match, got "
                f"{fitted.feature_names[-1]!r}"
            )
        unknown = [t for t in fitted.tenors if t not in config.tenor.admitted]
        if unknown:
            raise ModelArtifactError(
                f"the artifact was fitted on tenor buckets this config does not "
                f"admit: {unknown}. Scoring them would rank packages the "
                "environment cannot open."
            )

    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        # ``_opened_at`` deliberately survives, as in RulePolicy: a position
        # held across a month boundary still has an age.
        return None

    # -- features ---------------------------------------------------------

    def _rows(self, state_text: str) -> dict[str, list[float | None]]:
        state = read_observation(Observation(text=_market_only(state_text)))
        per_name = {n: dict(r.values) for n, r in state.market.items()}
        enriched = _cross_section(per_name, self._base_fields)
        return {
            name: [vals.get(f) for f in self.fitted.feature_names[:-1]]
            for name, vals in enriched.items()
        }

    # -- acting -----------------------------------------------------------

    def act(self, observation: Observation) -> PolicyResponse:
        import numpy as np

        state = read_observation(Observation(text=_market_only(observation.text)))
        full = read_observation(observation)
        features = self._rows(observation.text)

        lines = list(self._closes(full))
        closing = {line.split()[1] for line in lines}
        held = [p for p in full.positions if p.position_id not in closing]
        per_name_held: dict[str, int] = {}
        for p in held:
            per_name_held[p.underlying] = per_name_held.get(p.underlying, 0) + 1

        room = self.config.size.max_positions - len(held)
        budget = min(room, self.config.max_orders_per_step - len(lines))
        if budget > 0:
            scored: list[tuple[float, str, str, str]] = []
            for name, base in features.items():
                if any(v is None for v in base):
                    # A name with an unreadable market row is skipped rather
                    # than imputed: the imputer was fitted on the training
                    # window and would supply a confident median here.
                    continue
                for tenor in self.fitted.tenors:
                    vec = np.array(
                        [base + [float(TENOR_INDEX[tenor])]], dtype=float
                    )
                    for head, model in self.fitted.models.items():
                        p = float(model.predict_proba(vec)[0, 1])
                        # Two gates, and they do different jobs.
                        #
                        # The floor is absolute: never open a package the model
                        # thinks is more likely to lose than win, however
                        # flattering its lift.  It is what keeps `bf` out --
                        # base rate 0.10, and the side that would profit there
                        # is a naked short the action space forbids.
                        if p <= ABSOLUTE_FLOOR:
                            continue
                        # The ranking key is lift over this head's OWN base
                        # rate, so heads are comparable.  Raw P is not: it
                        # ranked `cv b` at 0.84 (AUC 0.504) above `ic n` at
                        # 0.55 (AUC 0.616), which is the whole bug.
                        lift = p / self.fitted.base_rates[head]
                        if lift > self.fitted.threshold:
                            scored.append((lift, name, head, tenor))
            scored.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
            cap = self.config.size.max_positions_per_underlying
            for p, name, head, tenor in scored:
                if budget <= 0:
                    break
                if per_name_held.get(name, 0) >= cap:
                    continue
                lines.append(f"O {name} {head} {tenor}")
                per_name_held[name] = per_name_held.get(name, 0) + 1
                budget -= 1

        text = "\n".join(lines) if lines else "H"
        return PolicyResponse(
            text=text,
            model=self.name,
            extra={"threshold": self.fitted.threshold, "scored": len(features)},
        )

    def _closes(self, state: ObservedState) -> list[str]:
        """Age then days-to-expiry. Never a predicted exit."""
        out: list[str] = []
        for position in state.positions:
            opened = self._opened_at.setdefault(position.position_id, state.step_index)
            if (
                state.step_index - opened >= self.hold_steps
                or position.dte <= self.min_dte
            ):
                out.append(f"C {position.position_id}")
        return out
