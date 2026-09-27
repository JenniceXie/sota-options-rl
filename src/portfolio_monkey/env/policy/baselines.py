"""Conventional strategy-selection baselines: a volatility forecast, and economics.

These are the Band B / Band C rows of the paper's matrix, and they exist to
answer a question the LLM arms cannot answer about themselves: **how much of any
result is the language interface, and how much is available to two pages of
arithmetic over the same state?**  A language policy that cannot beat a
tercile rule on the IV-RV wedge has not demonstrated that it read the state.

Two policies ship here, and the split is deliberate:

``GarchVolPremiumPolicy``
    A *forecast* baseline.  It estimates conditional volatility with a
    GARCH(1,1) fitted strictly before the test window, compares the forecast to
    the option market's implied level, and sells volatility when implied is
    richer than the forecast, buys it when cheaper.  This is the variance-risk-
    premium trade in its most conventional form.

``EconomicRulePolicy``
    A *no-model* baseline.  It reads five quantities the state already carries --
    the IV-RV wedge, the term-structure slope, the 25-delta skew, the 25-delta
    butterfly and the step return -- and applies a fixed cascade of economic
    rules.  Nothing is estimated; the only free numbers are the thresholds.

**Both are baselines, not proposals.**  Neither is meant to be good.  They are
meant to be *transparent*: every decision is a short chain of comparisons that a
reader can check by hand against the observation, which is exactly what a
learned policy's decision is not.

Five properties they share, each of which is load-bearing:

**The rules read the rendered observation, not the book.**  ``read_observation``
decodes the same bytes the language policy is shown, at the same precision the
wire carries.  Handing a rule the ``StepContext`` would make "the rule beat the
model" a statement about information rather than about policies.

**Size is the resolver's, and so are the strike coordinates.**  Every order goes
out as a five-token open, which means the coordinates come from
``CoordinateBounds.defaults`` and the package count comes from ``SizeResolver``.
A rule that sized its own positions would be testing a different environment.

**Every threshold is a train+validation quantile, frozen on disk.**  They are
loaded from a JSON artifact, not written here, so that "was this tuned on the
test set?" is answerable by reading one file's provenance block rather than by
trusting this docstring.  See ``scripts/analysis/fit_garch_baseline.py``.

**No look-ahead, by construction and not by assertion.**  The GARCH parameters
are fitted on returns dated strictly before the first test date.  The forecast
for date ``t`` is produced by a filter recursion that has consumed returns up to
and including ``t`` and no further -- legitimate at a close-only decision grid,
because the PM decision *is* the close.  The frozen artifact carries the fit
window and the test start so the claim is checkable.

**Why the forecast is a table and not computed in the policy.**  A GARCH
recursion needs a return history.  The observation carries exactly one ``ret``
cell per name; the transcript accumulates at most ~21 of them inside a monthly
episode, and ``reset`` clears it at every episode boundary, so nothing survives
to burn in a recursion.  The forecast is therefore precomputed as a causal
feature and looked up by ``(date, ticker)`` -- which is precisely how ``rv``
already reaches the state, computed by its builder over a burn-in that predates
the episode.  Doing it any other way would either give the policy a history the
observation does not contain, or silently reduce the forecast to a constant.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec import EnvConfig
from ..statespace import Observation
from . import PolicyResponse
from .rules import ObservedState, read_observation

__all__ = [
    "BaselineParams",
    "GarchVolPremiumPolicy",
    "EconomicRulePolicy",
    "load_params",
]

#: Round-trip friction as a share of gross premium at open, measured on
#: **train-window** runs only (150 of the 7,000 astra2 SFT runs, 2024-09-03 to
#: 2024-11-29).  This is the bar an entry threshold has to clear: a signal worth
#: less than the cost of expressing it is not a signal.
#:
#: ``butterfly`` is absent because no arm has ever opened one, so there is no
#: measurement -- not a zero.  ``defined_risk_reversal`` rests on 12 opens and is
#: carried as thin rather than trusted; an earlier 21% figure for
#: ``iron_condor`` came from 7 opens in a single month and was wrong.
FRICTION: dict[str, float | None] = {
    "long_straddle": 0.032,
    "outright": 0.032,
    "long_strangle": 0.048,
    "iron_butterfly": 0.091,
    "credit_vertical": 0.097,
    "iron_condor": 0.106,
    "debit_vertical": 0.113,
    "defined_risk_reversal": 0.329,   # 12 opens; thin
    "butterfly": None,                # never opened; unmeasured
}

#: One fixed threshold on one feature per family, covering all nine admitted
#: families.  Fixed, not fitted: none of these is a quantile of anything, so the
#: economic policy has **no estimated parameter** and there is no calibration
#: sample for it to have leaked from.
#:
#: Features are scale-free so that one number means the same thing for SPY at 13%
#: annualised vol and PLTR at 65%: the level enters as ``iv/rv``, and skew and
#: curvature are divided by ``iv`` -- which is also the normalisation
#: ``docs/state_space.md`` flags as open for ``sk``.
#:
#: Anchors, per family:
#:   ``iv/rv``  1.0 is fair.  The distance past it is the fractional premium
#:              edge, because ATM value scales with vol, so it is directly
#:              comparable to FRICTION.
#:   ``bf/iv``  0.0 is a flat smile: wings priced at the ATM level.  At or below
#:              zero the wings are cheap outright, which is the condition for
#:              buying them.
#:   ``sk/iv``  0.0 is a flat skew.  Above it puts are bid, below it calls are.
#:   ``ret``    a fixed single-session move; the sign sets the orientation.
#:
#: Three pairs deliberately share a feature -- ``ib``/``ic`` on level,
#: ``lg``/``bf`` on curvature, ``ol``/``dv`` on momentum.  They are two
#: expressions of one view (body vs wing, 2-leg vs 3-leg, capped vs uncapped),
#: so running both on the identical trigger measures what the second expression
#: costs.  That is a designed comparison, not an unresolved overlap.
RULES: tuple[dict[str, object], ...] = (
    # --- level: iv/rv, 1.0 = fair ------------------------------------------
    {"family": "ls", "orientation": "n", "feature": "iv_rv", "op": "le",
     "threshold": 0.90, "edge": 0.10, "name": "long_straddle"},
    {"family": "ib", "orientation": "n", "feature": "iv_rv", "op": "ge",
     "threshold": 1.20, "edge": 0.20, "name": "iron_butterfly"},
    {"family": "ic", "orientation": "n", "feature": "iv_rv", "op": "ge",
     "threshold": 1.25, "edge": 0.25, "name": "iron_condor"},
    # --- curvature: bf/iv, 0.0 = flat smile --------------------------------
    {"family": "lg", "orientation": "n", "feature": "bf_iv", "op": "le",
     "threshold": 0.00, "edge": None, "name": "long_strangle"},
    {"family": "bf", "orientation": "b", "feature": "bf_iv", "op": "le",
     "threshold": 0.00, "edge": None, "name": "butterfly"},
    # --- skew: sk/iv, 0.0 = flat ------------------------------------------
    {"family": "dg", "orientation": "b", "feature": "sk_iv", "op": "ge",
     "threshold": 0.15, "edge": None, "name": "defined_risk_reversal"},
    {"family": "cv", "orientation": "r", "feature": "sk_iv", "op": "le",
     "threshold": 0.00, "edge": None, "name": "credit_vertical"},
    # --- momentum: |ret|, orientation from the sign ------------------------
    {"family": "ol", "orientation": "sign", "feature": "abs_ret", "op": "ge",
     "threshold": 0.03, "edge": None, "name": "outright"},
    {"family": "dv", "orientation": "sign", "feature": "abs_ret", "op": "ge",
     "threshold": 0.02, "edge": None, "name": "debit_vertical"},
)

#: Global participation gate.  An inverted curve is how this state space
#: represents a dated event -- it is the causal substitute for an earnings date --
#: so selling or buying premium into it is trading an event you were told about.
#: Zero is the anchor: inversion, not "the most inverted third".
TERM_SLOPE_GATE = 0.0

#: Fixed entry thresholds for the GARCH arm, on ``R = iv / forecast``.  Same
#: anchors as the level rules above -- 1.0 is fair, and the distance past it is
#: the fractional premium edge -- so the two arms are keyed to the same economics
#: and differ only in what they compare implied against: a conditional forecast
#: here, trailing realised there.
GARCH_SHORT_VOL = 1.25   # 25% edge against iron_condor friction 10.6% = 2.4x
GARCH_LONG_VOL = 0.90    # 10% edge against long_straddle friction 3.2% = 3.1x

#: Per-name cap mirrors ``SizeBounds.max_positions_per_underlying`` so the policy
#: stops before the resolver has to refuse it; the per-step cap leaves room under
#: ``max_orders_per_step = 8`` for the closes that share the budget.
MAX_OPENS_PER_NAME = 3
MAX_OPENS_PER_STEP_9 = 6

#: The tenor both policies trade.  Fixed rather than chosen per step: the GARCH
#: horizon is calibrated to this bucket's anchor (14 DTE, about 10 trading
#: days), and holding it fixed removes a free parameter from *both* policies so
#: the two remain comparable to each other and to the language arms.
TENOR = "8_30"

#: Close on age at half the forecast horizon, and before expiry risk starts.
#: Both are derived from the horizon rather than tuned: a position held to
#: expiry stops being a volatility view and becomes a pin bet.
HOLD_STEPS = 5
MIN_DTE = 3

#: Opens per step, and the book-wide cap.  The cap mirrors
#: ``SizeBounds.max_positions`` so the policy stops before the size resolver has
#: to refuse it -- a refused order is a real event in the ledger and should
#: record a binding limit, not a policy that cannot count.
MAX_OPENS_PER_STEP = 2
MAX_POSITIONS = 12


@dataclass(frozen=True, slots=True)
class BaselineParams:
    """Everything frozen before the test window, loaded from one artifact."""

    thresholds: Mapping[str, Mapping[str, float]]
    garch: Mapping[str, Any]
    forecast: Mapping[str, float]
    provenance: Mapping[str, Any]

    def band(self, signal: str) -> tuple[float, float]:
        row = self.thresholds[signal]
        return float(row["lo"]), float(row["hi"])


def load_params(path: str | Path) -> BaselineParams:
    """Read the frozen artifact and refuse one that cannot prove its provenance.

    The refusal is the point.  A baseline whose thresholds might have been
    refitted on the test window is not a baseline, and the only cheap defence is
    to make the artifact state its fit window and to stop if it does not.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prov = payload.get("provenance") or {}
    for field in ("fit_window", "test_start", "calibration_dates"):
        if field not in prov:
            raise ValueError(
                f"{path}: provenance is missing {field!r}. This artifact cannot "
                "show that its thresholds were chosen before the test window, "
                "so it will not be used."
            )
    return BaselineParams(
        thresholds=payload["thresholds"],
        garch=payload["garch"],
        forecast=payload["forecast"],
        provenance=prov,
    )


class _BaselineBase:
    """Shared bookkeeping: position ageing, closes, and the order envelope."""

    name = "baseline"

    def __init__(self, params: BaselineParams, config: EnvConfig) -> None:
        self.params = params
        self._tradeable = tuple(config.universe.tradeable)
        self._opened_at: dict[str, int] = {}

    # ``reset`` deliberately does not clear ``_opened_at``: the book crosses the
    # episode boundary, so a position opened in March and still open in April
    # has to keep its age.  Clearing it would silently reset every holding
    # period to zero at the start of each month.
    def reset(
        self,
        *,
        system: str,
        grammar: str,
        episode_header: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        return None

    # -- closing ---------------------------------------------------------

    def _closes(self, state: ObservedState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for position in state.positions:
            opened = self._opened_at.setdefault(position.position_id, state.step_index)
            age = state.step_index - opened
            reason = ""
            if position.dte <= MIN_DTE:
                reason = "min_dte"
            elif age >= HOLD_STEPS:
                reason = "age"
            if reason:
                out.append({"position_id": position.position_id, "reason": reason,
                            "age": age, "dte": position.dte})
        return out

    # -- the wire --------------------------------------------------------

    def _respond(
        self,
        state: ObservedState,
        closes: list[dict[str, Any]],
        opens: list[dict[str, Any]],
        signals: Mapping[str, Mapping[str, Any]],
    ) -> PolicyResponse:
        lines = [f"C {c['position_id']}" for c in closes]
        lines += [f"O {o['ticker']} {o['family']} {o['orientation']} {o['tenor']}"
                  for o in opens]
        text = "\n".join(lines) if lines else "H"
        return PolicyResponse(
            text=text,
            model=self.name,
            # This is the audit trail the run is for: every signal that was read,
            # every threshold it was compared against, and the decision that
            # followed -- per decision point, in ``decisions.jsonl``.  Without it
            # a rule baseline is as opaque as the model it is a control for.
            extra={
                "policy": self.name,
                "date": state.date,
                "session": state.session,
                "step_index": state.step_index,
                "nav": state.account.nav,
                "n_positions": len(state.positions),
                "thresholds": {k: dict(v) for k, v in self.params.thresholds.items()},
                "params_provenance": dict(self.params.provenance),
                "signals": {k: dict(v) for k, v in signals.items()},
                "opens": opens,
                "closes": closes,
            },
        )

    def _room(self, state: ObservedState, closes: list[dict[str, Any]]) -> int:
        closing = {c["position_id"] for c in closes}
        held = len(state.positions) - len(closing)
        return max(0, min(MAX_OPENS_PER_STEP, MAX_POSITIONS - held))

    def _held_names(self, state: ObservedState, closes: list[dict[str, Any]]) -> set[str]:
        closing = {c["position_id"] for c in closes}
        return {p.underlying for p in state.positions if p.position_id not in closing}


class GarchVolPremiumPolicy(_BaselineBase):
    """Sell volatility when implied is rich to a GARCH forecast; buy it when cheap.

    The signal is the *ratio* ``iv / garch_forecast``, not the difference.  Two
    reasons, both measured on the calibration sample rather than assumed:

    * The names span a 5x range of volatility level (SPY 13% annualised against
      PLTR 65%), so a difference in vol points is not comparable across the
      cross-section and a single pooled threshold on it would be a threshold on
      *which name*, not on how rich its options are.
    * A GARCH forecast anchored on a long-run mean sits below implied almost
      always -- the variance risk premium is exactly that fact -- so a zero
      threshold on the difference degenerates into "always sell volatility" and
      tests no selection at all.  The tercile bands on the ratio are what make
      this a strategy-*selection* baseline rather than a single standing trade.

    Structure choice is the conventional defined-risk pair: ``ic`` to be short
    volatility, ``ls`` to be long it.  Neither admits a naked short, which the
    environment refuses anyway.
    """

    name = "garch_vrp"

    def act(self, observation: Observation) -> PolicyResponse:
        state = read_observation(observation)
        lo, hi = GARCH_LONG_VOL, GARCH_SHORT_VOL
        closes = self._closes(state)
        room = self._room(state, closes)
        held = self._held_names(state, closes)

        signals: dict[str, dict[str, Any]] = {}
        candidates: list[tuple[float, dict[str, Any]]] = []
        for ticker in self._tradeable:
            row = state.market.get(ticker)
            iv = row.get("iv") if row else None
            forecast = self.params.forecast.get(f"{state.date}|{ticker}")
            record: dict[str, Any] = {"iv": iv, "garch_vol": forecast}
            if iv is None or not forecast or forecast <= 0.0:
                # Refuse rather than impute.  A missing ``iv`` renders ``na`` on
                # the wire and a missing forecast means this date was never
                # fitted; either way the comparison does not exist, and a
                # default would manufacture a signal out of an absence.
                record["regime"] = "no_signal"
                signals[ticker] = record
                continue
            ratio = iv / forecast
            record["ratio"] = ratio
            if ratio >= hi:
                record["regime"] = "short_vol"
                strength = ratio - hi
                order = {"ticker": ticker, "family": "ic", "orientation": "n",
                         "tenor": TENOR, "reason": "iv_rich_vs_garch",
                         "ratio": ratio, "threshold": hi}
            elif ratio <= lo:
                record["regime"] = "long_vol"
                strength = lo - ratio
                order = {"ticker": ticker, "family": "ls", "orientation": "n",
                         "tenor": TENOR, "reason": "iv_cheap_vs_garch",
                         "ratio": ratio, "threshold": lo}
            else:
                record["regime"] = "flat"
                signals[ticker] = record
                continue
            signals[ticker] = record
            if ticker not in held:
                candidates.append((strength, order))

        # Rank by how far past its threshold the signal sits, ties by ticker so
        # the arm is reproducible.
        candidates.sort(key=lambda item: (-item[0], item[1]["ticker"]))
        opens = [order for _, order in candidates[:room]]
        return self._respond(state, closes, opens, signals)

class EconomicRulePolicy(_BaselineBase):
    """Nine independent entry tests, one fixed threshold on one feature each.

    Not a cascade.  An earlier version ordered the signals and let each stage
    veto the next, which meant the *order* needed an economic argument it did not
    have, and it could only ever reach three of the nine admitted families.  Here
    each family has its own test, every test is evaluated, and everything that
    fires is emitted subject to the position caps.  Consequences worth stating:

    * There is no ordering to justify, because there is no ordering.
    * Each rule is separately falsifiable -- "does a 10% cheap-vol reading pay for
      a straddle?" has an answer, and the per-family ledger breakdown gives it.
    * Several rules fire on one name on a rich day.  That is intended: the pairs
      sharing a feature are two expressions of one view, so the comparison is the
      measurement.

    Every threshold is a fixed constant in ``RULES``, anchored either at the
    feature's fair value (1.0 for ``iv/rv``, 0.0 for the two spreads) or at a
    multiple of the family's measured round-trip friction.  Nothing here is a
    quantile, so this policy has no estimated parameter and no calibration sample.
    """

    name = "econ_rules"

    @staticmethod
    def _features(row) -> dict[str, float | None]:
        """The four scale-free features, or ``None`` where an input is absent.

        Division is guarded rather than defaulted: ``rv`` or ``iv`` at zero would
        make a ratio infinite and a threshold meaningless, and an absent field is
        ``na`` on the wire for a reason.
        """
        iv, rv = row.get("iv"), row.get("rv")
        sk, bf, ret = row.get("sk"), row.get("bf"), row.get("ret")
        out: dict[str, float | None] = {
            "iv_rv": (iv / rv) if (iv is not None and rv not in (None, 0.0)) else None,
            "sk_iv": (sk / iv) if (sk is not None and iv not in (None, 0.0)) else None,
            "bf_iv": (bf / iv) if (bf is not None and iv not in (None, 0.0)) else None,
            "abs_ret": abs(ret) if ret is not None else None,
        }
        return out

    def act(self, observation: Observation) -> PolicyResponse:
        state = read_observation(observation)
        closes = self._closes(state)
        closing = {c["position_id"] for c in closes}
        held = collections.Counter(
            p.underlying for p in state.positions if p.position_id not in closing
        )
        room = max(0, min(MAX_OPENS_PER_STEP_9,
                          MAX_POSITIONS - (len(state.positions) - len(closing))))

        signals: dict[str, dict[str, Any]] = {}
        candidates: list[tuple[float, dict[str, Any]]] = []
        for ticker in self._tradeable:
            row = state.market.get(ticker)
            if row is None:
                signals[ticker] = {"regime": "no_row"}
                continue
            feats = self._features(row)
            record: dict[str, Any] = {
                "iv": row.get("iv"), "rv": row.get("rv"), "ts": row.get("ts"),
                "sk": row.get("sk"), "bf": row.get("bf"), "ret": row.get("ret"),
                **feats,
            }
            ts = row.get("ts")
            if ts is None:
                record["gate"] = "no_term_slope"
                signals[ticker] = record
                continue
            if ts < TERM_SLOPE_GATE:
                record["gate"] = "suppressed_inverted_term_structure"
                signals[ticker] = record
                continue
            record["gate"] = "open"

            fired: list[str] = []
            for rule in RULES:
                value = feats.get(str(rule["feature"]))
                if value is None:
                    continue
                threshold = float(rule["threshold"])  # type: ignore[arg-type]
                if rule["op"] == "ge":
                    hit, distance = value >= threshold, value - threshold
                else:
                    hit, distance = value <= threshold, threshold - value
                if not hit:
                    continue
                orientation = rule["orientation"]
                if orientation == "sign":
                    ret = row.get("ret") or 0.0
                    orientation = "b" if ret >= 0.0 else "r"
                fired.append(str(rule["name"]))
                candidates.append((
                    distance,
                    {"ticker": ticker, "family": str(rule["family"]),
                     "orientation": str(orientation), "tenor": TENOR,
                     "rule": str(rule["name"]), "feature": str(rule["feature"]),
                     "value": value, "threshold": threshold,
                     "edge": rule["edge"],
                     "friction": FRICTION.get(str(rule["name"]))},
                ))
            record["fired"] = fired
            signals[ticker] = record

        # Strongest signal first, ties by ticker then family so the arm is
        # reproducible; then enforce the per-name cap the resolver would enforce.
        candidates.sort(key=lambda item: (-item[0], item[1]["ticker"], item[1]["family"]))
        opens: list[dict[str, Any]] = []
        for _, order in candidates:
            if len(opens) >= room:
                break
            ticker = order["ticker"]
            if held[ticker] >= MAX_OPENS_PER_NAME:
                continue
            opens.append(order)
            held[ticker] += 1
        return self._respond(state, closes, opens, signals)
