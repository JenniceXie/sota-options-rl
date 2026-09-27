"""The training reward, and the one place it is allowed to differ from the metric.

THE RULING THIS FILE IMPLEMENTS.  ``r_t = log V_{t+1} - log V_t`` per decision,
**discounted** for training; evaluation reports the same log return
**undiscounted**.  Those are two numbers from one series, and conflating them
was explicitly overturned, so the discount factor is never defaulted here: a
caller that forgets it gets a refusal rather than silently-undiscounted
training.

WHERE THE NUMBERS COME FROM, AND WHERE THEY MUST NOT.  ``nav_panel.jsonl``,
never ``trip.profit``: a trip's profit is a per-position accounting figure that
double-counts a roll and ignores the cost of the leg that replaced it.  The
account-level mark is the only quantity that closes.

THE MARK ALIGNMENT PROBLEM, AND THE FIX.  The panel's PM ``nlv`` is **after**
that session's fills -- the first PM row of a fresh run reads
``1,000,000 - 2,246.29 (half-spread) - 421.22 (fees) = 997,332.50``.  So
differencing the raw PM series charges decision *t*'s transaction cost to
decision *t-1*: the book that earned the move gets the bill for the next
rebalance.  Totals still telescope, but per-step credit is shifted by one, and a
discounted sum weights the steps unequally, so the shift is not cosmetic.

The fix is exact rather than approximate, because at the instant of a fill the
only thing that moves NAV *is* the cost:

    NAV_before_trades = nlv + cost_half_spread + cost_fees

That identity is checked against the panel in
``test_the_pre_trade_mark_reconstruction_is_exact``, and it is checked *against
corrupted input* too -- a reconstruction that cannot fail proves nothing.

So the mark series is: the opening AM mark, then each interior decision's
**pre-trade** NAV, then the final decision's **post-trade** NAV.  Every decision
carries its own cost, the sum still telescopes to
``log(end_nav / start_nav)``, and the terminal trades are not made free.  The
raw post-trade alignment is kept as an option, not deleted, because it is what
the panel literally says and a reviewer will want to see both.

CREDIT WITHIN A STEP IS NOT SPLIT.  A decision step and its quote round are one
decision; the quote turn moves no NAV and receives no reward of its own.  A
framework that wants a per-turn reward vector gets zeros on the quote turns,
which is the truth rather than a smoothing choice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "RewardError",
    "StepReward",
    "Trajectory",
    "compute_score",
    "discounted_return",
    "group_advantages",
    "load_trajectory",
    "step_rewards",
    "undiscounted_return",
]


class RewardError(ValueError):
    """The NAV series cannot be turned into a reward."""


#: What a wiped-out account is worth as a log return.  NAV <= 0 makes the log
#: undefined, and an exception at that point would crash a rollout worker on
#: the one trajectory the policy most needs to be punished for.  ``log(0.01)``
#: says "you lost 99% of the account", which is strictly worse than any
#: survivable path this environment can produce and is finite.
RUIN_REWARD = math.log(0.01)


@dataclass(frozen=True)
class StepReward:
    """One decision's reward, with the two marks it was computed from."""

    index: int
    episode_id: str
    step_ts: str
    nav_before: float
    nav_after: float
    reward: float
    #: True when ``nav_after`` was non-positive and :data:`RUIN_REWARD` was used.
    ruined: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "episode_id": self.episode_id,
            "step_ts": self.step_ts,
            "nav_before": self.nav_before,
            "nav_after": self.nav_after,
            "reward": self.reward,
            "ruined": self.ruined,
        }


@dataclass(frozen=True)
class Trajectory:
    """A run's NAV series, reduced to what a reward needs.

    Deliberately not the whole panel: a reward function that can see
    ``dollar_vega`` is a reward function somebody will eventually shape with.
    """

    arm: str
    start_nav: float
    #: Chronological, one entry per decision.
    marks: tuple[tuple[str, str, float, float], ...]
    #: ``(episode_id, step_ts, nlv_post_trade, transaction_cost)``
    terminated: str | None = None

    @property
    def end_nav(self) -> float:
        return self.marks[-1][2] if self.marks else self.start_nav


def load_trajectory(run_dir: Path | str, *, track: str = "full") -> Trajectory:
    """Read ``nav_panel.jsonl`` and keep the decision grid.

    ``track`` exists because a run may carry a counterfactual panel beside the
    traded one; pooling the two produces a reward for decisions that were never
    made.
    """
    path = Path(run_dir) / "nav_panel.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("track", track) == track:
                rows.append(row)
    if not rows:
        raise RewardError(f"{path}: no rows on track {track!r}")
    rows.sort(key=lambda r: (str(r["step_ts"]), bool(r.get("is_decision_point"))))

    decisions = [r for r in rows if r.get("is_decision_point")]
    if not decisions:
        raise RewardError(
            f"{path}: no decision points on track {track!r}. Differencing every "
            "grid row instead would invent a reward for the opening mark, which "
            "no policy chose."
        )
    # The opening mark, which the PM grid does not contain: the first decision
    # happens at the close of the first day, so its entry cost is only visible
    # against the AM mark that precedes it.
    start_nav = float(rows[0]["nlv"])
    marks = tuple(
        (
            str(r.get("episode_id", "")),
            str(r["step_ts"]),
            float(r["nlv"]),
            float(r.get("cost_half_spread") or 0.0) + float(r.get("cost_fees") or 0.0),
        )
        for r in decisions
    )
    terminated = next((r["terminated"] for r in reversed(rows) if r.get("terminated")), None)
    return Trajectory(
        arm=str(rows[0].get("arm", "")),
        start_nav=start_nav,
        marks=marks,
        terminated=str(terminated) if terminated else None,
    )


def step_rewards(
    trajectory: Trajectory, *, trade_aligned: bool = True
) -> tuple[StepReward, ...]:
    """One :class:`StepReward` per decision.

    With ``trade_aligned`` (the default) each decision is charged its own
    transaction cost, by marking the interior steps *before* their fills.  With
    ``trade_aligned=False`` the raw panel marks are differenced, which is what
    the file literally says and shifts cost attribution back one step.  Both
    sum to ``log(end_nav / start_nav)``.
    """
    if not trajectory.marks:
        return ()
    levels = [trajectory.start_nav]
    last = len(trajectory.marks) - 1
    for i, (_, _, nlv, cost) in enumerate(trajectory.marks):
        # The final mark is post-trade in both alignments: the trades of the
        # last decision are real, and pricing them at their pre-trade value
        # would hand the policy a free liquidation.
        levels.append(nlv + cost if (trade_aligned and i != last) else nlv)

    out: list[StepReward] = []
    ruined = False
    for i, (episode_id, step_ts, _, _) in enumerate(trajectory.marks):
        before, after = levels[i], levels[i + 1]
        if ruined:
            # The account is gone; later steps are not the policy's doing.
            out.append(StepReward(i, episode_id, step_ts, before, after, 0.0, True))
            continue
        if before <= 0.0:
            raise RewardError(
                f"{trajectory.arm}: mark {i} starts from NAV {before}, which is "
                "not a survivable state the environment should have produced."
            )
        if after <= 0.0:
            ruined = True
            out.append(StepReward(i, episode_id, step_ts, before, after, RUIN_REWARD, True))
            continue
        out.append(
            StepReward(i, episode_id, step_ts, before, after, math.log(after / before))
        )
    return tuple(out)


def discounted_return(rewards: Sequence[StepReward], gamma: float) -> float:
    """``sum_t gamma^t r_t``.  ``gamma`` is required, on purpose.

    Training discounts and evaluation does not; that split was ruled and it
    supersedes an earlier note saying they were the same quantity.  A default
    here would let the two drift back together without anybody editing a config.
    """
    if not 0.0 < gamma <= 1.0:
        raise RewardError(f"gamma must be in (0, 1], got {gamma}")
    return sum(r.reward * gamma**i for i, r in enumerate(rewards))


def undiscounted_return(rewards: Sequence[StepReward]) -> float:
    """The evaluation quantity: ``log(V_T / V_0)``, by telescoping."""
    return sum(r.reward for r in rewards)


def group_advantages(returns: Sequence[float]) -> tuple[float, ...]:
    """GRPO's group-relative advantage: the z-score within the group.

    The zero-variance case is not an edge case here, it is the *common* case
    early in training: this environment lets a policy abstain, and a group of
    eight rollouts that all abstain have identical returns.  Dividing by that
    standard deviation is a NaN that propagates into the gradient and kills the
    run with no error message, so it returns zeros -- which is also the correct
    answer, since no member of the group was better than another.
    """
    n = len(returns)
    if n == 0:
        return ()
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return tuple(0.0 for _ in returns)
    return tuple((r - mean) / sd for r in returns)


# --------------------------------------------------------------------------
# framework entry point
# --------------------------------------------------------------------------


def compute_score(
    *,
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
    gamma: float | None = None,
    trade_aligned: bool = True,
    **_ignored: Any,
) -> float:
    """The trajectory reward, in the shape veRL's reward manager calls.

    Signature per ``verl/workers/reward_manager/naive.py:86-90``, which calls
    ``fn(data_source=, solution_str=, ground_truth=, extra_info=)``, and
    ``verl/trainer/ppo/reward.py:82`` which merges
    ``reward.custom_reward_function.reward_kwargs`` over those -- that is how
    ``gamma`` arrives.

    ``solution_str`` is accepted and **not read**.  The reward is realized
    after-cost PnL; scoring the model's text would be scoring its prose.
    """
    if gamma is None:
        raise RewardError(
            "compute_score needs gamma. Supply it as "
            "reward.custom_reward_function.reward_kwargs.gamma -- the training "
            "reward is discounted and the evaluation metric is not, and a "
            "default here would quietly merge them."
        )
    info = dict(extra_info or {})
    run_dir = info.get("run_dir")
    if run_dir:
        trajectory = load_trajectory(run_dir, track=info.get("track", "full"))
    elif "nav_marks" in info:
        trajectory = _trajectory_from_marks(info)
    else:
        raise RewardError(
            "compute_score found neither 'run_dir' nor 'nav_marks' in extra_info. "
            "The rollout session is responsible for putting the realized NAV "
            "path there; returning 0.0 instead would train the policy on a "
            "constant and look like a working run."
        )
    return discounted_return(
        step_rewards(trajectory, trade_aligned=trade_aligned), gamma
    )


def _trajectory_from_marks(info: Mapping[str, Any]) -> Trajectory:
    """In-memory path, for a rollout that never wrote a panel to disk."""
    marks: list[tuple[str, str, float, float]] = []
    for mark in info["nav_marks"]:
        if isinstance(mark, Mapping):
            marks.append((
                str(mark.get("episode_id", "")),
                str(mark.get("step_ts", "")),
                float(mark["nlv"]),
                float(mark.get("transaction_cost") or 0.0),
            ))
        else:  # a bare NAV series, cost unavailable
            marks.append(("", "", float(mark), 0.0))
    if "start_nav" not in info:
        raise RewardError(
            "extra_info carries nav_marks without start_nav. Taking the first "
            "mark as the opening level would silently drop the entry cost of "
            "the first decision, which is the one step whose cost is largest."
        )
    return Trajectory(
        arm=str(info.get("arm", "")),
        start_nav=float(info["start_nav"]),
        marks=tuple(marks),
        terminated=info.get("terminated"),
    )


def reward_vector(
    rewards: Sequence[StepReward], turns: Iterable[str]
) -> tuple[float, ...]:
    """Spread step rewards over a turn sequence, zero on the non-acting turns.

    ``turns`` is the recorded turn kinds in order (``"act"``, ``"quote"``, ...).
    A quote round moves no NAV, so it scores zero rather than inheriting the
    neighbouring step -- inheriting would credit the same PnL twice under any
    framework that sums the vector.

    Note for whoever reads a recorded ``decisions.jsonl``: the ``turn`` label
    does **not** sit on the observation it describes, so build this from the
    rollout's own turn order, not by trusting that field.
    """
    kinds = list(turns)
    acting = [i for i, kind in enumerate(kinds) if kind == "act"]
    if len(acting) != len(rewards):
        raise RewardError(
            f"{len(acting)} acting turns but {len(rewards)} step rewards. "
            "Padding either side would misalign every later step."
        )
    out = [0.0] * len(kinds)
    for slot, reward in zip(acting, rewards):
        out[slot] = reward.reward
    return tuple(out)
