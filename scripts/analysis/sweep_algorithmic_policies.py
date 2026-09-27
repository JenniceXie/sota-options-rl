"""Sweep non-LLM policies over the window and report whether any trajectory wins.

    python scripts/analysis/sweep_algorithmic_policies.py \
        --start 2024-09-03 --end 2025-08-29 \
        --risk-free-rate 0.043 \
        --out $PM_PROJECT_ROOT/runs/algo_sweep

The question this answers is not "which rule is good".  It is whether the
environment admits a positive after-cost trajectory at all.  Every LLM arm so
far has finished the window below the hold arm, and that observation is
consistent with two incompatible worlds: one where the policy is bad, and one
where the reward is negative for every reachable action and the policy was
never playing a winnable game.  Those two worlds call for opposite next steps
-- more capable policies in the first, a corrected cost or fill model in the
second -- and GRPO on a uniformly negative reward would spend a training run
learning to abstain before anyone noticed which world we were in.

So: run rules.  Rules cost nothing, take minutes rather than the ten hours an
LLM arm takes, and a rule that clears zero is a constructive existence proof
that survives having been chosen with hindsight -- the claim is about the
*reachable set*, not about the rule.

**What a winning rule is not.**  Every parameter here was chosen by looking at
the same window it is scored on.  The best cell of the sweep is an upper bound
on what hindsight buys, not a strategy and not a baseline the LLM should be
expected to match.  The honest baselines in this table are ``hold`` (which
earns ``r_f`` and nothing else) and ``random`` (which is the monkey the project
is named after).  If ``random`` is not the worst row, the sweep is measuring
something other than skill.

**One chain read, many arms.**  ``OptionChain`` and ``DatasetFeatureSource``
are built once and shared across every arm in the sweep, because on /ocean a
cold partition read is the dominant cost and re-reading the window per cell
would make a 40-cell sweep 40x the I/O for identical bytes.  Each arm still
gets a fresh ``OptionsEnv``, a fresh ``Book`` and its own ledger directory, so
nothing but the cache is shared.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from math import log
from pathlib import Path

from portfolio_monkey.env.chain import OptionChain
from portfolio_monkey.env.datasets import DatasetFeatureSource, data_root
from portfolio_monkey.env.environment import OptionsEnv, build_grid, monthly_episodes
from portfolio_monkey.env.ledger import LedgerWriter, read_jsonl, reconcile
from portfolio_monkey.env.policy import HoldPolicy
from portfolio_monkey.env.policy.rules import RuleParams, RulePolicy
from portfolio_monkey.env.runner import run_arm
from portfolio_monkey.env.spec import (
    DIRECTIONAL_FAMILIES,
    VOLATILITY_FAMILIES,
    CostModel,
    EnvConfig,
    GridSpec,
    HedgeSpec,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, type=_as_date)
    parser.add_argument("--end", required=True, type=_as_date)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument(
        "--chain-cache-dates",
        type=int,
        default=0,
        help=(
            "how many trading dates of chain to hold in memory; 0 (default) "
            "sizes it to the window. The sweep's whole cost is chain I/O -- "
            "measured 2.9 s/date against a cold partition and ~0 against a warm "
            "cache -- and OptionChain's own default of 24 dates is smaller than "
            "any real window, so every arm after the first re-reads every "
            "partition. Costs roughly 40 MB per date of resident memory"
        ),
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.043)
    parser.add_argument("--decision-sessions", default="PM")
    parser.add_argument(
        "--grid",
        default="core",
        choices=("smoke", "core", "wide", "decisive"),
        help=(
            "which parameter grid to sweep. 'smoke' is four cells for a "
            "plumbing check, 'core' is the signal x tenor x holding-period "
            "grid, 'wide' adds thresholds and position caps, 'decisive' is the "
            "handful of cells 'core' singled out, for decomposing them"
        ),
    )
    parser.add_argument(
        "--half-spread-multiplier",
        type=float,
        action="append",
        default=None,
        help=(
            "repeatable. Re-runs the whole grid at each value of "
            "CostModel.half_spread_multiplier, which is the sensitivity axis "
            "protocol section 10 already names (0, 0.25, 0.5, 1.0). Default 1.0 "
            "-- the honest one, since marks are taken at mid. Sweeping it is the "
            "only way to tell a policy that cannot find an edge from an edge "
            "that cannot survive the fill convention"
        ),
    )
    parser.add_argument(
        "--keep-ledgers",
        action="store_true",
        help=(
            "keep every arm's jsonl tables. Off by default: a wide sweep writes "
            "one ledger per cell and only the summary is read afterwards"
        ),
    )
    parser.add_argument(
        "--hedge",
        choices=("off", "volatility", "all"),
        default="off",
        help=(
            "which families get delta hedged. Off by default so that this "
            "sweep's history stays comparable. NOT a boolean, because "
            "HedgeSpec.hedged_families defaults to empty: enabling the "
            "resolver without naming families hedges nothing and produces an "
            "arm indistinguishable from 'off'. 'volatility' is the intended "
            "setting of env_contract.md section 9.1 -- straddles and condors "
            "are opened near delta-flat and drift, which is what a band is "
            "for. 'all' additionally hedges the directional families, which "
            "removes the only source of return a debit vertical has; it is a "
            "diagnostic, not a strategy"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# the grids
# ---------------------------------------------------------------------------


def _grid(name: str) -> Iterator[RuleParams]:
    """Parameter cells, coarsest first.

    The grids are deliberately small.  A fine grid over a single window finds a
    winner by construction -- with enough cells one of them fits the noise --
    and the finding this script exists for is binary, so resolution buys
    nothing and costs the right to report the winner as anything but an upper
    bound.
    """
    if name == "smoke":
        for signal in ("random", "wedge", "trend", "none"):
            yield RuleParams(signal=signal, tenor="8_30", hold_steps=4)
        return

    if name == "decisive":
        # The four cells the ``core`` sweep singled out, for re-running with
        # ledgers kept and across the cost axis.  ``wedge_long.8_30.h10`` won
        # ``core``; ``wedge.8_30.hNone`` is the best two-sided cell, so the
        # pair says whether the win needs the long-only restriction; ``random``
        # is the monkey and must stay last; ``hold`` is prepended by ``main``.
        # Nothing here was chosen before seeing the window, which is why this
        # grid is for decomposition and never for a fresh claim.
        yield RuleParams(signal="wedge_long", tenor="8_30", hold_steps=10)
        yield RuleParams(signal="wedge", tenor="8_30", hold_steps=None)
        yield RuleParams(signal="wedge_short", tenor="0_7", hold_steps=1)
        yield RuleParams(signal="random", tenor="8_30", hold_steps=4, seed=0)
        return

    tenors = ("0_7", "8_30", "31_90") if name == "core" else ("0_7", "8_30", "31_90", "91_180")
    holds = (1, 4, 10, None)

    # The monkey, one cell per seed.  Several seeds because a single random
    # trajectory that happens to clear zero proves nothing about the reachable
    # set that a coin flip does not.
    for seed in range(4):
        yield RuleParams(signal="random", tenor="8_30", hold_steps=4, seed=seed)

    for signal, tenor, hold in itertools.product(
        ("none", "wedge", "wedge_short", "wedge_long", "trend", "revert"), tenors, holds
    ):
        yield RuleParams(signal=signal, tenor=tenor, hold_steps=hold)

    if name != "wide":
        return

    for threshold, cap, tenor in itertools.product((0.02, 0.10), (2, 12), tenors):
        yield RuleParams(
            signal="wedge_short",
            tenor=tenor,
            hold_steps=4,
            wedge_threshold=threshold,
            max_positions=cap,
        )
        yield RuleParams(
            signal="wedge_long",
            tenor=tenor,
            hold_steps=4,
            wedge_threshold=threshold,
            max_positions=cap,
        )


# ---------------------------------------------------------------------------


def summarize(out: Path, arm: str, result) -> dict:
    """One row of the table, read back off the ledger rather than the runner.

    ``log_return`` comes from the runner's book; everything else comes from the
    tables.  Reading both is the cheap version of the cross-check the panel
    builder does -- if an arm's fills do not exist, the log return is the
    risk-free rate and the row is a hold arm wearing a rule's name.
    """
    fills = list(read_jsonl(out / "fills.jsonl"))
    decisions = list(read_jsonl(out / "decisions.jsonl"))
    opens = [f for f in fills if f["kind"] == "open"]
    closes = [f for f in fills if f["kind"] == "close"]
    expiries = [f for f in fills if f["kind"] == "expire"]
    rejects = [
        entry
        for row in decisions
        for entry in row.get("results", ())
        if entry.get("status") not in (None, "OK")
    ]
    held = [row for row in decisions if not row.get("results")]
    cost = sum(f["half_spread"] + f["fees"] for f in fills)
    return {
        "arm": arm,
        "log_return": result.log_return,
        # What the arm would have returned had every fill been at mid.  A first
        # cut and not a counterfactual run: it adds the friction back at the end
        # rather than re-sizing every position without it, so it overstates the
        # rescue for an arm that was collateral-bound.  It is here because it is
        # the one number that separates "the rules are bad" from "the round trip
        # is unaffordable", and those need different fixes.
        "log_return_at_mid": (
            log((result.end_nav + cost) / result.start_nav)
            if result.start_nav > 0 and result.end_nav + cost > 0
            else 0.0
        ),
        # Both ends, because the NAV panel's first row is written *after* that
        # step's fills: anything decomposing the arm from the panel alone has
        # no anchor for the first month and silently loses the opening trade.
        "start_nav": result.start_nav,
        "end_nav": result.end_nav,
        "n_open": len(opens),
        "n_close": len(closes),
        "n_expire": len(expiries),
        "n_reject": len(rejects),
        "abstain_share": len(held) / len(decisions) if decisions else 0.0,
        # ``Fill.cost`` is a property, so it is not in ``asdict`` and not on the
        # wire.  Summing a key that is never written reads as a costless run,
        # which is exactly the conclusion this script must not reach by accident.
        "cost": round(cost, 2),
        "notional": round(sum(abs(f["mid_value"]) for f in fills), 2),
        "terminated": result.terminated,
        "reconciles": reconcile(out).ok,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    sessions = tuple(s.strip().upper() for s in args.decision_sessions.split(",") if s.strip())
    multipliers = args.half_spread_multiplier or [CostModel().half_spread_multiplier]

    # Named families, never a bare ``enabled=True``.  ``hedged_families``
    # defaults to empty, so flipping the switch alone leaves the resolver with
    # nothing to hedge and the arm is byte-identical to ``off`` -- the same
    # unexercised-branch failure that ``delta_band`` already has a comment for.
    hedged_families = {
        "off": (),
        "volatility": VOLATILITY_FAMILIES,
        "all": VOLATILITY_FAMILIES + DIRECTIONAL_FAMILIES,
    }[args.hedge]
    hedge = HedgeSpec(enabled=args.hedge != "off", hedged_families=hedged_families)

    def config_at(multiplier: float) -> EnvConfig:
        return EnvConfig(
            grid=GridSpec(decision_sessions=sessions),
            risk_free_rate=args.risk_free_rate,
            hedge=hedge,
            cost=CostModel(half_spread_multiplier=multiplier),
        )

    # The grid, the chain and the episode plan do not depend on the cost model,
    # so they are built against the first one and reused.
    config = config_at(multipliers[0])
    root = data_root(args.data_root)
    extract_dir = None if args.feature_cache == "none" else (
        Path(args.feature_cache) if args.feature_cache else root / ".feature_extracts"
    )
    probe = OptionChain(config.resolver, root=root)
    dates = [d for d in probe.coverage if args.start <= d <= args.end]
    if len(dates) < 2:
        print(f"only {len(dates)} chain date(s) in the window under {root}", file=sys.stderr)
        return 1
    # Sized after the window is known, because the whole point is that it covers
    # it: a cache one date short of the window evicts date 1 just before the
    # next arm asks for it, and the sweep degrades to cold reads everywhere.
    chain = OptionChain(
        config.resolver,
        root=root,
        cache_size=args.chain_cache_dates or 2 * len(dates) + 4,
    )
    features = DatasetFeatureSource(root, extract_dir=extract_dir)
    episodes = monthly_episodes(build_grid(dates, config))

    cells = [None, *_grid(args.grid)]  # ``None`` is the hold control
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        print(f"window   {dates[0]} .. {dates[-1]}  ({len(dates)} dates, {len(episodes)} episodes)")
        print(f"sweep    {len(cells)} arms  grid={args.grid}")
        print(f"hedge    {args.hedge}  families={len(hedged_families)}")

    def publish(rows: list[dict]) -> None:
        """Rewrite ``sweep.json`` from whatever has finished so far.

        Called after every arm, not once at the end.  A ``core`` grid is 77 arms
        over about an hour, and each arm's ledger is deleted as soon as it is
        summarized, so this file is the only place its result survives.  A run
        killed on arm 76 therefore left nothing at all behind -- which is what
        happened.  The file is tens of kB; writing it 77 times costs nothing
        against losing 76 arms.
        """
        (args.out / "sweep.json").write_text(
            json.dumps(
                {
                    "window": [dates[0].isoformat(), dates[-1].isoformat()],
                    # Recorded because it is not recoverable from the arm names:
                    # a hedged and an unhedged sweep write identical directory
                    # trees, and comparing one against the other is the whole
                    # reason to run both.
                    "hedge": args.hedge,
                    "hedged_families": list(hedged_families),
                    "arms": sorted(rows, key=lambda r: -r["log_return"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    rows: list[dict] = []
    total = len(cells) * len(multipliers)
    for step, (multiplier, params) in enumerate(itertools.product(multipliers, cells)):
        cost_config = config_at(multiplier)
        policy = HoldPolicy() if params is None else RulePolicy(params, cost_config)
        # The multiplier goes in the directory name, not just the row, because
        # two cost models running the same rule would otherwise write their
        # ledgers over each other and the second would reconcile against the
        # first one's fills.
        arm = policy.name if len(multipliers) == 1 else f"{policy.name}@hs{multiplier:g}"
        out = args.out / arm
        out.mkdir(parents=True, exist_ok=True)
        ledger = LedgerWriter(out, arm=arm, track="full")
        env = OptionsEnv(cost_config, chain=chain, features=features, ledger=ledger)
        try:
            result = run_arm(env, policy, episodes, arm=arm, ledger=ledger, state_dir=out / "state")
        finally:
            ledger.close()
        row = summarize(out, arm, result)
        row["half_spread_multiplier"] = multiplier
        if params is not None:
            row["params"] = {k: getattr(params, k) for k in params.__slots__}
        rows.append(row)
        if not args.quiet:
            print(
                f"[{step + 1:>3}/{total}] {arm:<34} "
                f"logret {row['log_return']:+.4f}  opens {row['n_open']:>4}  "
                f"rejects {row['n_reject']:>4}  cost {row['cost']:>12,.0f}"
            )
        if not args.keep_ledgers:
            for table in out.glob("*.jsonl"):
                table.unlink()
        publish(rows)

    rows.sort(key=lambda r: -r["log_return"])
    publish(rows)

    def line(row: dict) -> str:
        return (
            f"{row['arm']:<34} {row['log_return']:>+9.4f} "
            f"{row['log_return_at_mid']:>+9.4f} {row['n_open']:>6} "
            f"{row['n_reject']:>5} {row['abstain_share']:>6.1%} {row['cost']:>13,.0f}"
        )

    header = f"{'arm':<34} {'logret':>9} {'@mid':>9} {'opens':>6} {'rej':>5} {'abst':>6} {'cost':>13}"
    for multiplier in multipliers:
        block = [r for r in rows if r["half_spread_multiplier"] == multiplier]
        # ``hold`` never trades, so its return is the same under every cost
        # model.  It is re-run per multiplier anyway rather than reused, so
        # that this equality is something the table shows rather than assumes.
        hold = next(r["log_return"] for r in block if r["arm"].startswith("hold"))
        rules = [r for r in block if not r["arm"].startswith("hold")]
        winners = [r for r in rules if r["log_return"] > 0.0]
        beat_hold = [r for r in rules if r["log_return"] > hold]
        at_mid = [r for r in rules if r["log_return_at_mid"] > 0.0]

        print()
        print(f"=== half_spread_multiplier = {multiplier:g} ===")
        print(f"hold                  {hold:+.4f}")
        print(f"positive after cost   {len(winners)} of {len(rules)} rule arms")
        print(f"beat hold             {len(beat_hold)} of {len(rules)} rule arms")
        # The gap between these counts is the finding.  Many arms positive at
        # mid and none positive after cost says the round trip is unaffordable,
        # not that the signals are empty -- and that is a statement about the
        # environment that no amount of policy capability changes.  Sweeping
        # the multiplier is what turns that from an inference into a
        # measurement: if the winners appear as the multiplier falls, the fill
        # convention is the binding constraint and its value is a modelling
        # choice that has to be defended before GRPO runs against it.
        print(f"positive at mid       {len(at_mid)} of {len(rules)} rule arms")
        print()
        print(header)
        for row in block[:15]:
            print(line(row))
        if len(block) > 15:
            print("...")
            for row in block[-3:]:
                print(line(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
