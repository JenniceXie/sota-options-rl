"""Fixed-strategy and uniform-random baselines on the held-out test window.

Two baseline families, both running inside the **unmodified** environment and
through the same resolvers, cost model and portfolio constraints every other arm
uses:

1. **Fixed-strategy.** One arm per action-space cell. The cells are
   ``rules._RANDOM_FAMILIES`` -- the repo's own enumeration of
   (family, orientation) heads -- and each is pinned with
   ``RuleParams(signal="none", short_vol_family=<head>)``, which
   ``rules._order_for`` opens unconditionally every step (``rules.py:569-570``).
   No new policy code: the rule policy already does this.

2. **Uniform-random.** ``RuleParams(signal="random")``, which draws uniformly
   from the same 14 heads. **This is uniform over the grammar, not over the
   admissible set** -- an inadmissible draw is refused by the resolver and
   recorded as a reject rather than resampled. The repo has no enumerator of
   admissible actions, so a true uniform-over-admissible arm would need
   rejection sampling through ``OptionsEnv.quote``; that is a separate arm and
   is deliberately not what this script runs. The distinction is recorded in
   each row as ``uniform_over="grammar"``.

**Both are seeded.** ``signal="none"`` ranks names by ``rng.random()``
(``rules.py:499-503``) and opens ``max_opens_per_step`` of them, so a "fixed"
strategy still carries a seed in its *name selection*. Running many seeds and
reporting the spread is therefore not optional for these arms; it is the only
honest way to report them.

Why a separate script rather than a new grid in ``sweep_algorithmic_policies``:
``RuleParams.label()`` does not include the family for ``signal="none"`` and
includes the seed only for ``signal="random"``, so every fixed arm would collide
on one output directory. Arm names are assigned here instead, and the policy's
own ``name`` is overwritten to match so the ledger and the manifest agree.

Usage::

    .venv/bin/python scripts/analysis/run_baseline_arms.py \\
        --start 2025-03-03 --end 2025-08-29 \\
        --data-root "$PM_DATA_ROOT" --out "$PM_ARM_ROOT/baselines_test" \\
        --risk-free-rate 0.043 --hedge volatility --tenor 0_7 --seeds 1000

Shard across an array job with ``--shard i --shards n``; each shard keeps its
own warm chain cache, which is where nearly all the wall time goes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_algorithmic_policies import summarize  # noqa: E402

from portfolio_monkey.env.chain import OptionChain  # noqa: E402
from portfolio_monkey.env.datasets import DatasetFeatureSource, data_root  # noqa: E402
from portfolio_monkey.env.environment import (  # noqa: E402
    OptionsEnv,
    build_grid,
    monthly_episodes,
)
from portfolio_monkey.env.ledger import LedgerWriter  # noqa: E402
from portfolio_monkey.env.policy import HoldPolicy  # noqa: E402
from portfolio_monkey.env.policy.rules import (  # noqa: E402
    _RANDOM_FAMILIES,
    RuleParams,
    RulePolicy,
)
from portfolio_monkey.env.runner import run_arm  # noqa: E402
from portfolio_monkey.env.markquotes import MARK_QUOTE_SOURCE_VERSION, MassiveMarkQuotes  # noqa: E402
from portfolio_monkey.training.spec import paper_base_config  # noqa: E402
from portfolio_monkey.env.spec import (  # noqa: E402
    DIRECTIONAL_FAMILIES,
    VOLATILITY_FAMILIES,
    CostModel,
    EnvConfig,
    GridSpec,
    HedgeSpec,
)


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _slug(head: str) -> str:
    """``"ic n"`` -> ``"icn"``; a directory name may not carry a space."""
    return head.replace(" ", "")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=_as_date, required=True)
    p.add_argument("--end", type=_as_date, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--feature-cache", default=None)
    p.add_argument("--chain-cache-dates", type=int, default=0)
    p.add_argument("--risk-free-rate", type=float, default=None,
                   help="ignored unless it matches the RL environment (None)")
    p.add_argument("--mark-cache", type=Path, required=True,
                   help="append-only Massive mark-quote cache")
    p.add_argument("--anonymize", type=int, choices=(0, 1), default=1)
    p.add_argument("--target-fingerprint", default=None,
                   help="refuse to run unless the config reproduces this")
    p.add_argument("--decision-sessions", default="PM")
    p.add_argument("--hedge", choices=("off", "volatility", "all"), default="volatility")
    p.add_argument(
        "--tenor",
        default="0_7",
        help="tenor bucket every arm trades; 0_7 is the bucket whose anchor (5) "
        "is nearest Cboe's weekly roll, which is the cadence of all 27 "
        "single-stock strategy indices and of WPUT",
    )
    p.add_argument(
        "--names",
        default="",
        help="comma-separated tickers the arms may trade; empty means the whole "
        "config universe. With a single name the fixed arms become "
        "seed-independent -- there is no cross-name ranking left for the RNG to "
        "do -- which is also the construction Cboe's single-index overlays use.",
    )
    p.add_argument(
        "--per-name",
        action="store_true",
        help="treat --names as a list to iterate rather than as one universe: "
        "each name gets its own single-name arms, all sharing this process's "
        "warm chain cache instead of paying one cold load per name",
    )
    p.add_argument(
        "--tenors",
        default="",
        help="comma-separated tenor buckets to sweep, overriding --tenor. The "
        "horizon dimension: the bucket sets the contract's DTE and therefore "
        "its gamma/theta/vega, but note the holding rule (--hold-steps / "
        "--min-dte) is unchanged across buckets, so a longer bucket is a "
        "longer-dated CONTRACT, not a longer HOLD.",
    )
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--mode", choices=("fixed", "random", "both"), default="both"
    )
    p.add_argument("--hold-steps", type=int, default=4)
    p.add_argument(
        "--min-dte",
        type=int,
        default=1,
        help="close at this many days to expiry; 1 approximates the "
        "hold-to-expiry of Cboe's weekly indices, against RuleParams' default 3",
    )
    p.add_argument("--max-positions", type=int, default=6)
    p.add_argument("--max-opens-per-step", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--no-keep-ledgers", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def cells(args: argparse.Namespace) -> Iterator[tuple[str, RuleParams | None, str]]:
    """``(arm_name, params, kind)``; ``params is None`` is the hold control."""

    names = tuple(n.strip().upper() for n in args.names.split(",") if n.strip())
    tenors = tuple(t.strip() for t in args.tenors.split(",") if t.strip()) or (args.tenor,)
    base = dict(
        hold_steps=args.hold_steps,
        min_dte=args.min_dte,
        max_positions=args.max_positions,
        max_opens_per_step=args.max_opens_per_step,
    )
    seeds = range(args.seed_start, args.seed_start + args.seeds)

    # ``hold`` never trades, so its NAV is the risk-free accrual and does not
    # depend on which name it was pointed at.  One row, not one per name.
    yield ("hold", None, "control")

    # In per-name mode each universe is a single ticker; otherwise the whole
    # ``--names`` list (or the config universe) is one universe.
    universes = [(n, (n,)) for n in names] if args.per_name else [(None, names)]

    for tag, universe in universes:
        prefix = f"{tag}." if tag else ""
        for tenor in tenors:
            common = dict(base, names=universe, tenor=tenor)
            if args.mode in ("fixed", "both"):
                for head in _RANDOM_FAMILIES:
                    for seed in seeds:
                        yield (
                            f"{prefix}fix.{_slug(head)}.{tenor}.s{seed:04d}",
                            RuleParams(
                                signal="none", short_vol_family=head, seed=seed, **common
                            ),
                            "fixed",
                        )
            if args.mode in ("random", "both"):
                for seed in seeds:
                    yield (
                        f"{prefix}rnd.{tenor}.s{seed:04d}",
                        RuleParams(signal="random", seed=seed, **common),
                        "random",
                    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sessions = tuple(
        s.strip().upper() for s in args.decision_sessions.split(",") if s.strip()
    )
    hedged = {
        "off": (),
        "volatility": VOLATILITY_FAMILIES,
        "all": VOLATILITY_FAMILIES + DIRECTIONAL_FAMILIES,
    }[args.hedge]
    # RL environment: identical recipe to pm_train/envfactory.arm_config().
    # Only the environment changes; every RuleParams field below is unchanged.
    overrides = {"mark_quote_source": MARK_QUOTE_SOURCE_VERSION}
    if not args.anonymize:
        overrides["flags.anonymize"] = False
    config = paper_base_config(**overrides)
    assert config.risk_free_rate is None, config.risk_free_rate
    assert config.size.size_rule == "scenario", config.size.size_rule
    assert set(config.hedge.hedged_families) == set(hedged), (config.hedge.hedged_families, hedged)
    assert config.grid.decision_sessions == sessions, (config.grid.decision_sessions, sessions)
    fp = config.fingerprint()
    print(f"env fingerprint {fp}  anonymize={config.flags.anonymize}", flush=True)
    if args.target_fingerprint and fp != args.target_fingerprint:
        print(f"REFUSING: fingerprint {fp} != target {args.target_fingerprint}", file=sys.stderr)
        return 2
    marks = MassiveMarkQuotes(cache_path=args.mark_cache)

    root = data_root(args.data_root)
    extract_dir = (
        None
        if args.feature_cache == "none"
        else (Path(args.feature_cache) if args.feature_cache else root / ".feature_extracts")
    )
    probe = OptionChain(config.resolver, root=root)
    dates = [d for d in probe.coverage if args.start <= d <= args.end]
    if len(dates) < 2:
        print(f"only {len(dates)} chain date(s) in window under {root}", file=sys.stderr)
        return 1
    chain = OptionChain(
        config.resolver,
        root=root,
        cache_size=args.chain_cache_dates or 2 * len(dates) + 4,
    )
    features = DatasetFeatureSource(root, extract_dir=extract_dir)
    episodes = monthly_episodes(build_grid(dates, config))

    plan = [c for i, c in enumerate(cells(args)) if i % args.shards == args.shard]
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / f"rows.shard{args.shard:03d}.jsonl"
    if not args.quiet:
        print(f"window   {dates[0]} .. {dates[-1]}  ({len(dates)} dates, {len(episodes)} episodes)")
        shown = args.tenors or args.tenor
        print(f"tenors   {shown}   hedge {args.hedge}   min_dte {args.min_dte}")
        print(f"plan     {len(plan)} arms  (shard {args.shard + 1}/{args.shards})")

    # Append-only: a killed shard keeps every arm it finished, and a resume
    # skips them by row rather than by re-deriving the plan.  A truncated final
    # row is tolerated -- a process killed mid-write leaves one, and refusing to
    # parse it would strand every completed arm behind it.
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["arm"])
            except (json.JSONDecodeError, KeyError):
                print(f"skipping unparseable row (truncated write?): {line[:80]}")

    # An arm that ran but was never recorded left a *partial* ledger behind.
    # ``LedgerWriter`` opens its tables in append mode, so re-running it would
    # concatenate the new run onto the old fragment and `summarize` would then
    # parse a line that is half one run and half the next.  That is exactly how
    # the 2026-09-25 resume died, 811 s into a chain load.  Clear them first so
    # a resume is idempotent.
    for arm, _params, _kind in plan:
        target = args.out / arm
        if arm not in done and target.exists():
            for stale in target.rglob("*"):
                if stale.is_file():
                    stale.unlink()
            print(f"cleared partial ledger for {arm}")

    with rows_path.open("a") as sink:
        for step, (arm, params, kind) in enumerate(plan):
            if arm in done:
                continue
            policy = HoldPolicy() if params is None else RulePolicy(params, config)
            # ``RuleParams.label()`` is not unique across families or seeds, so
            # the arm name is assigned here and pushed onto the policy so the
            # ledger, the manifest and ``decisions[].model`` all agree.
            policy.name = arm
            out = args.out / arm
            out.mkdir(parents=True, exist_ok=True)
            ledger = LedgerWriter(out, arm=arm, track="full")
            env = OptionsEnv(config, chain=chain, features=features, ledger=ledger, marks=marks)
            try:
                result = run_arm(
                    env, policy, episodes, arm=arm, ledger=ledger, state_dir=out / "state"
                )
            finally:
                ledger.close()
            row = summarize(out, arm, result)
            row["kind"] = kind
            row["tenor"] = args.tenor if params is None else params.tenor
            row["seed"] = None if params is None else params.seed
            row["head"] = None if params is None else params.short_vol_family
            row["names"] = [] if params is None else list(params.names)
            row["uniform_over"] = "grammar" if kind == "random" else None
            if params is not None:
                row["params"] = {k: getattr(params, k) for k in params.__slots__}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if args.no_keep_ledgers:
                for table in out.glob("*.jsonl"):
                    table.unlink()
            if not args.quiet and step % 25 == 0:
                print(
                    f"[{step + 1:>5}/{len(plan)}] {arm:<28} "
                    f"logret {row['log_return']:+.4f}  opens {row['n_open']:>4}  "
                    f"rej {row['n_reject']:>4}"
                )
    if not args.quiet:
        print(f"wrote {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
