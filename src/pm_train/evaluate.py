"""Evaluation run: one checkpoint, the whole eval window, one run directory.

Inference only.  The six monthly episodes run as ONE ``run_arm`` call with the
book carried across month boundaries, exactly as the teacher runs and
``jobs/run_policy_episodes.py`` do, so the output directory has the same
shape the paper tables and the Bridges-2 replay arms read:

    ledger (nav_panel, position_steps, fills, strategies, decisions),
    manifest.json (+ the manifest_extra fields run_policy_episodes records),
    state/<episode>.json, summary.json (reconciliation + decision quality),
    metrics.json (ArmMetrics on the PM grid + a month-block bootstrap).

Decoding is greedy (temperature 0), the run_policy_episodes default.

Usage:
  python -m pm_train.evaluate --arm sys_sft_rl --checkpoint <hf dir> --out <run dir> \
      --dates <dates_eval.txt> [--gpu-mem 0.85] [--max-turn-tokens 512]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

from portfolio_monkey.env.chain import OptionChain
from portfolio_monkey.env.datasets import DatasetFeatureSource
from portfolio_monkey.env.ledger import reconcile
from portfolio_monkey.env.tokens import build_token_counter
from portfolio_monkey.eval.builders.from_ledger import build_ledger_panel, decision_quality
from portfolio_monkey.eval.metrics import compute
from portfolio_monkey.jobs.run_policy_episodes import _coverage_gaps
from portfolio_monkey.env.spec import SIZE_RULE_LIMITS

from .envfactory import arm_config, env_factory, episodes_for, read_dates
from .episode import GenOut, drive


def manifest_extra(config, dates, root: Path, checkpoint: str, max_turn_tokens: int) -> dict:
    """The fields run_policy_episodes adds, computed the same way."""
    chain = OptionChain(config.resolver, root=root)
    features = DatasetFeatureSource(root, extract_dir=None)
    gaps = _coverage_gaps(features, config, dates)
    by_name = chain.name_coverage(config.universe.tradeable, dates)
    h = config.hedge
    return {
        "window": [dates[0].isoformat(), dates[-1].isoformat()],
        "feature_coverage_gaps": {k: len(v) for k, v in gaps.items()},
        "chain_coverage": {k: len(v) for k, v in by_name.items()},
        "textual_context": not config.flags.suppress_textual_context,
        "suppress_future_knowledge": config.flags.suppress_future_knowledge,
        "anonymize": config.flags.anonymize,
        "hedge_band": {
            "rule": h.band_rule, "delta_band": h.delta_band, "risk_aversion": h.risk_aversion,
            "band_multiple": h.band_multiple, "min_band_fraction": h.min_band_fraction,
            "max_band_fraction": h.max_band_fraction, "fallback_half_spread": h.fallback_half_spread,
            "spread_table": None, "measured_spread_costs": config.flags.measured_spread_costs,
        },
        "sizing": {
            "rule": config.size.size_rule, "limits": list(SIZE_RULE_LIMITS[config.size.size_rule]),
            "target_scenario_risk": config.size.target_scenario_risk,
            "vol_shock_relative": config.size.vol_shock_relative,
            "nav_fraction": config.size.nav_fraction, "max_positions": config.size.max_positions,
        },
        "quotes": {"enabled": config.quotes_enabled, "max_per_name": config.max_quotes_per_name,
                   "channel": config.quote_channel},
        "student_checkpoint": checkpoint,
        "decoding": {"temperature": 0.0, "max_turn_tokens": max_turn_tokens,
                     "chat_template": "pm_train.chat.CHAT_TEMPLATE (no think block)"},
    }


def month_block_bootstrap(panel, *, n: int = 10_000, seed: int = 0) -> dict:
    """Resample whole months of daily PM log returns, with replacement.

    Written for this report: the hand-off cites a date-block bootstrap in
    eval/metrics.py, but none exists in the package.  Blocks are calendar
    months (the episode unit), so within-month dependence is kept intact.  This
    is within-run variability only -- it says nothing about training-seed
    variance, which one seed cannot measure.
    """
    pm = panel.pm_rows()
    # First segment: the opening mark to the first close, so the bootstrap's
    # point estimate equals cumulative_log_return (the table's criterion).
    series = [(pm[0].trade_date, math.log(pm[0].nlv) - math.log(panel.rows[0].nlv))]
    series += [(b.trade_date, math.log(b.nlv) - math.log(a.nlv)) for a, b in zip(pm, pm[1:])]
    months: dict[str, list[float]] = {}
    for day, r in series:
        months.setdefault(day.isoformat()[:7], []).append(r)
    total = sum(r for _, r in series)
    if abs(total - (math.log(panel.rows[-1].nlv) - math.log(panel.rows[0].nlv))) > 1e-9:
        return {"error": "PM series does not telescope to the cumulative log return"}
    blocks = list(months.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [x for _ in blocks for x in rng.choice(blocks)]
        stats.append(sum(sample))
    stats.sort()
    q = lambda p: stats[min(len(stats) - 1, int(p * len(stats)))]
    return {
        "method": "calendar-month block bootstrap of the sum of daily PM log returns",
        "blocks": {k: len(v) for k, v in months.items()},
        "n_resamples": n, "seed": seed,
        "cumulative_log_return": total,
        "ci95": [q(0.025), q(0.975)],
        "se": (sum((s - sum(stats) / n) ** 2 for s in stats) / (n - 1)) ** 0.5,
    }


async def run(args) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    root = Path(args.data_root)
    config = arm_config(args.arm)
    dates = read_dates(args.dates)
    episodes = list(episodes_for(config, dates))
    if args.max_model_len is None:
        args.max_model_len = config.max_context_tokens + 4096
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    counter = build_token_counter(str(Path(args.checkpoint) / "tokenizer.json"))
    llm = LLM(model=args.checkpoint, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True, seed=0,
              dtype=args.dtype, tensor_parallel_size=args.tp,
              **({"language_model_only": True} if args.language_model_only else {}))

    async def generate(prompt_ids, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        res = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp, use_tqdm=False)
        return GenOut(list(res[0].outputs[0].token_ids))

    out_dir = Path(args.out)
    per_episode = []
    t0 = time.time()
    result = await drive(
        arm=args.arm_label or f"{args.arm}_eval",
        episodes=episodes,
        env_factory_for=lambda ledger: env_factory(config, data_root=root,
                                                   mark_cache_dir=args.mark_cache, ledger=ledger),
        run_dir=out_dir, tokenizer=tok, counter=counter, generate=generate,
        max_turn_tokens=args.max_turn_tokens, context_budget=config.max_context_tokens,
        gamma=None, multi_episode=True,
        on_episode_end=lambda msgs, used: per_episode.append(
            {"turns": sum(m["role"] == "assistant" for m in msgs), "context_tokens": used}),
        manifest_extra=manifest_extra(config, dates, root, args.checkpoint, args.max_turn_tokens),
    )
    if result.budget_exceeded:
        # Fatal by the environment's rule (ContextBudgetExceeded): the run is
        # incomplete and is not a result.  Say so; never write metrics for it.
        (out_dir / "summary.json").write_text(json.dumps({
            "status": "context_budget_exceeded", "budget": config.max_context_tokens,
            "context_tokens": result.context_tokens, "episodes_completed": len(per_episode),
            "turns_in_failing_episode": result.extra.get("episode_turns"),
            "gen_seconds": result.gen_seconds, "env_seconds": result.env_seconds,
        }, indent=2))
        print(f"CONTEXT BUDGET EXCEEDED after {len(per_episode)} complete episode(s): "
              f"{result.context_tokens} > {config.max_context_tokens}", file=sys.stderr)
        return 3
    check = reconcile(out_dir)
    quality = decision_quality(out_dir)
    summary = {
        "log_return": result.log_return,
        "gen_seconds": result.gen_seconds, "env_seconds": result.env_seconds,
        "wall_seconds": time.time() - t0, "truncated_turns": result.truncated_turns,
        "episodes": per_episode,
        "reconciliation": {"ok": check.ok, "steps": check.steps, "max_error": check.max_error,
                           "worst_step": check.worst_step, "by_identity": dict(check.by_identity)},
        "decision_quality": quality.as_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    panel = build_ledger_panel(out_dir, grid="PM", start=dates[0], end=dates[-1])
    metrics = compute(panel)
    boot = month_block_bootstrap(panel)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"arm_metrics": metrics.as_dict(),
         "bootstrap": boot}, indent=2, default=str))
    print(json.dumps({"log_return": result.log_return, "reconcile_ok": check.ok,
                      "bootstrap": boot}, default=str))
    return 0 if check.ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("sys_sft", "sys_sft_rl", "s4_none"))
    p.add_argument("--arm-label", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dates", required=True)
    p.add_argument("--data-root", default=str(Path.home() / "rl_package_v1/data"))
    p.add_argument("--mark-cache", default=str(Path.home() / "work/markquotes"))
    p.add_argument("--max-turn-tokens", type=int, default=512)
    # default: context budget + 4096 (36,864 at the 32k budget, as before)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    # bf16 explicitly: veRL exports fp32 HF weights, and "auto" would pick a
    # half precision by vLLM version; rollouts ran in bf16.
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--language-model-only", action="store_true",
                   help="VLM checkpoint used text-only (Qwen3.8); see pm_train.chat.text_only_view")
    args = p.parse_args(argv)
    if Path(args.out).exists() and any(Path(args.out).iterdir()):
        print(f"{args.out} exists and is not empty; refusing to append to a run", file=sys.stderr)
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
