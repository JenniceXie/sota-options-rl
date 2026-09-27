"""8 concurrent rollouts in ONE process (veRL 0.9.1's layout): thread vs process env.

Replays 8 teacher runs' September decisions through drive(); reports mean env
seconds per rollout and wall time for each mode.
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from portfolio_monkey.env.tokens import build_token_counter
from pm_train.envfactory import arm_config, env_factory, episodes_for, read_dates
from pm_train.episode import GenOut, drive
from pm_train.procsession import ProcessRolloutSession

PKG = Path.home() / "rl_package_v1"
MODEL = Path.home() / "models" / "Qwen3-4B-pm"
CACHE = Path.home() / "work" / "markquotes"
RUNS = ["sft_astra2_text_r0584", "sft_astra2_text_r0785", "sft_astra2_text_r0008", "sft_astra2_text_r0059",
        "sft_astra2_text_r0339", "sft_astra2_text_r0511", "sft_astra2_text_r0632", "sft_astra2_text_r0926"]


async def one(run_id, mode, tok, counter, config, sept):
    comps = iter(json.loads(l)["completion"] for l in (PKG / "runs_main_sample" / run_id / "decisions.jsonl").open())
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    async def gen(ids, n):
        await asyncio.sleep(0)
        return GenOut(tok.encode(next(comps), add_special_tokens=False) + [im_end])

    rd = Path.home() / "work" / "bench" / mode / run_id
    factory = None
    if mode == "process":
        factory = lambda: ProcessRolloutSession(
            arm_id="sys_sft", label=run_id, episode_ids=[sept.episode_id],
            dates_file=str(PKG / "code/configs/dates_sft.txt"), run_dir=rd,
            data_root=PKG / "data", mark_cache_dir=CACHE)
    out = await drive(arm=run_id, episodes=[sept], session_factory=factory,
                      env_factory_for=lambda led: env_factory(config, data_root=PKG / "data",
                                                              mark_cache_dir=CACHE, ledger=led),
                      run_dir=rd, tokenizer=tok, counter=counter, generate=gen,
                      max_turn_tokens=256, context_budget=10**9, gamma=0.99)
    return out.env_seconds, out.log_return


async def main(mode):
    tok = AutoTokenizer.from_pretrained(MODEL)
    counter = build_token_counter(str(MODEL / "tokenizer.json"))
    config = arm_config("sys_sft")
    sept = episodes_for(config, read_dates(PKG / "code/configs/dates_sft.txt"))[0]
    t = time.perf_counter()
    res = await asyncio.gather(*(one(r, mode, tok, counter, config, sept) for r in RUNS))
    print(f"{mode:8s}: 8 concurrent rollouts, env s/rollout mean {statistics.mean(e for e, _ in res):.1f} "
          f"max {max(e for e, _ in res):.1f}, wall {time.perf_counter() - t:.1f} s, "
          f"returns {[round(r, 4) for _, r in res]}")
    return [r for _, r in res]


if __name__ == "__main__":
    a = asyncio.run(main("thread"))
    b = asyncio.run(main("process"))
    print("returns identical across modes:", a == b)
    sys.exit(0 if a == b else 1)
