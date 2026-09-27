"""The rollout conversation must be the SFT conversation, byte for byte.

Drives the real environment through ``drive()`` with a generator that emits a
teacher run's recorded completions, and requires:

1. every episode's assembled ``messages`` == that run's rows in
   ``sft_corpus/train.jsonl`` (system block, header, observations, labels);
2. the token ids ``drive()`` assembled == tokenizing the same messages through
   the pinned chat template in one go (so SFT and RL tokenize identically);
3. decode(encode(completion)) == completion for every label;
4. the run's log return == the teacher manifest's, to the last digit.

Usage: python test_corpus_equivalence.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

from portfolio_monkey.env.tokens import build_token_counter
from pm_train.chat import CHAT_TEMPLATE
from pm_train.envfactory import arm_config, env_factory, episodes_for, read_dates
from pm_train.episode import GenOut, drive

PKG = Path.home() / "rl_package_v1"
MODEL = Path.home() / "models" / "Qwen3-4B-pm"
CACHE = Path.home() / "work" / "markquotes"


def corpus_rows(arm: str) -> dict[str, list[dict]]:
    out = {}
    for line in (PKG / "sft_corpus" / "train.jsonl").open():
        row = json.loads(line)
        if row["metadata"]["arm"] == arm:
            out[row["metadata"]["episode_id"]] = [
                {"role": m["role"], "content": m["content"]} for m in row["messages"]
            ]
    return out


async def check(run_id: str, tok, counter) -> bool:
    run = PKG / "runs_main_sample" / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    completions = [json.loads(l)["completion"] for l in (run / "decisions.jsonl").open()]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    it = iter(completions)
    roundtrip_bad = 0

    async def generate(prompt_ids, max_tokens):
        nonlocal roundtrip_bad
        text = next(it)
        ids = tok.encode(text, add_special_tokens=False)
        if tok.decode(ids, skip_special_tokens=True) != text:
            roundtrip_bad += 1
        return GenOut(ids + [im_end])

    config = arm_config("sys_sft")
    episodes = episodes_for(config, read_dates(PKG / "code/configs/dates_sft.txt"))
    seen: dict[str, tuple[list, int]] = {}
    order = [e.episode_id for e in episodes]

    def on_end(messages, used):
        seen[order[len(seen)]] = (list(messages), used)

    factory = None
    if os.environ.get("PM_PROC_SESSION") == "1":
        from pm_train.procsession import ProcessRolloutSession
        factory = lambda: ProcessRolloutSession(
            arm_id="sys_sft", label=run_id, episode_ids=order,
            dates_file=str(PKG / "code/configs/dates_sft.txt"),
            run_dir=Path.home() / "work" / "equiv_proc" / run_id,
            data_root=PKG / "data", mark_cache_dir=CACHE)
    out = await drive(
        session_factory=factory,
        arm=run_id, episodes=episodes,
        env_factory_for=lambda ledger: env_factory(config, data_root=PKG / "data",
                                                   mark_cache_dir=CACHE, ledger=ledger),
        run_dir=Path.home() / "work" / ("equiv_proc" if factory else "equiv") / run_id,
        tokenizer=tok, counter=counter, generate=generate,
        max_turn_tokens=512, context_budget=10**9, gamma=None,
        multi_episode=True, on_episode_end=on_end,
    )
    expected = corpus_rows(run_id)
    ok = True
    for eid, (messages, _) in seen.items():
        if eid not in expected:
            print(f"  {eid}: not in corpus (dropped by the cap), skipped")
            continue
        same = messages == expected[eid]
        if not same:
            ok = False
            for i, (a, b) in enumerate(zip(messages, expected[eid])):
                if a != b:
                    print(f"  {eid}: first differing message {i} role {a['role']}/{b['role']}")
                    print("    got     ", repr(a["content"][:300]))
                    print("    expected", repr(b["content"][:300]))
                    break
            else:
                print(f"  {eid}: length {len(messages)} vs {len(expected[eid])}")
        print(f"  {eid}: messages identical to corpus: {same}")
    # token assembly (last episode) vs one-shot template render
    last = seen[order[-1]][0]
    oneshot = tok.apply_chat_template(last, tokenize=False, add_generation_prompt=False)
    oneshot_ids = tok.encode(oneshot, add_special_tokens=False)
    assembled = out.prompt_ids + out.response_ids
    tok_ok = assembled == oneshot_ids
    print(f"  token ids identical to one-shot template render: {tok_ok} "
          f"({len(assembled)} vs {len(oneshot_ids)})")
    ret_ok = out.log_return == manifest["log_return"]
    print(f"  log return {out.log_return!r} vs teacher {manifest['log_return']!r}: {ret_ok}")
    print(f"  completion decode round-trip failures: {roundtrip_bad}")
    print(f"  env seconds {out.env_seconds:.1f}  wall {out.wall_seconds:.1f}")
    return ok and tok_ok and ret_ok and roundtrip_bad == 0


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Behavioural check: the model dir's template must render exactly like the
    # pinned one (the 4B dir keeps the string it was trained with; the current
    # CHAT_TEMPLATE additionally accepts list content and renders strings the same).
    from jinja2 import Template
    probe = [{"role": "system", "content": "s\nx"}, {"role": "user", "content": "u"},
             {"role": "assistant", "content": "a"}]
    assert Template(tok.chat_template).render(messages=probe, add_generation_prompt=True) == \
        Template(CHAT_TEMPLATE).render(messages=probe, add_generation_prompt=True)
    counter = build_token_counter(str(MODEL / "tokenizer.json"))
    results = {}
    for run_id in sys.argv[1:]:
        print(run_id)
        results[run_id] = asyncio.run(check(run_id, tok, counter))
    print("ALL OK" if all(results.values()) else f"FAILED: {[k for k, v in results.items() if not v]}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
