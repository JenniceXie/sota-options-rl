"""Per-step summary of a running RL job: reward, time split, context, overflow.

Usage: python rl_status.py <log> <rollouts.jsonl> [G*months]
"""
import json
import re
import statistics
import sys

log, rollouts = sys.argv[1], sys.argv[2]
per_step = int(sys.argv[3]) if len(sys.argv) > 3 else 24
K = {
    "gen": "timing_s/gen", "oldlp": "timing_s/old_log_prob", "ref": "timing_s/ref",
    "upd": "timing_s/update_actor", "sync": "timing_s/update_weights",
    "save": "timing_s/save_checkpoint", "step": "timing_s/step",
    "score": "critic/score/mean", "gnorm": "actor/grad_norm", "kl": "actor/kl_loss",
}
steps = {}
for line in open(log, errors="replace"):
    if "timing_s/gen" not in line:
        continue
    d = dict(re.findall(r"([\w/]+):([-0-9.e+]+)", line))
    steps[int(float(d["training/global_step"]))] = {k: float(d[v]) for k, v in K.items() if v in d}
rs = [json.loads(l) for l in open(rollouts)]
print(f"{'step':>4} {'score':>8} {'step_s':>6} {'gen_s':>6} {'opt_s':>6} {'env/ro':>6} {'gen/ro':>6} "
      f"{'ctx_med':>7} {'ctx_max':>7} {'over':>4} {'gnorm':>6} {'kl':>8}")
for s in sorted(steps):
    m = steps[s]
    b = rs[(s - 1) * per_step: s * per_step]
    opt = sum(m.get(k, 0.0) for k in ("oldlp", "ref", "upd", "sync"))
    print(f"{s:>4} {m.get('score', 0):>+8.4f} {m.get('step', 0):>6.1f} {m.get('gen', 0):>6.1f} {opt:>6.1f} "
          f"{statistics.mean(r['env_seconds'] for r in b) if b else 0:>6.1f} "
          f"{statistics.mean(r['gen_seconds'] for r in b) if b else 0:>6.1f} "
          f"{int(statistics.median(r['context_tokens'] for r in b)) if b else 0:>7} "
          f"{max(r['context_tokens'] for r in b) if b else 0:>7} "
          f"{sum(r['budget_exceeded'] for r in b):>4} {m.get('gnorm', 0):>6.3f} {m.get('kl', 0):>8.5f}")
if not steps:
    # veRL 0.9.1 buffers the console metrics until exit: summarize from rollouts.
    print("(no step metrics in log yet; from rollouts.jsonl)")
    print(f"{'step':>4} {'mean_rew':>9} {'mean_lr':>8} {'env/ro':>6} {'gen/ro':>6} {'ctx_med':>7} {'ctx_max':>7} {'over':>4}")
    for i in range(len(rs) // per_step):
        b = rs[i * per_step:(i + 1) * per_step]
        lrs = [r['log_return'] for r in b if r['log_return'] == r['log_return']]
        print(f"{i+1:>4} {statistics.mean(r['reward'] for r in b):>+9.4f} {statistics.mean(lrs) if lrs else float('nan'):>+8.4f} "
              f"{statistics.mean(r['env_seconds'] for r in b):>6.1f} {statistics.mean(r['gen_seconds'] for r in b):>6.1f} "
              f"{int(statistics.median(r['context_tokens'] for r in b)):>7} {max(r['context_tokens'] for r in b):>7} "
              f"{sum(r['budget_exceeded'] for r in b):>4}")
print(f"rollouts written: {len(rs)}  (steps logged: {len(steps)})")
