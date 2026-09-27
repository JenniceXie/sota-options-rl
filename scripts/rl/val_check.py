"""Round-3 validation (REPORT.md, pre-registered 10:20 UTC).

Sep / Oct / Nov 2024 (the SFT window: never seen by RL, not in the eval
window), each month separately from a flat book, greedy, 256 tokens/turn.
Metric: mean monthly run log-return; any failed month -> ineligible.

usage: val_check.py <out_root> <label>=<arm>:<ckpt> [...]
"""
from __future__ import annotations

import json
import os
import queue
import statistics
import subprocess
import sys
import threading
from pathlib import Path

HOME = Path.home()
DATES = HOME / "rl_package_v1/code/configs/dates_sft.txt"
MONTHS = ("2024-09", "2024-10", "2024-11")
GPUS = [int(g) for g in os.environ.get("VAL_GPUS", "0,1,2,3,4,5,6,7").split(",")]
# 27B: VAL_TP=2 groups GPUs in pairs; VAL_PY picks the interpreter; VAL_LMO=1 adds
# --language-model-only; the 0.9.1 stack also needs LD_LIBRARY_PATH cleared.
TP = int(os.environ.get("VAL_TP", "1"))
PY = os.environ.get("VAL_PY", str(HOME / "verl-env/bin/python"))
LMO = os.environ.get("VAL_LMO") == "1"
SLOTS = [",".join(str(g) for g in GPUS[i:i + TP]) for i in range(0, len(GPUS) - TP + 1, TP)]


def month_file(root: Path, month: str) -> Path:
    p = root / f"dates_{month}.txt"
    if not p.exists():
        p.write_text("".join(l for l in DATES.read_text().splitlines(True) if l.startswith(month)))
    return p


def main() -> int:
    root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
    specs = []
    for s in sys.argv[2:]:
        label, rest = s.split("=", 1); arm, ckpt = rest.split(":", 1)
        specs.append((label, arm, ckpt))
        subprocess.run([PY, "-c",
                        f"from pm_train.chat import install_template, text_only_view; install_template({ckpt!r})"
                        + (f"; text_only_view({ckpt!r})" if LMO else "")], check=True)
    jobs: queue.Queue = queue.Queue()
    for label, arm, ckpt in specs:
        for m in MONTHS:
            jobs.put((label, arm, ckpt, m))
    env = {k: v for k, v in os.environ.items() if k != "AWS_BEARER_TOKEN_BEDROCK"}
    if TP > 1 or LMO:
        env["LD_LIBRARY_PATH"] = ""
        env["PATH"] = str(Path(PY).parent) + ":" + env.get("PATH", "")

    def worker(gpu: int):
        while True:
            try:
                label, arm, ckpt, m = jobs.get_nowait()
            except queue.Empty:
                return
            out = root / label / m
            if (out / "summary.json").exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(root / label / f"{m}.log", "w") as log:
                rc = subprocess.run(
                    [PY, "-m", "pm_train.evaluate", "--arm", arm, "--tp", str(TP),
                     *(["--language-model-only"] if LMO else []),
                     "--arm-label", f"val_{label}_{m}", "--checkpoint", ckpt, "--out", str(out),
                     "--dates", str(month_file(root, m)), "--max-turn-tokens", "256", "--gpu-mem", "0.85"],
                    env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT).returncode
            with open(root / "status.txt", "a") as fh:
                fh.write(f"{label} {m} exit {rc}\n")

    threads = [threading.Thread(target=worker, args=(g,)) for g in SLOTS]
    [t.start() for t in threads]; [t.join() for t in threads]

    print(f"{'label':22s} " + " ".join(f"{m:>9s}" for m in MONTHS) + f" {'mean':>9s}  eligible  ctx")
    table = {}
    for label, _, _ in specs:
        vals, ctx, ok = [], [], True
        for m in MONTHS:
            s = json.loads((root / label / m / "summary.json").read_text())
            if s.get("status") == "context_budget_exceeded":
                vals.append(None); ok = False; ctx.append(s["context_tokens"])
            else:
                vals.append(s["log_return"]); ctx.append(s["episodes"][0]["context_tokens"])
        mean = statistics.mean(v for v in vals if v is not None) if any(v is not None for v in vals) else None
        table[label] = {"months": dict(zip(MONTHS, vals)), "mean": mean, "eligible": ok, "ctx": ctx}
        fmt = lambda v: "     FAIL" if v is None else f"{v:+9.4f}"
        print(f"{label:22s} " + " ".join(fmt(v) for v in vals) + f" {fmt(mean)}  {str(ok):8s}  {ctx}")
    (root / "validation.json").write_text(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
