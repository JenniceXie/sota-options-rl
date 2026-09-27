#!/bin/bash
# Tuning criterion (REPORT.md rule 2): greedy, RL window (2024-12..2025-02),
# book carried across the 3 months.  SFT vs every saved RL checkpoint, one GPU
# each, in parallel.  Never touches the evaluation window.
#
# usage: rl_window_check.sh <rl_output_dir> <tag>
set -u
RL=$1; TAG=$2
C=$HOME/work/campaign
OUT=$C/rlwin/$TAG
mkdir -p $OUT
source $HOME/massive
DATES=$HOME/rl_package_v1/code/configs/dates_rl.txt
PY=$HOME/verl-env/bin/python
gpu=0
run() {  # arm label ckpt
  local arm=$1 label=$2 ckpt=$3 g=$4
  [ -d $OUT/$label ] && { echo "skip $label (exists)"; return; }
  CUDA_VISIBLE_DEVICES=$g env -u AWS_BEARER_TOKEN_BEDROCK $PY -m pm_train.evaluate --arm $arm \
    --arm-label ${TAG}_$label --checkpoint $ckpt --out $OUT/$label --dates $DATES \
    --max-turn-tokens 256 --gpu-mem 0.85 > $OUT/$label.log 2>&1
  echo "$label exit $?" >> $OUT/status.txt
}
if [ ! -d $C/rlwin/sft ]; then
  run sys_sft sft $C/sys_sft/hf $gpu &
  gpu=$((gpu+1))
fi
for s in $(ls -d $RL/checkpoints/global_step_* | sed 's/.*global_step_//' | sort -n); do
  hf=$RL/checkpoints/global_step_$s/actor/huggingface
  $PY -c "from pm_train.chat import install_template; install_template('$hf')"
  arm=$(basename $RL); case $arm in sys_sft_rl*) a=sys_sft_rl;; s4_none*) a=s4_none;; esac
  run $a step$s $hf $gpu &
  gpu=$((gpu+1))
done
wait
[ -d $C/rlwin/sft ] || ln -s $OUT/sft $C/rlwin/sft 2>/dev/null
$PY - <<EOF
import json, pathlib
out = pathlib.Path("$OUT")
rows = []
for d in sorted(list(out.glob("*/")) + [pathlib.Path("$C/rlwin/sft")]):
    s = d / "summary.json"
    if not s.exists():
        continue
    j = json.loads(s.read_text())
    m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    am = m.get("arm_metrics", {})
    rows.append((d.name, j.get("status", "complete"), j.get("log_return"),
                 am.get("cumulative_log_return"), am.get("sharpe"), am.get("max_drawdown_daily"),
                 [e["context_tokens"] for e in j.get("episodes", [])]))
print(f"{'ckpt':10s} {'status':24s} {'run_logret':>10s} {'PM_cumlog':>10s} {'sharpe':>7s} {'mdd':>7s}  ctx/episode")
for r in rows:
    f = lambda x, p: "-" if x is None else f"{x:{p}}"
    print(f"{r[0]:10s} {r[1]:24s} {f(r[2], '+10.4f')} {f(r[3], '+10.4f')} {f(r[4], '7.2f')} {f(r[5], '7.4f')}  {r[6]}")
EOF
