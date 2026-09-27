#!/bin/bash
# Evaluation window (124 dates), greedy, book carried.  One GPU per checkpoint.
# usage: eval_window.sh <label>=<arm>:<ckpt> [...]      (runs in parallel)
set -u
C=$HOME/work/campaign; OUT=${EVAL_OUT:-$C/evalwin}; mkdir -p $OUT
source $HOME/massive
DATES=$HOME/rl_package_v1/code/configs/dates_eval.txt
g=${GPU0:-0}
TP=${EVAL_TP:-1}; PY=${EVAL_PY:-$HOME/verl-env/bin/python}; LMO=${EVAL_LMO:+--language-model-only}
EXTRA_ENV=""; [ "$TP" -gt 1 -o -n "$LMO" ] && EXTRA_ENV="LD_LIBRARY_PATH= PATH=$(dirname $PY):$PATH"
for spec in "$@"; do
  label=${spec%%=*}; rest=${spec#*=}; arm=${rest%%:*}; ckpt=${rest#*:}
  [ -d $OUT/$label ] && { echo "skip $label"; continue; }
  $PY -c "from pm_train.chat import install_template, text_only_view; install_template('$ckpt'); [text_only_view('$ckpt') for _ in [0] if '$LMO']"
  gl=$(seq -s, $g $((g+TP-1)))
  ( env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$gl AWS_BEARER_TOKEN_BEDROCK= $PY -m pm_train.evaluate --tp $TP $LMO \
      --arm $arm --arm-label ${label} --checkpoint $ckpt --out $OUT/$label --dates $DATES \
      --max-turn-tokens 256 --gpu-mem 0.85 > $OUT/$label.log 2>&1; echo "$label exit $?" >> $OUT/status.txt ) &
  g=$((g+TP))
done
wait
