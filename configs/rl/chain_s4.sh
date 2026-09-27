#!/bin/bash
# Unattended chain: wait for 27B eval runs -> train s4_none (identical command) -> eval its checkpoints.
set -u
C=$HOME/work/campaign27; cd $C
log(){ echo "$(date '+%F %T') $*" >> $C/logs/chain.log; }
log "waiting for eval_window (sys_sft_rl 27B) to finish"
while pgrep -f "eval_window.sh sys_sft_rl=sys_sft_rl" > /dev/null; do sleep 30; done
log "eval status: $(tr '\n' ';' < evalwin_SEALED/status.txt)"
$HOME/verl-env/bin/ray stop --force > /dev/null 2>&1; sleep 5
G=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '); [ -n "$G" ] && kill -9 $G 2>/dev/null; sleep 5
$HOME/verl-env/bin/python -c "from pm_train.envfactory import consolidate_marks; print(consolidate_marks('$HOME/work/markquotes'))" >> $C/logs/chain.log 2>&1
mkdir -p s4_none ${PM_CKPT_ROOT}/s4_none27/checkpoints && ln -sfn ${PM_CKPT_ROOT}/s4_none27/checkpoints s4_none/checkpoints
diff <(tr ' ' '\n' < cmd_sys_sft_rl.sh | sed 's/sys_sft_rl/ARM/g') <(tr ' ' '\n' < cmd_s4_none.sh | sed 's/s4_none/ARM/g') >> $C/logs/chain.log 2>&1 && log "commands identical up to arm name"
source $HOME/massive
log "s4_none 27B training start"
date +%s > logs/rl2_start
env LD_LIBRARY_PATH= PATH=$HOME/verl91-env/bin:$PATH RAY_DEDUP_LOGS=0 /usr/bin/time -v bash cmd_s4_none.sh 2>&1 | awk '{ print strftime("%H:%M:%S"), $0; fflush() }' > logs/rl2.log
rc=${PIPESTATUS[0]}; echo "exit $rc" >> logs/rl2.log; date +%s > logs/rl2_end
log "s4_none 27B training exit $rc"
$HOME/verl91-env/bin/python -m pm_train.finalize $C/s4_none rl >> logs/rl2.log 2>&1
$HOME/verl-env/bin/ray stop --force > /dev/null 2>&1; sleep 5
G=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '); [ -n "$G" ] && kill -9 $G 2>/dev/null; sleep 5
Q=$C/s4_none/checkpoints
cp -rL $Q/global_step_20/actor/huggingface $HOME/work/campaign/final_ckpts/s4_none_27b_step20_hf &
log "s4_none eval start"
EVAL_OUT=$C/evalwin_SEALED EVAL_TP=2 EVAL_LMO=1 EVAL_PY=$HOME/verl91-env/bin/python $HOME/work/pm_train/scripts/eval_window.sh \
  s4_none=s4_none:$Q/global_step_20/actor/huggingface s4_none_step10=s4_none:$Q/global_step_10/actor/huggingface \
  s4_none_step30=s4_none:$Q/global_step_30/actor/huggingface s4_none_step40=s4_none:$Q/global_step_40/actor/huggingface >> logs/eval27_s4.out 2>&1
wait
log "ALL DONE; eval status: $(tr '\n' ';' < evalwin_SEALED/status.txt)"
