#!/usr/bin/env bash
# Orchestrate the full IDP pipeline after E1 finishes. Idempotent-ish; logs to logs/.
set -u
ROOT=~/shangyu_comfyui/IDP_research
PY=~/shangyu_comfyui/venvs/comfyui/bin/python
export PYTHONWARNINGS=ignore GLOG_minloglevel=2
cd "$ROOT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1. wait for E1 to reach 2000 npz
log "waiting for E1 (2000 npz)..."
while true; do
  n=$(ls E1_data/raw/ 2>/dev/null | wc -l)
  [ "$n" -ge 2000 ] && break
  sleep 30
done
log "E1 has $(ls E1_data/raw/ | wc -l) npz; building labels.csv"
$PY experiments/e1_generate.py --csv_only 1 --outdir E1_data > logs/e1_csv.log 2>&1

# 2. E2 probe (kill-gate)
log "E2 probe..."
$PY experiments/e2_probe.py > logs/e2.log 2>&1
log "E2 done; gate: $(grep -o '\"kill_gate\".*' E2_probes/summary.json 2>/dev/null | head -1)"

# 3. generate test500 (25 seeds x 20 prompts, seeds 1000..1024)
log "generating test500..."
$PY experiments/e1_generate.py --seeds 25 --seed_start 1000 --outdir E4_efficiency/test500 --save_png 1 > logs/test500.log 2>&1

# 4. E4 efficiency
log "E4 efficiency..."
$PY experiments/e4_efficiency.py > logs/e4.log 2>&1

# 5. E3 ablation
log "E3 ablation..."
$PY experiments/e3_ablation.py > logs/e3.log 2>&1

# 6. E5 qualitative
log "E5 qualitative..."
$PY experiments/e5_qualitative.py > logs/e5.log 2>&1

log "PIPELINE DONE"
{
  echo "IDP pipeline finished $(date)"
  echo "=== E2 summary ==="; cat E2_probes/summary.json 2>/dev/null
  echo "=== E3 ablation ==="; cat E3_ablation/ablation_table.csv 2>/dev/null
  echo "=== E4 summary ==="; cat E4_efficiency/summary.json 2>/dev/null
} > results/PIPELINE_DONE.txt 2>&1
log "wrote results/PIPELINE_DONE.txt"
