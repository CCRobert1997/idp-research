#!/usr/bin/env bash
# Resume v2 from analyze (E1 gen + E1.5 labeling + reliability already done).
set -u
ROOT=~/shangyu_comfyui/IDP_research
PY=~/shangyu_comfyui/venvs/comfyui/bin/python
export PYTHONWARNINGS=ignore GLOG_minloglevel=2
cd "$ROOT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "E2+E3+E4 analyze..."
$PY experiments/v2_analyze.py > logs/v2_analyze.log 2>&1
log "analyze done; E2: $(grep -o '\"first_step_ci_low_gt_0.5\".*' v2/E2_probes/summary.json 2>/dev/null)"

log "E5 held-out generation (504, seeds 2000..)..."
$PY experiments/v2_e1_generate.py --seeds 7 --seed_start 2000 --subdir E5_heldout --save_png 1 > logs/v2_e5gen.log 2>&1
log "E5 held-out VLM labeling..."
$PY experiments/v2_e15_label.py --subdir E5_heldout --batch 8 --reliability 0 > logs/v2_e5label.log 2>&1
log "E5 efficiency..."
$PY experiments/v2_e5_efficiency.py > logs/v2_e5.log 2>&1

log "E6 qualitative..."
$PY experiments/v2_e6_qual.py > logs/v2_e6.log 2>&1
log "figures..."
$PY experiments/v2_figures.py > logs/v2_figs.log 2>&1

log "PIPELINE v2 DONE (resume)"
{
  echo "IDP v2 finished $(date)"
  echo "=== E1.5 reliability ==="; cat v2/E1_data/reliability.json 2>/dev/null
  echo; echo "=== E2 overall ==="; cat v2/E2_probes/summary.json 2>/dev/null
  echo; echo "=== E2 layer ranking (top) ==="; head -8 v2/E2_probes/layer_ranking.csv 2>/dev/null
  echo; echo "=== E3 per-type summary (CORE) ==="; cat v2/E3_pertype/pertype_summary.csv 2>/dev/null
  echo; echo "=== E4 ablation ==="; cat v2/E4_ablation/ablation_table.csv 2>/dev/null
  echo; echo "=== E5 efficiency ==="; cat v2/E5_efficiency/summary.json 2>/dev/null
} > v2/PIPELINE_DONE_v2.txt 2>&1
log "wrote v2/PIPELINE_DONE_v2.txt"
