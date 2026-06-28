#!/usr/bin/env bash
# IDP v2 full pipeline: waits for E1 gen (5040), then label -> analyze -> heldout -> eff -> qual -> figs.
set -u
ROOT=~/shangyu_comfyui/IDP_research
PY=~/shangyu_comfyui/venvs/comfyui/bin/python
export PYTHONWARNINGS=ignore GLOG_minloglevel=2
cd "$ROOT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for v2 E1 (5040 npz)..."
while [ "$(ls v2/E1_data/raw 2>/dev/null | wc -l)" -lt 5040 ]; do sleep 30; done
log "E1 done ($(ls v2/E1_data/raw | wc -l)); manifest"
$PY experiments/v2_e1_generate.py --manifest_only 1 --subdir E1_data > logs/v2_manifest.log 2>&1

log "E1.5 VLM labeling (5040) + reliability..."
$PY experiments/v2_e15_label.py --subdir E1_data --batch 8 --reliability 1 > logs/v2_e15.log 2>&1
log "labeled: $(tail -1 v2/E1_data/labels_vlm.csv >/dev/null 2>&1 && wc -l < v2/E1_data/labels_vlm.csv) rows; reliability: $(cat v2/E1_data/reliability.json 2>/dev/null)"

log "E2+E3+E4 analyze..."
$PY experiments/v2_analyze.py > logs/v2_analyze.log 2>&1
log "analyze done; E2 gate: $(grep -o '\"first_step_ci_low_gt_0.5\".*' v2/E2_probes/summary.json 2>/dev/null)"

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

log "PIPELINE v2 DONE"
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
