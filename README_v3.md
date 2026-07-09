# IDP v3 — local handoff (cross-architecture: SD1.5 / SD3.5 / FLUX)

Pulled bundle = **lightweight results only** (CSV/JSON/npy/figures/code). The heavy raw data
(activations `*.npz`, images `*.png`) stays on the server — see "If you need the raw" below.

## What's in here
```
IDP_v3/
  M2_SD35/                      ← SD3.5 (DiT), FINAL
    labels.csv                  4436 rows: file,defective,defect_types,severity,confidence,note
    manifest.csv                file → prompt_type,prompt_text,seed,model
    reliability.json            judge self-consistency κ=0.733 (NOT human κ)
    E2_probes/
      summary.json              ★ overall probe headline (AUC, best step/block, sig step)
      auc_map.npy               (28 steps × 24 blocks) AUC grid
      best_step_auc_ci.csv      best-block AUC vs step + bootstrap CI  → Fig
      layer_ranking.csv         block, peak_auc, peak_step, norm_step  → layer-localization
    E3_pertype/pertype_summary.csv   per-type (COUNT,TEXT viable; rest underpowered)
    E4_ablation/ablation_table.csv   linear vs MLP / single vs concat / vs final-latent
    figures/fig_M2_auc.png      ★ heatmap + best-step AUC curve
  M3_FLUX/                      ← FLUX (DiT), PILOT (720) — full run in progress on server
    labels.csv, manifest.csv    pilot labels; per-type defect rates (COUNT 8.3%, TEXT 10.8%, anatomy 0)
  code/                         all v3 experiment scripts
  README_v3.md                  this file
```
M1 (SD1.5) = the v2 bundle you already pulled (`Research/IDP/v2/`).

## The numbers (final for M2; pilot for M3)
- **M2 overall:** AUC **0.886 @ step 2 / block B22**, significant from step 1 (`E2_probes/summary.json`).
- **M2 ablation:** linear 0.886 > MLP 0.859 > concat-24 0.795; final-latent 0.616 (`E4_ablation/ablation_table.csv`).
- **M2 per-type:** COUNT(n=90) 0.986@step9/B07; TEXT(n=35) 0.981@step9/B10 (`E3_pertype/...`).
- **M3 pilot defect rate:** overall 3.5%; COUNT 8.3% TEXT 10.8% PHYS 1.7% HAND/FACE/LIMB 0.
- **Label caveat (write in paper):** human-vs-judge κ=0.26 (agree 63%) — VLM "instrument-labels",
  reported as a material limitation. Judge self-consistency κ=0.733.

## Map to the paper (AAAI2027/sec/experiments.tex \PH{...})
| paper slot | source file |
|---|---|
| §Setup defect rates / label validity | each model `labels.csv` + `reliability.json` + κ=0.26 caveat |
| Overall-probe AUC + "early/linear" | `M*/E2_probes/summary.json` + `E4_ablation/ablation_table.csv` |
| Figure (AUC heatmap + curve) | `M*/figures/fig_M2_auc.png` (+ M1/M3 once made) |
| Layer-localization table | `M*/E2_probes/layer_ranking.csv` |
| Per-type (COUNT/TEXT) | `M*/E3_pertype/pertype_summary.csv` |
| ★ Cross-arch main fig/table (E7) | pending: all 3 models' auc_map + layer_ranking, normalized step (k/T) |

## How to "对接" your local process
**Paper-writing (recommended):** these CSV/JSON/png are the deliverables — read the numbers straight
into the `\PH{...}` slots and drop the figures into LaTeX. Everything is small + diff-able.

**Re-run analysis locally:** the analysis is plain sklearn (`code/v3_analyze.py`, `v2_common.py`).
It needs the activation `.npz` (heavy: M2 ~9 GB, M3 ~18 GB) which are NOT in this bundle. Either
(a) re-run on the server (`OMP_NUM_THREADS=8 python experiments/v3_analyze.py --model M2`), or
(b) pull the npz dirs (`rsync .../v3/M2/E1_data/raw/`) and run locally.

**Regenerate figures locally:** `auc_map.npy` + `best_step_auc_ci.csv` are all a plot needs — no GPU.

## If you need the raw (images / activations)
On the server: `~/shangyu_comfyui/IDP_research/v3/{M2,M3}/E1_data/{raw,png}/`. Tell me which slice
(e.g. qualitative example images per type, or full npz for local re-analysis) and I'll bundle it.
