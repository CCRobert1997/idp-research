# IDP — Intermediate Defect Prediction (research task)

Created 2026-06-27. Research-isolated (own dir + own private repo `idp-research`;
NEVER commit into the Talia repo). Code-only backup; data/weights/outputs gitignored.
Pre-registration + kill-gate: `HYPOTHESIS.md`. Paper: AAAI 2027, kill-gate 2026-07-07.

## What this is
Sibling of para-carve (KILL). para-carve tried to *remove* hand defects (failed: decodable
in activations AUC≈0.73 but not causally removable). IDP exploits the *decodable* half:
predict hand-defect outcome from EARLY-step UNet activations and abort bad generations early
(compute saving vs full reject-sampling). Make-or-break: a linear probe on GAP'd activations
at an early step k* separates clean vs defective hands with AUC ≫ chance.

## Setup decisions (2026-06-27)
- venv: `~/shangyu_comfyui/venvs/comfyui` (diffusers 0.31, mediapipe 0.10.35 Tasks-API,
  sklearn 1.8, torch 2.11) — all deps present, NOTHING installed (zero pollution).
- weights: symlink `models_sd15` -> geopara sd15 (fp16 diffusers), `models_clip` -> clip-b32.
  `hand_landmarker.task` symlinked from para_carve assets (this mediapipe build ships only the
  Tasks-API HandLandmarker, not legacy `solutions`).
- 22 UNet resnet blocks (spec's "25" is a typo; auc_map is 50×22). Defect metric = para-carve's
  vetted scale-normalized geometric-anomaly score.

## CRITICAL label finding (pilot, results/pilot_labels.json)
100-img pilot: 27 no-hand, 73 detected; among detected, geom-anomaly ≈0 at ALL percentiles
(max 0.375). => **spec-strict scheme S (detected-clean vs detected-defective) is DEGENERATE**
(positive class 0–1/100; probe untrainable). Reproduces para-carve's bimodal finding (severe
defects = no-detection, which S excludes). **Decision: primary target = scheme I (no-hand-
inclusive): label1 = (no confident hand) OR geom-anomalous, label0 = clean-detected; pilot 72/28.**
IDP under I = "predict early whether the final image will have a well-formed *detectable* hand"
(≈ predict catastrophic hand failure). E1 stores ALL raw signals so labels are re-derivable
(incl. a future Claude-visual-judge gold) without regenerating. Needs Shangyu sign-off.

## Status (2026-06-27, overnight autonomous run)
- E0 sanity: PASS (5 imgs, 22/22 blocks captured, 2.02 MB/img fp16).
- Pilot: DONE (label scheme decided as above).
- E1 DONE (2000 imgs). E2-E5 + figures DONE 2026-06-27 21:33. **Kill-gate: GO** (sig from step1; AUC* 0.84@k14/U1R0; held-out 0.71). See RESULTS.md.
- Orchestrator `experiments/run_pipeline.sh` (tracked bg) chains: E1 done -> E2 probe (kill-gate)
  -> test500 gen -> E4 efficiency -> E3 ablation -> E5 qualitative -> results/PIPELINE_DONE.txt.
- Pending: review E2 kill-gate verdict; GitHub remote `CCRobert1997/idp-research` not yet created
  (local git only) — create + push code after results land / on user OK.

## Code map (experiments/)
idp_common.py (pipe+22-block hooks+HandLandmarker+labels) · e0_sanity · pilot_labelcheck ·
e1_generate (resumable, stores raw) · e2_probe (AUC map+bootstrap CI+gate) · e3_ablation ·
e4_efficiency · e5_qualitative · run_pipeline.sh

See [[project_paracarve_t0_kill]], [[project_geopara_appendix_battery]],
[[feedback_research_talia_isolation]], [[project_research_backup_state]].
