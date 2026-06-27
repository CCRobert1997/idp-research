# IDP — Intermediate Defect Prediction (pre-registration)

Created 2026-06-27. Research-isolated project (own dir `~/shangyu_comfyui/IDP_research/`,
own private repo `idp-research` — NOT inside the Talia repo). Code-only backup; data/outputs/weights gitignored.
Paper target: AAAI 2027. Kill-gate: 2026-07-07.

## Core hypothesis
Hand-defect outcomes of an SD1.5 generation are **linearly decodable from intermediate UNet
activations at an EARLY denoising step** — early enough that aborting bad generations saves
compute (intermediate defect prediction → early stopping).

Make-or-break (H1): there exists a step k* and block ℓ* such that a logistic probe on the
GAP'd activation separates clean vs defective hands with AUC well above chance, and k* is early.

## Relationship to prior work (honest)
Sibling of **para-carve** (KILL): para-carve tried to *remove* hand defects via low-rank weight
carving and failed — defect is decodable in activations (AUC≈0.73) but not causally removable.
IDP exploits the *decodable* half: don't fix it, *predict & abort* early. The para-carve finding
that "defect IS decodable from activations" is the encouraging prior; the open question is whether
it's decodable EARLY and cheaply (linear), and whether early-abort beats full reject-sampling on
the compute/quality Pareto.

## KNOWN RISK surfaced at setup (E0, 2026-06-27)
Para-carve established SD hand failures are **bimodal**: severe defects mostly produce
**no MediaPipe detection** (label -1), while *detected* hands are mostly geometrically clean.
The spec's probe (E2) trains only on `label∈{0,1}` (confidently-detected hands), EXCLUDING the
no-hand bucket where most real defects live → risk of a degenerate/tiny positive class.
Mitigation: E1 stores ALL raw MediaPipe signals per image (landmarks, per-hand conf, per-hand
geom-anomaly, finger counts) so E2 can compute the AUC map under MULTIPLE label definitions
WITHOUT re-running the expensive generation:
- **S (spec-strict):** -1 no-hand (excluded) / 0 clean-detected / 1 geom-defective-detected.
- **I (no-hand-inclusive):** 0 clean-detected / 1 = (no-hand OR geom-defective). Matches the
  actual "is this generation's hands bad?" question and is non-degenerate.
Both reported; the kill-gate is evaluated on whichever is the well-posed target (decided from the
pilot's measured class balance, recorded in results/pilot_labels.json).

## Locked spec
- SD1.5 (geopara fp16 diffusers weights), DDIM, T=50, guidance 7.5, 512², fp16, A100D-40C.
- 20 hand prompts × 100 seeds (0..99) = 2000 imgs. test500 = seeds 1000..1499.
- 22 UNet resnet blocks (8 down + 2 mid + 12 up — the spec's "25" is a typo; auc_map is 50×22),
  GAP per step, all 50 steps. Defect metric = para-carve's vetted scale-normalized geom anomaly.
- Probe = sklearn LogisticRegression(C=1, max_iter=1000, random_state=42); 80/10/10 stratified
  split, random_state=42; 1000× bootstrap CI on the best-step curve.

## Kill-gate (2026-07-07), from best_step_auc_ci.csv (first step k with ci_low>0.5)
- k ≤ 25 (first half)            → **GO**, write the paper.
- k > 25 AND AUC* < 0.6          → **KILL**, report to Shangyu.
- k > 25 AND AUC* ≥ 0.65         → discuss downgrade (late prediction still has value).

## Integrity
Everything measured, no fabrication. No goalpost-moving on the kill-gate. Class balance and the
chosen label scheme are reported honestly even if they weaken the headline.
