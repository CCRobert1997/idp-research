# IDP — Results (overnight run, 2026-06-27)

**Kill-gate verdict: GO** (pre-registered criterion: first step with bootstrap CI_low > 0.5 is
**step 1** ≤ 25). Defect is linearly decodable from intermediate UNet activations *from the very
first denoising step*. All numbers measured on 2000 generated SD1.5 images; no fabrication.

## Headline (E2 — probing, the strong contribution)
- Probe = logistic regression on GAP'd block activations; label = **scheme I** (see caveat A).
- Class balance: 1390 clean / 610 defective (30.5% positive). Stratified 80/10/10, RS=42.
- **AUC* rises from 0.764 at step 1 → peaks 0.841 at step 14 (block U1R0)**, gently declines after.
  Significant (CI_low > 0.5) at EVERY step. Early-step snapshot: k1=0.763, k5=0.767, k10=0.796,
  k25=0.816. Fig: `figures/fig2_auc_map.png`.
- Signal is spatially localized: mid/upper-decoder + bottleneck blocks (U1R0, U0R1, MidR1, D3R0,
  U2R1, U3R1) carry it; the finest encoder block D0R0 is ~chance.

## Ablations (E3, at k*=14 / U1R0) — `E3_ablation/ablation_table.csv`
| feature source | probe | AUC | note |
|---|---|---|---|
| best block U1R0 | logistic | **0.840** | headline |
| best block U1R0 | MLP(64) | 0.824 | **no gain over linear → linearly decodable** |
| concat all 22 blocks | logistic | 0.818 | **no gain over single block → localized** |
| final latent z0 (GAP 4D) | logistic | 0.550 | raw latent summary ~uninformative |
| pixel CLIP-img (final image, 512d) | logistic | 0.759 | **early activation (0.84) BEATS post-hoc CLIP on the finished image (0.76)** |

The last row is the crux: a step-14 internal activation predicts the defect *better* than a CLIP
read of the fully-rendered image — and 36 steps earlier.

## Efficiency / early-stop (E4 — the weaker half) — `E4_efficiency/`
Trained on all E1, evaluated on **fresh held-out test500** (seeds 1000.., raw defect 27%).
- **Held-out probe AUC = 0.709** — notably below the in-distribution 0.84. The 0.84 is optimistic
  (k*,ℓ* were selected on the in-dist test); **0.71 is the honest deployment number.**
- At τ=0.1: defect among accepted 27% → **16%**, abort 41%, **59.9 UNet calls/accepted vs
  full-reject-sampling 68.5 (−12.6%)**; oracle = 55.2. IDP sits on the Pareto between no-filter
  (50 calls, 27% defect) and full-reject (68.5, ~0%), a real but **modest** win bounded by the
  0.71 held-out AUC. Fig: `figures/fig3_pareto.png`.

## Qualitative (E5) — `E4_efficiency/qualitative/`
TP examples are no-hand catastrophic renders (probe score 1.0); TN a clean detected hand (0.0);
the FP is "a person counting on their fingers" (MediaPipe detected hands, probe still flagged) —
the hardest prompt.

## CAVEATS (must be in the paper)
- **A. Label semantics.** Scheme-strict (spec) is DEGENERATE: 0 geom-defective among *detected*
  hands (pilot + full run). ALL 610 scheme-I positives are the **no-hand** bucket → IDP here
  predicts **hand presence/detectability ≈ catastrophic hand failure, NOT fine malformation**
  (reproduces para-carve's bimodal finding). Honest, but the "defect" is coarse. Upgrade path:
  Claude-as-visual-judge gold labels (re-derivable from stored PNGs+landmarks, no regeneration) to
  test whether activations also predict *subtle* malformation — the more ambitious claim.
- **B. Generalization gap.** 0.84 in-dist (selected) vs 0.71 fresh held-out. Report 0.71 as headline
  deployment AUC; the efficiency win is modest because of it.
- **C. Deviations from spec.** 22 blocks (spec "25" = typo; auc_map is 50×22); primary label =
  scheme I not S; output dir = standalone `IDP_research/` not `talia_research/IDP/` (isolation).

## Paper framing recommendation
GO on the **probing/decodability** story (strong, clean, novel: early + linear + localized + beats
post-hoc pixel CLIP). Present **early-stop efficiency as a downstream application**, honestly modest
(−12.6% vs reject-sampling at 0.71 held-out AUC). Needs Shangyu sign-off on label scheme A before
writing §Setup. Data/figs/CSVs all under E2/E3/E4 + figures/.
