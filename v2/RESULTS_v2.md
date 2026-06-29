# IDP v2 — Results (2026-06-29)

**Kill-check verdict: GO** (with honest tempering — see §Differentiation). All measured on 5040
SD1.5 images, VLM-judged. No fabrication.

## E1.5 labeling (Qwen2.5-VL-7B on-server, refined rubric)
- 5040 imgs, defect_rate **44.9%** (balanced), only 4 dropped at conf<0.6.
- Reliability (primary greedy vs 2nd-pass temp0.8 on 500): **77.2% agreement, Cohen's κ=0.535**
  (moderate; below the 80-93% human-agreement target — an honest limitation of a 7B judge).
- Per-type base defect rate: LIMB 80 · HAND 70 · TEXT 61 · FACE 27 · COUNT 21 · PHYS 11 (%).

## E2 overall probe (defect ~ activation)
- Best **AUC* 0.846 @ step 1, block D0R1**; significant (CI_low>0.5) from step 1; AUC FLAT across
  steps (0.846→0.829 over k=1..25). Top blocks: D0R1,D1R0,U2R1,D1R1,U2R2 (early encoder dominates).
- Reading: structural-defect outcome is **decodable from the first denoising step and stays
  decodable** — defects are largely determined early (consistent with v1's step-1 AUC 0.76).

## E3 per-type tracks (the differentiator) — all 6 significant from step 1
| type | n_pos | peak AUC | peak step | peak block |
|---|---|---|---|---|
| text    | 451 | 0.970 | 7 | D3R0 (bottleneck entry) |
| physics | 90  | 0.962 | 3 | U1R0 |
| limb    | 452 | 0.954 | 3 | U0R0 |
| hand    | 1098| 0.925 | 1 | U2R2 |
| face    | 269 | 0.902 | 1 | U1R0 |
| count   | 55  | 0.858 | 3 | U2R0 |

**Honest read of Fig 3 (`figures/fig3_pertype_auc.png`):** the curves largely OVERLAP high & flat
(~0.9–0.97) — every structural-defect type is decodable early and persistently. The differentiation
is therefore NOT a dramatic per-type temporal ordering; it lives in:
1. **COUNT is the clear outlier** — lowest AUC, uniquely DECLINING over denoising (0.85→0.72), wide
   CI (underpowered n=55). Interpretable: counting is a global/semantic property, hardest to read
   from local activations and it degrades as the model commits to layout. This is the most tellable
   inter-type difference and a genuine finding.
2. **Layer localization** — each type peaks at a different block (text→bottleneck D3R0; hand→U2R2;
   limb→U0R0; face/physics→U1R0), i.e. different defect classes are read out from different depths.
3. **TEXT peaks latest (step 7) and deepest (D3R0)** — letters need more denoising to commit.

So differentiation HOLDS (kill-check pass: ≥3 clean early + tellable differences), but the headline
is "structural defects are broadly encoded early+persistently, read from class-specific layers, with
counting the semantic exception" — NOT "six cleanly time-separated tracks." Report it that way.

## E4 ablation (`E4_ablation/ablation_table.csv`, at k*=1/D0R1)
| feature | AUC | note |
|---|---|---|
| best block D0R1 (logistic) | **0.846** | headline |
| MLP(64) same block | 0.790 | linear BEATS MLP → linearly decodable / MLP overfits |
| concat 22 blocks | 0.801 | worse than single block (dilution/overfit) |
| final latent z0 (4D) | 0.570 | ~uninformative |
| pixel CLIP (final image) | 0.827 | activation 0.846 only MARGINALLY beats post-hoc CLIP (+0.019) |

Caveat: the "beats post-hoc CLIP" pillar is WEAK here (+0.019, vs v1's +0.08) — VLM-defined defects
are visible in the final image so CLIP reads them well. The honest activation advantage is "as good
as post-hoc CLIP but available at step 1 + linearly + layer-localized," not "much better."

## E5 efficiency (held-out 503, `E5_efficiency/`) — downstream app, downplayed
- raw defect 48.5%, held-out probe AUC **0.761** (vs in-dist 0.846 — generalization gap).
- k*=1 ⇒ aborting costs ONE UNet call. At τ=0.1: defect 48.5%→20%, abort 64%, 51.8 calls/accepted
  vs full-reject 97.1 (**−46.7%**), ~no-filter cost (50). Strong because abort is nearly free at k=1.
- Fig 4 Pareto. Honest: report 0.761 held-out as the deployment number.

## Caveats for the paper (must state)
1. Judge κ=0.535 (moderate); 7B VLM self-consistency limited. Mitigate with human spot-check.
2. Rubric was refined from the spec's (which flagged ~0%); see HYPOTHESIS_v2.md — needs sign-off.
3. count/physics underpowered (n_pos 55/90); count CI wide.
4. defect_type tags noisy (the binary `defective` is the reliable signal).
5. Generalization gap 0.846→0.761. CLIP-advantage marginal.

## Recommendation
GO. Lead with: early + linear + layer-localized decodability of structural defects across 6 classes,
with the COUNT semantic exception and class-specific readout layers as the novel, collision-free
contribution. Temper the per-type *temporal* claim (curves overlap). Files: E2/E3/E4/E5 + figures.
