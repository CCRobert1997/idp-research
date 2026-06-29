# IDP v2 — pre-registration (2026-06-28)

**Big pivot from v1.** Lit-Radar found the "mid-denoising probe predicts final quality + early-stop"
mechanism is already taken by 2 concurrent 2026 works (Probe-Select 2603.02829, Diffusion Probe
CVPR'26 2602.23783). **v2 re-positions as an INTERPRETABILITY characterization of structural-defect
formation.** Five pillars: linear probe · structural-defect-specific · AUC-vs-step timing · layer
localization · **per-defect-type tracks (E3)** — the last is the differentiator the collision works
don't do. Target AAAI 2027 (abstract 07-20, full 07-27).

## Design
- Reuses v1 infra verbatim (idp_common: 22-block hooks, GAP, logistic probe, bootstrap CI).
- E1: 6 defect-type prompt pools × 12 prompts × 70 seeds = **5040 images** (diverse content, not
  just hands). prompt_type ∈ {HAND,FACE,LIMB,COUNT,PHYS,TEXT} = the E3 track axis.
- **E1.5: VLM visual judge replaces MediaPipe** = Qwen2.5-VL-7B-Instruct on-server (user chose the
  open-source route; ANTHROPIC_API_KEY was available but declined for the no-external-call habit).
  Binary `defective` + multi-label `defect_types` + severity + confidence. conf≥0.6 used in probes.
- E2 overall AUC map + layer_ranking; **E3 per-type AUC-vs-step (CORE)**; E4 ablation; E5 early-stop
  demo (held-out 504, seeds 2000.., explicitly a modest downstream app); E6 qualitative.

## ⚠ Rubric deviation (2026-06-28) — APPROVED by Shangyu 2026-06-29
The spec's pre-registered rubric ("Be conservative: only mark defective=true if a non-expert would
clearly notice") made Qwen2.5-VL-7B flag **~0% defects** on a calibration set — it called a blatantly
garbled "OPEN" storefront sign and a melted crowd-of-faces "clean" (verified by eye). Running 5040
like that = a dead, degenerate target (the v1 failure again). **I replaced it with a checklist +
reason-then-JSON rubric** (explicit hand/face/limb/count/physics/text checks; describe-then-judge;
dropped the over-conservative clause). On the same calibration images it restored discrimination:
garbled-text/melted-face/fused-tentacles/noodle-arm all caught, genuinely-clean stayed clean. This is
a labeling-QUALITY fix, NOT results-biasing (it doesn't push toward any hypothesis). Validated by
E1.5 reliability (Cohen's κ, primary-vs-second-pass) + Shangyu's planned 100-image human spot-check.
Residual known noise: COUNT under-detected; defect_type sometimes mis-assigned (binary `defective`,
which E2/E3-binary use, is sound). Full rubric in `experiments/v2_e15_label.py`.

## Kill-check (after E2/E3)
- Overall best AUC* still significant (v1 had strong GO) — expected to hold.
- **E3: ≥3 defect types show clean early-segment signal AND there is tellable inter-type difference**
  (e.g. TEXT decodable earlier/later than HAND/FACE; COUNT only late) → differentiation holds → GO.
- If all 6 types collapse onto one indistinguishable curve → differentiation dead → report to Shangyu,
  rethink angle.

## Integrity
Everything measured, no fabrication. Don't move the kill-check. Report the rubric deviation, the
class balance, the reliability κ, and any weak/degenerate type honestly. Output dir = standalone
`IDP_research/v2/` (NOT talia_research/, isolation [[feedback_research_talia_isolation]]).
