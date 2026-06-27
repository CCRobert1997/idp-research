"""Pilot: generate 5 imgs/prompt (100 total, NO activation capture) and measure
the label distribution under the spec's scheme and alternatives, so we don't burn
the 2-4h E1 on a degenerate probe target. Writes results/pilot_labels.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C

N_PER = 5
OUT = os.path.join(C.ROOT, "results", "pilot_labels.json")

def main():
    C.get_pipe()
    recs = []
    t0 = time.time()
    for pi, prompt in enumerate(C.PROMPTS):
        for s in range(N_PER):
            pil, _, _ = C.generate(prompt, seed=s, capture=False)
            a = C.analyze_hands(pil)
            recs.append({"prompt_idx": pi, "seed": s, "num_hands": a["num_hands"],
                         "det_score": a["det_score"], "best_geom": a["best_geom"],
                         "finger_count": a["finger_count"],
                         "per_hand_scores": a["per_hand_scores"],
                         "per_hand_geom": a["per_hand_geom"]})
        done = (pi + 1) * N_PER
        print(f"[{done}/{len(C.PROMPTS)*N_PER}] {time.time()-t0:.0f}s  prompt='{prompt}'", flush=True)

    # --- label distributions ---
    def dist_spec(geom_tau, det_conf=C.DET_CONF):
        c = {-1: 0, 0: 0, 1: 0}
        for r in recs:
            c[C.compute_label(r, det_conf=det_conf, geom_tau=geom_tau)] += 1
        return c
    # no-hand-inclusive scheme: label1 = (no confident hand) OR (geom>tau); label0 = clean detected
    def dist_inclusive(geom_tau, det_conf=C.DET_CONF):
        c = {0: 0, 1: 0}
        for r in recs:
            conf = [x for x in r["per_hand_scores"] if x >= det_conf]
            bad = (not conf) or (r["best_geom"] > geom_tau)
            c[1 if bad else 0] += 1
        return c

    geoms_det = [r["best_geom"] for r in recs if r["best_geom"] < 9]
    summary = {
        "n": len(recs), "elapsed_s": round(time.time() - t0, 1),
        "sec_per_img": round((time.time() - t0) / len(recs), 2),
        "n_no_hand": sum(1 for r in recs if not [x for x in r["per_hand_scores"] if x >= C.DET_CONF]),
        "geom_detected_pctl": {p: round(float(np.percentile(geoms_det, p)), 3) for p in [10, 25, 50, 75, 90, 95]} if geoms_det else {},
        "geom_detected_max": round(float(np.max(geoms_det)), 3) if geoms_det else None,
        "spec_scheme_label_counts": {str(tau): dist_spec(tau) for tau in [0.2, 0.3, 0.5, 0.8]},
        "inclusive_scheme_label_counts": {str(tau): dist_inclusive(tau) for tau in [0.2, 0.3, 0.5, 0.8]},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "recs": recs}, f, indent=2)
    print("\n=== PILOT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("->", OUT)

if __name__ == "__main__":
    main()
