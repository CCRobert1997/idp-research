"""v2 E5: early-stop efficiency demo (explicitly a downstream application, modest).
Train probe on ALL E1 (k*,l*); evaluate on fresh held-out E5_heldout (VLM-labeled).
Outputs E5_efficiency/{efficiency_table.csv, summary.json, scores.npz}."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
import v2_common as V

OUT = os.path.join(V.V2ROOT, "E5_efficiency"); os.makedirs(OUT, exist_ok=True)
T = C.STEPS


def feats_at(D, block, k):
    return D["FEAT"][block][:, k].astype(np.float32)


def main():
    ks = json.load(open(os.path.join(V.V2ROOT, "E4_ablation", "kstar.json")))
    k_star, l_star = int(ks["k_star"]) - 1, ks["l_star"]
    Dtr = V.load_feats("E1_data", conf_min=0.6, blocks=[l_star])
    Dte = V.load_feats("E5_heldout", conf_min=0.6, blocks=[l_star])
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=V.RS))
    clf.fit(feats_at(Dtr, l_star, k_star), Dtr["y"])
    yte = Dte["y"]; s = clf.predict_proba(feats_at(Dte, l_star, k_star))[:, 1]
    auc = roc_auc_score(yte, s); raw = float(yte.mean()); kc = k_star + 1
    np.savez(os.path.join(OUT, "scores.npz"), scores=s, labels=yte, files=Dte["files"])
    print(f"held-out E5: n={len(yte)} raw_defect={raw:.3f} probe_AUC={auc:.3f}")

    rows = ["method,threshold,defect_rate,unet_calls_per_accepted,fpr,fnr,abort_rate"]
    calls = lambda na, nc: (na * kc + nc * T) / max(nc, 1)
    pts = []
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ab = s > tau; ac = ~ab; nac = int(ac.sum()); nab = int(ab.sum())
        dr = float(yte[ac].mean()) if nac else 0.0
        fpr = float((ab & (yte == 0)).sum() / max((yte == 0).sum(), 1))
        fnr = float((ac & (yte == 1)).sum() / max((yte == 1).sum(), 1))
        rows.append(f"IDP,{tau},{dr:.4f},{calls(nab,nac):.3f},{fpr:.4f},{fnr:.4f},{nab/len(yte):.4f}")
        pts.append({"tau": tau, "defect_rate": dr, "unet_calls": calls(nab, nac), "abort_rate": nab / len(yte)})
    rows.append(f"NoFilter,-,{raw:.4f},{float(T):.3f},0,1,0")
    frc = T / (1 - raw) if raw < 1 else float("inf")
    rows.append(f"FullRejectSampling,-,0,{frc:.3f},0,0,{raw:.4f}")
    nd, ncl = int((yte == 1).sum()), int((yte == 0).sum())
    rows.append(f"Oracle,-,0,{(nd*kc+ncl*T)/max(ncl,1):.3f},0,0,{raw:.4f}")
    rng = np.random.RandomState(V.RS)
    for p in pts:
        ab = rng.rand(len(yte)) < p["abort_rate"]; ac = ~ab; nac = int(ac.sum())
        rows.append(f"RandomAbort,{p['tau']},{float(yte[ac].mean()) if nac else 0:.4f},{calls(int(ab.sum()),nac):.3f},-,-,{p['abort_rate']:.4f}")
    open(os.path.join(OUT, "efficiency_table.csv"), "w").write("\n".join(rows) + "\n")
    cand = [p for p in pts if p["defect_rate"] <= raw / 2]
    pick = min(cand, key=lambda p: p["unet_calls"]) if cand else min(pts, key=lambda p: p["defect_rate"])
    summ = {"k_star": kc, "l_star": l_star, "n": len(yte), "raw_defect_rate": round(raw, 4),
            "heldout_auc": round(float(auc), 4), "full_reject_calls": round(frc, 3),
            "tau_target": pick["tau"], "idp_at_target": {k: round(v, 4) for k, v in pick.items()},
            "savings_vs_full_reject_pct": round(100 * (1 - pick["unet_calls"] / frc), 1)}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("E5 summary:", json.dumps(summ))


if __name__ == "__main__":
    main()
