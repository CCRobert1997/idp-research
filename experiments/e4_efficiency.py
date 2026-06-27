"""E4: early-stop efficiency. Train final probe on ALL of E1 (scheme I) at (k*,l*),
evaluate on held-out test500 (seeds 1000..). Sweep abort thresholds vs baselines.

Prereq: test500 generated at E4_efficiency/test500/raw (via e1_generate --outdir).
Outputs: E4_efficiency/{efficiency_table.csv, summary.json}, probe scores npz.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

E1DIR = os.path.join(C.ROOT, "E1_data", "raw")
TESTDIR = os.path.join(C.ROOT, "E4_efficiency", "test500", "raw")
OUT = os.path.join(C.ROOT, "E4_efficiency")
RS = 42
T = C.STEPS  # 50


def feats_at(files, block, k):
    return np.stack([np.load(f)[f"feat_{block}"][k] for f in files]).astype(np.float32)


def main():
    summ = json.load(open(os.path.join(C.ROOT, "E2_probes", "summary.json")))
    k_star = int(summ["best"]["step"]) - 1
    l_star = summ["best"]["block"]
    print(f"k*={k_star+1} (cost {k_star+1} unet calls if abort) l*={l_star}")

    tr_files = sorted(glob.glob(os.path.join(E1DIR, "*.npz")))
    te_files = sorted(glob.glob(os.path.join(TESTDIR, "*.npz")))
    assert te_files, f"no test500 at {TESTDIR} — generate first"
    ytr = np.array([int(np.load(f)["label_incl"]) for f in tr_files])
    yte = np.array([int(np.load(f)["label_incl"]) for f in te_files])
    Xtr = feats_at(tr_files, l_star, k_star)
    Xte = feats_at(te_files, l_star, k_star)

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=RS))
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, s)
    raw = float(yte.mean())  # raw defect rate on test500
    kc = k_star + 1
    np.savez(os.path.join(OUT, "test500_scores.npz"), scores=s, labels=yte,
             files=[os.path.basename(f) for f in te_files])
    print(f"held-out test500: n={len(yte)} raw_defect_rate={raw:.3f} probe_AUC={auc:.3f}")

    rows = ["method,threshold,defect_rate,unet_calls_per_accepted,fpr,fnr,abort_rate,accept_frac"]

    def calls(n_abort, n_acc):
        return (n_abort * kc + n_acc * T) / max(n_acc, 1)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    idp_pts = []
    for tau in thresholds:
        abort = s > tau
        acc = ~abort
        n_acc = int(acc.sum()); n_ab = int(abort.sum())
        dr = float(yte[acc].mean()) if n_acc else 0.0
        fpr = float((abort & (yte == 0)).sum() / max((yte == 0).sum(), 1))
        fnr = float((acc & (yte == 1)).sum() / max((yte == 1).sum(), 1))
        upc = calls(n_ab, n_acc)
        rows.append(f"IDP,{tau},{dr:.4f},{upc:.3f},{fpr:.4f},{fnr:.4f},{n_ab/len(yte):.4f},{n_acc/len(yte):.4f}")
        idp_pts.append({"tau": tau, "defect_rate": dr, "unet_calls": upc,
                        "fpr": fpr, "fnr": fnr, "abort_rate": n_ab / len(yte)})

    # baselines
    rows.append(f"NoFilter,-,{raw:.4f},{float(T):.3f},0.0000,1.0000,0.0000,1.0000")
    full_reject_calls = T / (1 - raw) if raw < 1 else float("inf")
    rows.append(f"FullRejectSampling,-,0.0000,{full_reject_calls:.3f},0.0000,0.0000,{raw:.4f},{1-raw:.4f}")
    # Oracle: perfectly abort the defective at k*
    n_def = int((yte == 1).sum()); n_clean = int((yte == 0).sum())
    oracle_calls = (n_def * kc + n_clean * T) / max(n_clean, 1)
    rows.append(f"Oracle,-,0.0000,{oracle_calls:.3f},0.0000,0.0000,{raw:.4f},{1-raw:.4f}")
    # Random abort matched to IDP abort rate at each tau (defect_rate stays ~raw)
    rng = np.random.RandomState(RS)
    for pt in idp_pts:
        ar = pt["abort_rate"]
        ab = rng.rand(len(yte)) < ar
        acc = ~ab
        n_acc = int(acc.sum()); n_ab = int(ab.sum())
        dr = float(yte[acc].mean()) if n_acc else 0.0
        upc = calls(n_ab, n_acc)
        rows.append(f"RandomAbort,{pt['tau']},{dr:.4f},{upc:.3f},-,-,{ar:.4f},{n_acc/len(yte):.4f}")

    with open(os.path.join(OUT, "efficiency_table.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # Table-2 pick: lowest unet_calls among IDP points with defect_rate <= raw/2
    cand = [p for p in idp_pts if p["defect_rate"] <= raw / 2]
    pick = min(cand, key=lambda p: p["unet_calls"]) if cand else min(idp_pts, key=lambda p: p["defect_rate"])
    summary = {
        "k_star": kc, "l_star": l_star, "test500_n": len(yte),
        "raw_defect_rate": round(raw, 4), "probe_auc_heldout": round(float(auc), 4),
        "full_reject_calls": round(full_reject_calls, 3), "oracle_calls": round(oracle_calls, 3),
        "tau_target": pick["tau"], "idp_at_target": {k: round(v, 4) for k, v in pick.items()},
        "no_filter_calls": T, "savings_vs_full_reject_pct": round(100 * (1 - pick["unet_calls"] / full_reject_calls), 1),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== E4 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
