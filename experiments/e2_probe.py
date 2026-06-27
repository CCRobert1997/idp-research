"""E2: train logistic probes for every (step k, block l) pair -> 50x22 AUC map,
best-step curve with 1000x bootstrap CI. Primary label = scheme I (inclusive);
scheme S reported only if non-degenerate.

Outputs: E2_probes/{auc_map.npy, auc_map.csv, best_step_auc_ci.csv, summary.json}
"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATADIR = os.path.join(C.ROOT, "E1_data", "raw")
OUT = os.path.join(C.ROOT, "E2_probes")
os.makedirs(OUT, exist_ok=True)
RS = 42


def load_all():
    files = sorted(glob.glob(os.path.join(DATADIR, "*.npz")))
    n = len(files)
    d0 = np.load(files[0])
    dims = {b: int(d0[f"feat_{b}"].shape[1]) for b in C.BLOCK_NAMES}
    FEAT = {b: np.zeros((n, C.STEPS, dims[b]), np.float16) for b in C.BLOCK_NAMES}
    y_spec = np.zeros(n, np.int64); y_incl = np.zeros(n, np.int64)
    pidx = np.zeros(n, np.int64)
    for i, f in enumerate(files):
        d = np.load(f)
        for b in C.BLOCK_NAMES:
            FEAT[b][i] = d[f"feat_{b}"]
        y_spec[i] = int(d["label"]); y_incl[i] = int(d["label_incl"]); pidx[i] = int(d["prompt_idx"])
    return files, FEAT, y_spec, y_incl, pidx, dims


def probe_auc(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=1000, random_state=RS))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), p


def main():
    t0 = time.time()
    files, FEAT, y_spec, y_incl, pidx, dims = load_all()
    n = len(files)
    print(f"loaded {n} samples in {time.time()-t0:.0f}s; dims={dims}")
    print(f"scheme I dist: 0={int((y_incl==0).sum())} 1={int((y_incl==1).sum())}")
    print(f"scheme S dist: -1={int((y_spec==-1).sum())} 0={int((y_spec==0).sum())} 1={int((y_spec==1).sum())}")

    # ---- primary: scheme I, stratified 80/10/10 (train/val/test), RS=42 ----
    y = y_incl
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=RS)
    val, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=RS)
    print(f"split train={len(tr)} val={len(val)} test={len(te)} "
          f"(test pos={int(y[te].sum())})")

    auc_map = np.zeros((C.STEPS, C.N_BLOCKS), np.float64)
    proba_cache = {}  # (k,l_idx) -> test proba
    for li, b in enumerate(C.BLOCK_NAMES):
        for k in range(C.STEPS):
            Xtr = FEAT[b][tr, k, :].astype(np.float32)
            Xte = FEAT[b][te, k, :].astype(np.float32)
            auc, p = probe_auc(Xtr, y[tr], Xte, y[te])
            auc_map[k, li] = auc
            proba_cache[(k, li)] = p
        print(f"  block {b} ({li+1}/{C.N_BLOCKS}) done  best-step auc={auc_map[:,li].max():.3f}", flush=True)

    np.save(os.path.join(OUT, "auc_map.npy"), auc_map)
    with open(os.path.join(OUT, "auc_map.csv"), "w") as f:
        f.write("step,block_name,auc\n")
        for k in range(C.STEPS):
            for li, b in enumerate(C.BLOCK_NAMES):
                f.write(f"{k+1},{b},{auc_map[k,li]:.5f}\n")

    # ---- best-step curve + bootstrap CI ----
    yte = y[te]
    rng = np.random.RandomState(RS)
    boot_idx = [rng.randint(0, len(te), len(te)) for _ in range(1000)]
    with open(os.path.join(OUT, "best_step_auc_ci.csv"), "w") as f:
        f.write("step,block,auc_mean,ci_low,ci_high\n")
        ci_rows = []
        for k in range(C.STEPS):
            li_star = int(np.argmax(auc_map[k]))
            p = proba_cache[(k, li_star)]
            aucs = []
            for bi in boot_idx:
                yy = yte[bi]
                if yy.min() == yy.max():
                    continue
                aucs.append(roc_auc_score(yy, p[bi]))
            aucs = np.array(aucs)
            lo, hi = np.percentile(aucs, [2.5, 97.5])
            mean = aucs.mean()
            ci_rows.append((k + 1, C.BLOCK_NAMES[li_star], mean, lo, hi))
            f.write(f"{k+1},{C.BLOCK_NAMES[li_star]},{mean:.5f},{lo:.5f},{hi:.5f}\n")

    # ---- headline numbers ----
    k_star, l_star = np.unravel_index(np.argmax(auc_map), auc_map.shape)
    auc_star = auc_map[k_star, l_star]
    first_sig = next((r[0] for r in ci_rows if r[3] > 0.5), None)  # first step ci_low>0.5
    summary = {
        "n": n, "elapsed_s": round(time.time() - t0, 1),
        "scheme_I_dist": {"0": int((y_incl == 0).sum()), "1": int((y_incl == 1).sum())},
        "scheme_S_dist": {"-1": int((y_spec == -1).sum()), "0": int((y_spec == 0).sum()), "1": int((y_spec == 1).sum())},
        "best": {"step": int(k_star + 1), "block": C.BLOCK_NAMES[l_star], "auc": round(float(auc_star), 4)},
        "first_step_ci_low_gt_0.5": first_sig,
        "auc_at_step1_bestblock": round(float(auc_map[0].max()), 4),
        "auc_at_step5": round(float(auc_map[4].max()), 4),
        "auc_at_step10": round(float(auc_map[9].max()), 4),
        "auc_at_step25": round(float(auc_map[24].max()), 4),
        "kill_gate": _gate(first_sig, auc_star),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== E2 SUMMARY ===")
    print(json.dumps(summary, indent=2))


def _gate(first_sig, auc_star):
    if first_sig is None:
        return "KILL (never significant)"
    if first_sig <= 25:
        return f"GO (significant at step {first_sig} <= 25)"
    if auc_star < 0.6:
        return f"KILL (first-sig step {first_sig} > 25 and AUC* {auc_star:.3f} < 0.6)"
    if auc_star >= 0.65:
        return f"DISCUSS DOWNGRADE (late but AUC* {auc_star:.3f} >= 0.65)"
    return f"BORDERLINE (first-sig {first_sig} > 25, AUC* {auc_star:.3f} in [0.6,0.65))"


if __name__ == "__main__":
    main()
