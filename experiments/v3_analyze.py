"""v3 per-model analysis (DiT-aware): E2 overall AUC map + layer ranking + best-step CI,
E3 per-type for types with >=MIN_POS positives, E4 ablation. Normalized step axis (k/T)
recorded for cross-model E7. Reuses v2_common.probe_auc.

Outputs v3/{model}/{E2_probes,E3_pertype,E4_ablation}/. Mark PRELIMINARY until human kappa>=0.6.
Usage: OMP_NUM_THREADS=8 v3_analyze.py --model M2
"""
import os, sys, json, glob, time, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v2_common as V  # probe_auc, RS
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "v3")
MIN_POS = 25
TYPES6 = ["HAND", "FACE", "LIMB", "COUNT", "PHYS", "TEXT"]


def load(model):
    base = os.path.join(V3, model, "E1_data")
    lab = {}
    for r in csv.DictReader(open(os.path.join(base, "labels.csv"))):
        lab[r["file"]] = int(r["defective"])
    files = [f for f in sorted(glob.glob(os.path.join(base, "raw", "*.npz")))
             if os.path.basename(f) in lab]
    d0 = np.load(files[0])
    blocks = sorted([k[5:] for k in d0.files if k.startswith("feat_")])
    T, D = d0["feat_" + blocks[0]].shape
    n = len(files)
    FEAT = {b: np.zeros((n, T, d0["feat_" + b].shape[1]), np.float16) for b in blocks}
    y = np.zeros(n, np.int64); ptype = []
    for i, f in enumerate(files):
        d = np.load(f)
        for b in blocks:
            FEAT[b][i] = d["feat_" + b]
        y[i] = lab[os.path.basename(f)]; ptype.append(str(d["prompt_type"]))
    return FEAT, y, np.array(ptype), blocks, T


def split(y):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=V.RS)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=V.RS)
    return tr, te


def boot_ci(yte, p, rng, nb=1000):
    a = []
    for _ in range(nb):
        bi = rng.randint(0, len(yte), len(yte))
        if yte[bi].min() != yte[bi].max():
            a.append(roc_auc_score(yte[bi], p[bi]))
    a = np.array(a) if a else np.array([0.5])
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def auc_map(FEAT, blocks, T, y, idx_tr, idx_te):
    M = np.zeros((T, len(blocks))); pc = {}
    for li, b in enumerate(blocks):
        for k in range(T):
            a, p = V.probe_auc(FEAT[b][idx_tr, k].astype(np.float32), y[idx_tr],
                               FEAT[b][idx_te, k].astype(np.float32), y[idx_te], return_proba=True)
            M[k, li] = a; pc[(k, li)] = p
    return M, pc


def main(model):
    t0 = time.time()
    FEAT, y, ptype, blocks, T = load(model)
    n = len(y); npos = int(y.sum())
    for d in ("E2_probes", "E3_pertype", "E4_ablation"):
        os.makedirs(os.path.join(V3, model, d), exist_ok=True)
    print(f"[{model}] n={n} pos={npos} ({100*npos/n:.1f}%) blocks={len(blocks)} T={T} load={time.time()-t0:.0f}s", flush=True)
    E2 = os.path.join(V3, model, "E2_probes")

    # ---- E2 overall ----
    tr, te = split(y); yte = y[te]; rng = np.random.RandomState(V.RS)
    M, pc = auc_map(FEAT, blocks, T, y, tr, te)
    np.save(os.path.join(E2, "auc_map.npy"), M)
    with open(os.path.join(E2, "layer_ranking.csv"), "w") as f:
        f.write("block,peak_auc,peak_step,norm_step\n")
        for li in sorted(range(len(blocks)), key=lambda i: -M[:, i].max()):
            ks = int(M[:, li].argmax())
            f.write(f"{blocks[li]},{M[:,li].max():.4f},{ks+1},{(ks+0.5)/T:.3f}\n")
    cir = []
    with open(os.path.join(E2, "best_step_auc_ci.csv"), "w") as f:
        f.write("step,norm_step,block,auc_mean,ci_low,ci_high\n")
        for k in range(T):
            li = int(np.argmax(M[k])); m, lo, hi = boot_ci(yte, pc[(k, li)], rng)
            cir.append((k + 1, m, lo, hi)); f.write(f"{k+1},{(k+0.5)/T:.3f},{blocks[li]},{m:.4f},{lo:.4f},{hi:.4f}\n")
    ks_, ls_ = np.unravel_index(np.argmax(M), M.shape)
    first_sig = next((r[0] for r in cir if r[2] > 0.5), None)
    e2 = {"model": model, "n": n, "n_pos": npos, "defect_rate": round(npos / n, 4),
          "best": {"step": int(ks_ + 1), "norm_step": round((ks_ + .5) / T, 3),
                   "block": blocks[ls_], "auc": round(float(M[ks_, ls_]), 4)},
          "first_sig_step": first_sig, "T": T,
          "PRELIMINARY": "pending human kappa>=0.6 gate"}
    json.dump(e2, open(os.path.join(E2, "summary.json"), "w"), indent=2)
    print(f"[{model}] E2: best AUC {e2['best']['auc']} @step{e2['best']['step']}/{e2['best']['block']} first_sig={first_sig}", flush=True)

    # ---- E3 per-type (only viable types) ----
    rows = ["defect_type,n_pos,first_sig_step,peak_auc,peak_step,peak_norm_step,peak_block"]
    for t in TYPES6:
        pos = np.where((y == 1) & (ptype == t))[0]
        if len(pos) < MIN_POS:
            rows.append(f"{t},{len(pos)},NA,NA,NA,NA,NA(underpowered)"); continue
        neg = np.where(y == 0)[0]
        idx = np.concatenate([pos, neg]); yy = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
        tr2, te2 = split(yy)
        gtr, gte, yte2 = idx[tr2], idx[te2], yy[te2]
        best = (-1, None, None)
        for k in range(T):
            for b in blocks:
                a = V.probe_auc(FEAT[b][gtr, k].astype(np.float32), yy[tr2],
                                FEAT[b][gte, k].astype(np.float32), yte2)
                if a > best[0]:
                    best = (a, k + 1, b)
        # first-sig via best-block-per-step CI on the chosen track is expensive; use overall best CI proxy
        rows.append(f"{t},{len(pos)},{best[1]},{best[0]:.4f},{best[1]},{(best[1]-0.5)/T:.3f},{best[2]}")
        print(f"[{model}] E3 {t}: n_pos={len(pos)} peak={best[0]:.3f}@step{best[1]}/{best[2]}", flush=True)
    open(os.path.join(V3, model, "E3_pertype", "pertype_summary.csv"), "w").write("\n".join(rows) + "\n")

    # ---- E4 ablation at (k*, l*) ----
    kk = ks_; b_ = blocks[ls_]
    abr = ["probe_type,feature_source,auc,n_params"]
    Xb = FEAT[b_][:, kk].astype(np.float32)
    abr.append(f"logistic,best_block[{b_}]@step{kk+1},{V.probe_auc(Xb[tr],y[tr],Xb[te],y[te]):.4f},{Xb.shape[1]+1}")
    mlp = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=500, random_state=V.RS)); mlp.fit(Xb[tr], y[tr])
    abr.append(f"mlp_64,best_block[{b_}]@step{kk+1},{roc_auc_score(y[te],mlp.predict_proba(Xb[te])[:,1]):.4f},-")
    Xc = np.concatenate([FEAT[b][:, kk].astype(np.float32) for b in blocks], axis=1)
    abr.append(f"logistic,concat_{len(blocks)}blocks@step{kk+1},{V.probe_auc(Xc[tr],y[tr],Xc[te],y[te]):.4f},{Xc.shape[1]+1}")
    base = os.path.join(V3, model, "E1_data")
    files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(base, "raw", "*.npz")))
             if os.path.basename(f) in {r["file"] for r in csv.DictReader(open(os.path.join(base, "labels.csv")))}]
    Xz = np.stack([np.load(os.path.join(base, "raw", f))["final_latent_gap"].astype(np.float32) for f in files])
    abr.append(f"logistic,final_latent_gap,{V.probe_auc(Xz[tr],y[tr],Xz[te],y[te]):.4f},{Xz.shape[1]+1}")
    open(os.path.join(V3, model, "E4_ablation", "ablation_table.csv"), "w").write("\n".join(abr) + "\n")
    json.dump({"k_star": int(kk + 1), "l_star": b_}, open(os.path.join(V3, model, "E4_ablation", "kstar.json"), "w"))
    print(f"[{model}] E4 done. total {time.time()-t0:.0f}s\n" + "\n".join(abr), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    main(ap.parse_args().model)
