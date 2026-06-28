"""v2 E2 (overall AUC map + layer ranking) + E3 (per-type tracks, CORE) + E4 (ablation),
all in one process (single load). Writes each stage's outputs as it completes.
Label = VLM `defective`, confidence>=0.6. Stratified 80/10/10, RS=42."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
import v2_common as V
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

SUB = "E1_data"
E2 = os.path.join(V.V2ROOT, "E2_probes"); E3 = os.path.join(V.V2ROOT, "E3_pertype")
E4 = os.path.join(V.V2ROOT, "E4_ablation")
for d in (E2, E3, E4):
    os.makedirs(d, exist_ok=True)
TYPES6 = ["hand", "face", "limb", "count", "physics", "text"]


def split(y, rs=V.RS):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=rs)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=rs)
    return tr, va, te


def boot_ci(yte, p, rng, nb=1000):
    aucs = []
    for _ in range(nb):
        bi = rng.randint(0, len(yte), len(yte))
        if yte[bi].min() == yte[bi].max():
            continue
        aucs.append(roc_auc_score(yte[bi], p[bi]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    t0 = time.time()
    D = V.load_feats(SUB, conf_min=0.6)
    FEAT, y = D["FEAT"], D["y"]
    n = D["n"]
    print(f"loaded {n} (conf>=0.6) of {D['total_seen']} labeled in {time.time()-t0:.0f}s; "
          f"defect_rate={y.mean():.3f} dropped_lowconf={D['total_seen']-n}", flush=True)

    # ================= E2: overall AUC map =================
    tr, va, te = split(y)
    auc_map = np.zeros((C.STEPS, C.N_BLOCKS))
    pcache = {}
    for li, b in enumerate(C.BLOCK_NAMES):
        for k in range(C.STEPS):
            a, p = V.probe_auc(FEAT[b][tr, k].astype(np.float32), y[tr],
                               FEAT[b][te, k].astype(np.float32), y[te], return_proba=True)
            auc_map[k, li] = a; pcache[(k, li)] = p
        print(f"  E2 block {b} ({li+1}/22) best={auc_map[:,li].max():.3f}", flush=True)
    np.save(os.path.join(E2, "auc_map.npy"), auc_map)
    with open(os.path.join(E2, "auc_map.csv"), "w") as f:
        f.write("step,block_name,auc\n")
        for k in range(C.STEPS):
            for li, b in enumerate(C.BLOCK_NAMES):
                f.write(f"{k+1},{b},{auc_map[k,li]:.5f}\n")
    # layer ranking
    with open(os.path.join(E2, "layer_ranking.csv"), "w") as f:
        f.write("block,peak_auc,peak_step\n")
        order = sorted(range(C.N_BLOCKS), key=lambda li: -auc_map[:, li].max())
        for li in order:
            f.write(f"{C.BLOCK_NAMES[li]},{auc_map[:,li].max():.5f},{int(auc_map[:,li].argmax())+1}\n")
    # best-step CI curve
    yte = y[te]; rng = np.random.RandomState(V.RS)
    ci_rows = []
    with open(os.path.join(E2, "best_step_auc_ci.csv"), "w") as f:
        f.write("step,block,auc_mean,ci_low,ci_high\n")
        for k in range(C.STEPS):
            li = int(np.argmax(auc_map[k]))
            m, lo, hi = boot_ci(yte, pcache[(k, li)], rng)
            ci_rows.append((k + 1, C.BLOCK_NAMES[li], m, lo, hi))
            f.write(f"{k+1},{C.BLOCK_NAMES[li]},{m:.5f},{lo:.5f},{hi:.5f}\n")
    k_star, l_star = np.unravel_index(np.argmax(auc_map), auc_map.shape)
    l_star_name = C.BLOCK_NAMES[l_star]
    first_sig = next((r[0] for r in ci_rows if r[3] > 0.5), None)
    e2sum = {"n": n, "defect_rate": round(float(y.mean()), 4),
             "best": {"step": int(k_star + 1), "block": l_star_name, "auc": round(float(auc_map[k_star, l_star]), 4)},
             "first_step_ci_low_gt_0.5": first_sig,
             "auc_step1": round(float(auc_map[0].max()), 4), "auc_step5": round(float(auc_map[4].max()), 4),
             "auc_step10": round(float(auc_map[9].max()), 4), "auc_step25": round(float(auc_map[24].max()), 4),
             "top5_blocks": [C.BLOCK_NAMES[li] for li in order[:5]]}
    json.dump(e2sum, open(os.path.join(E2, "summary.json"), "w"), indent=2)
    print("E2 done:", json.dumps(e2sum), flush=True)

    # ================= E3: per-type AUC-vs-step (CORE) =================
    rng = np.random.RandomState(V.RS)
    auc_rows = ["defect_type,step,auc,ci_low,ci_high,block_mode,block"]
    sum_rows = ["defect_type,n_pos,n_neg,first_sig_step,peak_auc,peak_step,peak_block"]
    neg_all = np.where(y == 0)[0]
    for t in TYPES6:
        pos = np.array([i for i in range(n) if y[i] == 1 and t in D["dtypes"][i]])
        if len(pos) < 20:
            sum_rows.append(f"{t},{len(pos)},{len(neg_all)},NA,NA,NA,NA")
            print(f"  E3 {t}: only {len(pos)} pos -> skip", flush=True); continue
        idx = np.concatenate([pos, neg_all])
        yt = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_all))]).astype(int)
        tr2, _, te2 = split(yt)
        gtr, gte = idx[tr2], idx[te2]; yte2 = yt[te2]
        # per (step, block) auc; derive lstar curve + perbest curve
        best_overall = (-1, None, None)  # auc, step, block
        lstar_curve = []
        perbest_curve = []
        for k in range(C.STEPS):
            best_k = (-1, None, None)
            for li, b in enumerate(C.BLOCK_NAMES):
                a, p = V.probe_auc(FEAT[b][gtr, k].astype(np.float32), yt[tr2],
                                   FEAT[b][gte, k].astype(np.float32), yte2, return_proba=True)
                if li == l_star:
                    m, lo, hi = boot_ci(yte2, p, rng, nb=400)
                    lstar_curve.append((k + 1, a, lo, hi, b))
                if a > best_k[0]:
                    best_k = (a, b, p)
            mb, lob, hib = boot_ci(yte2, best_k[2], rng, nb=400)
            perbest_curve.append((k + 1, best_k[0], lob, hib, best_k[1]))
            if best_k[0] > best_overall[0]:
                best_overall = (best_k[0], k + 1, best_k[1])
        for (k, a, lo, hi, b) in lstar_curve:
            auc_rows.append(f"{t},{k},{a:.5f},{lo:.5f},{hi:.5f},lstar,{b}")
        for (k, a, lo, hi, b) in perbest_curve:
            auc_rows.append(f"{t},{k},{a:.5f},{lo:.5f},{hi:.5f},perbest,{b}")
        fsig = next((k for (k, a, lo, hi, b) in perbest_curve if lo > 0.5), None)
        sum_rows.append(f"{t},{len(pos)},{len(neg_all)},{fsig},{best_overall[0]:.4f},{best_overall[1]},{best_overall[2]}")
        print(f"  E3 {t}: n_pos={len(pos)} peak={best_overall[0]:.3f}@k{best_overall[1]}/{best_overall[2]} first_sig={fsig}", flush=True)
    open(os.path.join(E3, "pertype_auc.csv"), "w").write("\n".join(auc_rows) + "\n")
    open(os.path.join(E3, "pertype_summary.csv"), "w").write("\n".join(sum_rows) + "\n")
    print("E3 done", flush=True)

    # ================= E4: ablation at (k*, l*) =================
    kk = k_star
    rows = ["probe_type,feature_source,auc,n_params"]
    Xb = FEAT[l_star_name][:, kk].astype(np.float32)
    rows.append(f"logistic,best_block[{l_star_name}]@k{kk+1},{V.probe_auc(Xb[tr],y[tr],Xb[te],y[te]):.4f},{Xb.shape[1]+1}")
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=V.RS))
    mlp.fit(Xb[tr], y[tr])
    rows.append(f"mlp_64,best_block[{l_star_name}]@k{kk+1},{roc_auc_score(y[te], mlp.predict_proba(Xb[te])[:,1]):.4f},{(Xb.shape[1]+1)*64+65}")
    Xcat = np.concatenate([FEAT[b][:, kk].astype(np.float32) for b in C.BLOCK_NAMES], axis=1)
    rows.append(f"logistic,concat_22blocks@k{kk+1},{V.probe_auc(Xcat[tr],y[tr],Xcat[te],y[te]):.4f},{Xcat.shape[1]+1}")
    # final latent z0 GAP 4D
    import glob as _g
    raw = os.path.join(V.V2ROOT, SUB, "raw")
    Xz = np.stack([np.load(os.path.join(raw, fn))["final_latent"].astype(np.float32).mean(axis=(1, 2)) for fn in D["files"]])
    rows.append(f"logistic,final_latent_z0_gap4d,{V.probe_auc(Xz[tr],y[tr],Xz[te],y[te]):.4f},5")
    # pixel CLIP
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        cm = CLIPModel.from_pretrained(os.path.join(C.ROOT, "models_clip")).to("cuda").eval()
        cp = CLIPProcessor.from_pretrained(os.path.join(C.ROOT, "models_clip"))
        png = os.path.join(V.V2ROOT, SUB, "png")
        Xc = np.zeros((n, cm.config.projection_dim), np.float32)
        with torch.no_grad():
            for i, fn in enumerate(D["files"]):
                im = Image.open(os.path.join(png, fn.replace(".npz", ".png"))).convert("RGB")
                Xc[i] = cm.get_image_features(**cp(images=im, return_tensors="pt").to("cuda"))[0].float().cpu().numpy()
        rows.append(f"logistic,pixel_clip_img,{V.probe_auc(Xc[tr],y[tr],Xc[te],y[te]):.4f},{Xc.shape[1]+1}")
    except Exception as e:
        rows.append(f"logistic,pixel_clip_img,NA,NA"); print("clip fail", repr(e))
    open(os.path.join(E4, "ablation_table.csv"), "w").write("\n".join(rows) + "\n")
    print("E4 done:\n" + "\n".join(rows), flush=True)
    json.dump({"k_star": int(kk + 1), "l_star": l_star_name}, open(os.path.join(E4, "kstar.json"), "w"))
    print(f"v2_analyze total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
