"""E3: ablation at the best step k* (from E2). Scheme I labels, same RS=42 split.
  3.1 linear vs MLP probe on best block l*
  3.2 feature source: best-block / concat-all-22 / final-latent z0 (GAP 4D) / pixel(CLIP-img)
Outputs: E3_ablation/ablation_table.csv
"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import idp_common as C
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATADIR = os.path.join(C.ROOT, "E1_data", "raw")
PNGDIR = os.path.join(C.ROOT, "E1_data", "png")
OUT = os.path.join(C.ROOT, "E3_ablation")
os.makedirs(OUT, exist_ok=True)
RS = 42


def get_split(y):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=RS)
    val, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=RS)
    return tr, val, te


def lr_auc(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=RS))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def mlp_auc(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=RS))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def clip_features(files):
    """CLIP image embeddings (local openai/clip-vit-base-patch32) as pixel-space
    features. Substitutes for InceptionV3-pool (no local weights); documented."""
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    model = CLIPModel.from_pretrained(os.path.join(C.ROOT, "models_clip")).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(os.path.join(C.ROOT, "models_clip"))
    feats = np.zeros((len(files), model.config.projection_dim), np.float32)
    with torch.no_grad():
        for i, f in enumerate(files):
            base = os.path.basename(f).replace(".npz", ".png")
            img = Image.open(os.path.join(PNGDIR, base)).convert("RGB")
            inp = proc(images=img, return_tensors="pt").to("cuda")
            feats[i] = model.get_image_features(**inp)[0].float().cpu().numpy()
    return feats


def main():
    k_star = int(json.load(open(os.path.join(C.ROOT, "E2_probes", "summary.json")))["best"]["step"]) - 1
    l_star = json.load(open(os.path.join(C.ROOT, "E2_probes", "summary.json")))["best"]["block"]
    print(f"k*={k_star+1} l*={l_star}")
    files = sorted(glob.glob(os.path.join(DATADIR, "*.npz")))
    n = len(files)
    y = np.array([int(np.load(f)["label_incl"]) for f in files])
    tr, val, te = get_split(y)

    rows = ["probe_type,feature_source,auc,n_params"]

    # feature source 1: best block l* at k*
    Xbest = np.stack([np.load(f)[f"feat_{l_star}"][k_star] for f in files]).astype(np.float32)
    a = lr_auc(Xbest[tr], y[tr], Xbest[te], y[te])
    rows.append(f"logistic,best_block[{l_star}]@k{k_star+1},{a:.4f},{Xbest.shape[1]+1}")
    # 3.1 MLP on same
    am = mlp_auc(Xbest[tr], y[tr], Xbest[te], y[te])
    rows.append(f"mlp_64,best_block[{l_star}]@k{k_star+1},{am:.4f},{(Xbest.shape[1]+1)*64+65}")

    # feature source 2: concat all 22 blocks at k*
    Xcat = np.concatenate([np.stack([np.load(f)[f"feat_{b}"][k_star] for f in files])
                           for b in C.BLOCK_NAMES], axis=1).astype(np.float32)
    a = lr_auc(Xcat[tr], y[tr], Xcat[te], y[te])
    rows.append(f"logistic,concat_22blocks@k{k_star+1},{a:.4f},{Xcat.shape[1]+1}")

    # feature source 3: final latent z0 GAP -> 4D
    Xz = np.stack([np.load(f)["final_latent"].astype(np.float32).mean(axis=(1, 2)) for f in files])
    a = lr_auc(Xz[tr], y[tr], Xz[te], y[te])
    rows.append(f"logistic,final_latent_z0_gap4d,{a:.4f},{Xz.shape[1]+1}")

    # feature source 4: pixel-space CLIP image features
    try:
        Xc = clip_features(files)
        a = lr_auc(Xc[tr], y[tr], Xc[te], y[te])
        rows.append(f"logistic,pixel_clip_img512,{a:.4f},{Xc.shape[1]+1}")
    except Exception as e:
        rows.append(f"logistic,pixel_clip_img512,NA({type(e).__name__}),NA")
        print("clip feature step failed:", repr(e))

    with open(os.path.join(OUT, "ablation_table.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))
    print("-> E3_ablation/ablation_table.csv")


if __name__ == "__main__":
    main()
