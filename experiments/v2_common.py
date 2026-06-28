"""v2 shared loader + probe helper."""
import os, sys, glob, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

V2ROOT = os.path.join(C.ROOT, "v2")
RS = 42


def load_labels(subdir):
    lab = {}
    with open(os.path.join(V2ROOT, subdir, "labels_vlm.csv")) as f:
        for row in csv.DictReader(f):
            lab[row["file"]] = {
                "defective": int(row["defective"]),
                "dtypes": [t for t in row["defect_types"].split("|") if t],
                "severity": int(row["severity"]), "conf": float(row["confidence"])}
    return lab


def load_feats(subdir, conf_min=0.6, blocks=None):
    lab = load_labels(subdir)
    blocks = blocks or C.BLOCK_NAMES
    raw = os.path.join(V2ROOT, subdir, "raw")
    files = [f for f in sorted(glob.glob(os.path.join(raw, "*.npz")))
             if os.path.basename(f) in lab and lab[os.path.basename(f)]["conf"] >= conf_min]
    n = len(files)
    d0 = np.load(files[0])
    dims = {b: int(d0[f"feat_{b}"].shape[1]) for b in blocks}
    FEAT = {b: np.zeros((n, C.STEPS, dims[b]), np.float16) for b in blocks}
    y = np.zeros(n, np.int64); conf = np.zeros(n); ptype = []; dtypes = []; fnames = []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        for b in blocks:
            FEAT[b][i] = d[f"feat_{b}"]
        bn = os.path.basename(f); fnames.append(bn)
        y[i] = lab[bn]["defective"]; conf[i] = lab[bn]["conf"]
        ptype.append(str(d["prompt_type"])); dtypes.append(lab[bn]["dtypes"])
    return {"FEAT": FEAT, "y": y, "conf": conf, "ptype": np.array(ptype),
            "dtypes": dtypes, "files": fnames, "dims": dims, "n": n,
            "total_seen": len(lab)}


def probe_auc(Xtr, ytr, Xte, yte, return_proba=False):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=1000, random_state=RS))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    return (auc, p) if return_proba else auc
