"""Figures: F2 AUC heatmap + best-step CI curve; F3 efficiency Pareto."""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import idp_common as C

OUT = os.path.join(C.ROOT, "figures")
os.makedirs(OUT, exist_ok=True)


def fig2():
    auc = np.load(os.path.join(C.ROOT, "E2_probes", "auc_map.npy"))  # (50,22)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(auc.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.5, vmax=auc.max())
    ax[0].set_yticks(range(C.N_BLOCKS)); ax[0].set_yticklabels(C.BLOCK_NAMES, fontsize=7)
    ax[0].set_xlabel("denoising step k (1..50)"); ax[0].set_ylabel("UNet block")
    ax[0].set_title("Probe AUC heatmap (block x step)")
    fig.colorbar(im, ax=ax[0], label="test AUC")
    ks, mean, lo, hi = [], [], [], []
    with open(os.path.join(C.ROOT, "E2_probes", "best_step_auc_ci.csv")) as f:
        for r in csv.DictReader(f):
            ks.append(int(r["step"])); mean.append(float(r["auc_mean"]))
            lo.append(float(r["ci_low"])); hi.append(float(r["ci_high"]))
    ks, mean, lo, hi = map(np.array, (ks, mean, lo, hi))
    ax[1].plot(ks, mean, "-o", ms=3, color="C0", label="best-block AUC*(k)")
    ax[1].fill_between(ks, lo, hi, alpha=0.25, color="C0", label="95% bootstrap CI")
    ax[1].axhline(0.5, ls="--", c="gray", lw=1)
    kbest = int(np.argmax(mean)) + 1
    ax[1].axvline(kbest, ls=":", c="C3", label=f"k*={kbest} (AUC*={mean.max():.3f})")
    ax[1].set_xlabel("denoising step k"); ax[1].set_ylabel("AUC*"); ax[1].set_ylim(0.45, 1.0)
    ax[1].set_title("Best-block AUC vs step"); ax[1].legend(fontsize=8)
    plt.tight_layout(); p = os.path.join(OUT, "fig2_auc_map.png"); plt.savefig(p, dpi=130); plt.close()
    print("->", p)


def fig3():
    rows = list(csv.DictReader(open(os.path.join(C.ROOT, "E4_efficiency", "efficiency_table.csv"))))
    def pts(m):
        r = [x for x in rows if x["method"] == m and x["defect_rate"] not in ("", "NA")]
        return [(float(x["unet_calls_per_accepted"]), float(x["defect_rate"])) for x in r]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    idp = sorted(pts("IDP"))
    if idp:
        x, y = zip(*idp); ax.plot(x, y, "-o", color="C0", label="IDP (threshold sweep)")
    for m, c, mk in [("NoFilter", "C1", "s"), ("FullRejectSampling", "C2", "^"),
                     ("Oracle", "C3", "*"), ("RandomAbort", "C4", "x")]:
        p = pts(m)
        if p:
            x, y = zip(*p); ax.scatter(x, y, c=c, marker=mk, s=70, label=m)
    ax.set_xlabel("UNet calls per accepted image (lower=cheaper)")
    ax.set_ylabel("defect rate among accepted (lower=better)")
    ax.set_title("Early-stop efficiency Pareto (held-out test500)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); p = os.path.join(OUT, "fig3_pareto.png"); plt.savefig(p, dpi=130); plt.close()
    print("->", p)


if __name__ == "__main__":
    fig2(); fig3()
