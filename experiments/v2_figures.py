"""v2 figures: F2 overall AUC heatmap+CI curve; F3 (CORE) per-type AUC-vs-step; F4 Pareto."""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import idp_common as C
import v2_common as V

FIG = os.path.join(V.V2ROOT, "figures"); os.makedirs(FIG, exist_ok=True)


def f2():
    auc = np.load(os.path.join(V.V2ROOT, "E2_probes", "auc_map.npy"))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(auc.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.5, vmax=auc.max())
    ax[0].set_yticks(range(C.N_BLOCKS)); ax[0].set_yticklabels(C.BLOCK_NAMES, fontsize=7)
    ax[0].set_xlabel("denoising step k"); ax[0].set_title("Overall probe AUC (block x step)")
    fig.colorbar(im, ax=ax[0], label="AUC")
    rows = list(csv.DictReader(open(os.path.join(V.V2ROOT, "E2_probes", "best_step_auc_ci.csv"))))
    k = [int(r["step"]) for r in rows]; m = [float(r["auc_mean"]) for r in rows]
    lo = [float(r["ci_low"]) for r in rows]; hi = [float(r["ci_high"]) for r in rows]
    ax[1].plot(k, m, "-o", ms=3); ax[1].fill_between(k, lo, hi, alpha=0.25)
    ax[1].axhline(0.5, ls="--", c="gray"); ax[1].set_ylim(0.45, 1.0)
    ax[1].set_xlabel("denoising step k"); ax[1].set_ylabel("AUC*"); ax[1].set_title("Best-block AUC vs step")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_overall_auc.png"), dpi=130); plt.close()
    print("-> fig2")


def f3():
    rows = list(csv.DictReader(open(os.path.join(V.V2ROOT, "E3_pertype", "pertype_auc.csv"))))
    fig, ax = plt.subplots(figsize=(9, 6))
    types = ["hand", "face", "limb", "count", "physics", "text"]
    colors = dict(zip(types, ["C0", "C1", "C2", "C3", "C4", "C5"]))
    for t in types:
        rr = sorted([r for r in rows if r["defect_type"] == t and r["block_mode"] == "perbest"],
                    key=lambda r: int(r["step"]))
        if not rr:
            continue
        k = [int(r["step"]) for r in rr]; a = [float(r["auc"]) for r in rr]
        lo = [float(r["ci_low"]) for r in rr]; hi = [float(r["ci_high"]) for r in rr]
        ax.plot(k, a, "-", color=colors[t], label=t, lw=1.8)
        ax.fill_between(k, lo, hi, color=colors[t], alpha=0.12)
    ax.axhline(0.5, ls="--", c="gray"); ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("denoising step k"); ax.set_ylabel("per-type AUC (per-best block)")
    ax.set_title("When is each defect type decodable? (per-type AUC vs step)")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_pertype_auc.png"), dpi=130); plt.close()
    print("-> fig3 (CORE)")


def f4():
    p = os.path.join(V.V2ROOT, "E5_efficiency", "efficiency_table.csv")
    if not os.path.exists(p):
        return
    rows = list(csv.DictReader(open(p)))
    def pts(m):
        r = [x for x in rows if x["method"] == m and x["defect_rate"] not in ("", "NA")]
        return sorted((float(x["unet_calls_per_accepted"]), float(x["defect_rate"])) for x in r)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    idp = pts("IDP")
    if idp:
        x, y = zip(*idp); ax.plot(x, y, "-o", label="IDP")
    for m, c, mk in [("NoFilter", "C1", "s"), ("FullRejectSampling", "C2", "^"), ("Oracle", "C3", "*"), ("RandomAbort", "C4", "x")]:
        pp = pts(m)
        if pp:
            x, y = zip(*pp); ax.scatter(x, y, c=c, marker=mk, s=70, label=m)
    ax.set_xlabel("UNet calls per accepted"); ax.set_ylabel("defect rate among accepted")
    ax.set_title("Early-stop efficiency Pareto (held-out)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_pareto.png"), dpi=130); plt.close()
    print("-> fig4")


if __name__ == "__main__":
    f2(); f3(); f4()
