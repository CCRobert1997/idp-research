"""v2 E6: qualitative. Per defect type, 1-2 representative defective examples; plus
overall TP/TN/FP. Re-generate to decode the k* intermediate latent next to the final.
Outputs E6_qual/."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
import v2_common as V

OUT = os.path.join(V.V2ROOT, "E6_qual"); os.makedirs(OUT, exist_ok=True)
HELDRAW = os.path.join(V.V2ROOT, "E5_heldout", "raw")


def main():
    ks = json.load(open(os.path.join(V.V2ROOT, "E4_ablation", "kstar.json")))
    k_star = int(ks["k_star"]) - 1
    e5 = json.load(open(os.path.join(V.V2ROOT, "E5_efficiency", "summary.json")))
    tau = e5["tau_target"]
    sc = np.load(os.path.join(V.V2ROOT, "E5_efficiency", "scores.npz"), allow_pickle=True)
    scores, labels, files = sc["scores"], sc["labels"], [str(x) for x in sc["files"]]
    lab = V.load_labels("E5_heldout")
    info = [f"tau={tau} k*={k_star+1}"]
    picks = []
    # overall TP/TN/FP
    ab = scores > tau
    for tag, mask in [("TP", ab & (labels == 1)), ("TN", (~ab) & (labels == 0)), ("FP", ab & (labels == 0))]:
        w = np.where(mask)[0]
        if len(w):
            w = w[np.argsort(-scores[w])] if tag != "TN" else w[np.argsort(scores[w])]
            picks.append((tag, int(w[0])))
    # per-type defective example (correctly aborted)
    for t in ["hand", "face", "limb", "count", "physics", "text"]:
        cand = [i for i in range(len(files)) if labels[i] == 1 and t in lab[files[i]]["dtypes"] and scores[i] > tau]
        if cand:
            cand.sort(key=lambda i: -scores[i]); picks.append((f"type_{t}", cand[0]))

    for tag, i in picks:
        fn = files[i]; parts = fn.replace(".npz", "").split("_")
        ptype, pidx, seed = parts[0], int(parts[1]), int(parts[2])
        d = np.load(os.path.join(HELDRAW, fn), allow_pickle=True)
        prompt = str(d["prompt"])
        pil, _, _, mid = C.generate(prompt, seed=seed, capture=False, return_step_latent=k_star)
        base = f"{tag}_{fn.replace('.npz','')}"
        pil.save(os.path.join(OUT, base + "_final.png"))
        if mid is not None:
            mid.save(os.path.join(OUT, base + f"_step{k_star+1}.png"))
        L = lab[fn]
        line = (f"{tag} {base}: prompt='{prompt}' score={float(scores[i]):.3f} "
                f"vlm_defective={int(labels[i])} dtypes={L['dtypes']} sev={L['severity']}")
        print(line); info.append(line)
    open(os.path.join(OUT, "qual_info.txt"), "w").write("\n".join(info) + "\n")
    print("E6 ->", OUT)


if __name__ == "__main__":
    main()
