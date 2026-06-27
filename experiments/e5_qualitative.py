"""E5: qualitative examples from test500. Pick 2 true-positive (correctly aborted
defective), 1 true-negative (correctly accepted clean), 1 false-positive (wrongly
aborted clean). Re-generate each to decode the latent at k* (noisy) next to the
final image. Outputs E4_efficiency/qualitative/.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C

OUT = os.path.join(C.ROOT, "E4_efficiency", "qualitative")
os.makedirs(OUT, exist_ok=True)
TESTDIR = os.path.join(C.ROOT, "E4_efficiency", "test500", "raw")


def main():
    summ = json.load(open(os.path.join(C.ROOT, "E2_probes", "summary.json")))
    k_star = int(summ["best"]["step"]) - 1
    e4 = json.load(open(os.path.join(C.ROOT, "E4_efficiency", "summary.json")))
    tau = e4["tau_target"]
    sc = np.load(os.path.join(C.ROOT, "E4_efficiency", "test500_scores.npz"), allow_pickle=True)
    scores, labels, names = sc["scores"], sc["labels"], sc["files"]

    abort = scores > tau
    tp = np.where(abort & (labels == 1))[0]
    tn = np.where((~abort) & (labels == 0))[0]
    fp = np.where(abort & (labels == 0))[0]
    # rank by confidence for nicer examples
    tp = tp[np.argsort(-scores[tp])][:2]
    tn = tn[np.argsort(scores[tn])][:1]
    fp = fp[np.argsort(-scores[fp])][:1]
    picks = [("TP", i) for i in tp] + [("TN", i) for i in tn] + [("FP", i) for i in fp]

    info = [f"tau={tau} k*={k_star+1}"]
    for tag, i in picks:
        fn = str(names[i]); pi, seed = fn.replace(".npz", "").split("_")
        pi, seed = int(pi), int(seed)
        d = np.load(os.path.join(TESTDIR, fn))
        prompt = str(d["prompt"])
        pil, _, _, mid = C.generate(prompt, seed=seed, capture=False, return_step_latent=k_star)
        base = f"{tag}_{pi:02d}_{seed:04d}"
        pil.save(os.path.join(OUT, base + "_final.png"))
        if mid is not None:
            mid.save(os.path.join(OUT, base + "_step{}.png".format(k_star + 1)))
        a = C.analyze_hands(pil)
        line = (f"{tag} {base}: prompt='{prompt}' probe_score={float(scores[i]):.3f} "
                f"label={int(labels[i])} fingers={a['finger_count']} hands={a['num_hands']} "
                f"best_geom={a['best_geom']}")
        print(line); info.append(line)
    with open(os.path.join(OUT, "qual_info.txt"), "w") as f:
        f.write("\n".join(info) + "\n")
    print("-> ", OUT)


if __name__ == "__main__":
    main()
