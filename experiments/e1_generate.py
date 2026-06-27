"""E1: generate N seeds x 20 prompts, capture per-step 22-block GAP activations,
run HandLandmarker, store ALL raw signals per image so the label is re-derivable.

Resumable: skips any (prompt_idx, seed) whose npz already exists.
Usage: e1_generate.py [--seeds 100] [--seed_start 0] [--outdir E1_data] [--save_png 1]
"""
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C


def run(seeds, seed_start, outdir, save_png):
    raw = os.path.join(C.ROOT, outdir, "raw")
    png = os.path.join(C.ROOT, outdir, "png")
    os.makedirs(raw, exist_ok=True)
    if save_png:
        os.makedirs(png, exist_ok=True)
    C.get_pipe()
    total = len(C.PROMPTS) * seeds
    done = 0
    t0 = time.time()
    for pi, prompt in enumerate(C.PROMPTS):
        for s in range(seed_start, seed_start + seeds):
            fn = os.path.join(raw, f"{pi:02d}_{s:04d}.npz")
            done += 1
            if os.path.exists(fn):
                continue
            pil, latent, feats = C.generate(prompt, seed=s, capture=True)
            a = C.analyze_hands(pil)
            label = C.compute_label(a)              # scheme S (spec, degenerate)
            label_incl = C.compute_label_inclusive(a)  # scheme I (primary)
            if save_png:
                pil.save(os.path.join(png, f"{pi:02d}_{s:04d}.png"))
            # store: feats dict (22 keys, each (50,C) fp16), final latent, raw signals
            np.savez_compressed(
                fn,
                seed=s, prompt=prompt, prompt_idx=pi,
                label=label, label_incl=label_incl, finger_count=a["finger_count"],
                hand_detected=bool(a["num_hands"] > 0),
                num_hands=a["num_hands"], det_score=a["det_score"],
                best_geom=a["best_geom"],
                per_hand_scores=np.array(a["per_hand_scores"], dtype=np.float32),
                per_hand_geom=np.array(a["per_hand_geom"], dtype=np.float32),
                per_hand_fingers=np.array(a["per_hand_fingers"], dtype=np.int32),
                landmarks=np.array(a["landmarks"], dtype=np.float32) if a["landmarks"] else np.zeros((0, 21, 3), np.float32),
                final_latent=latent,
                **{f"feat_{k}": feats[k] for k in C.BLOCK_NAMES},
            )
            if done % 25 == 0 or done == total:
                el = time.time() - t0
                print(f"[{done}/{total}] {el:.0f}s ({el/max(done,1):.2f}s/img) "
                      f"last={pi:02d}_{s:04d} hands={a['num_hands']} label={label}", flush=True)
    print(f"E1 gen done in {time.time()-t0:.0f}s -> {raw}")


def build_csv(outdir):
    raw = os.path.join(C.ROOT, outdir, "raw")
    rows = ["file,prompt_idx,seed,hand_detected,label,label_incl,finger_count,det_score,best_geom,num_hands"]
    files = sorted(glob.glob(os.path.join(raw, "*.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        li = int(d['label_incl']) if 'label_incl' in d else -9
        rows.append(f"{os.path.basename(f)},{int(d['prompt_idx'])},{int(d['seed'])},"
                    f"{bool(d['hand_detected'])},{int(d['label'])},{li},{int(d['finger_count'])},"
                    f"{float(d['det_score']):.4f},{float(d['best_geom']):.4f},{int(d['num_hands'])}")
    csv = os.path.join(C.ROOT, outdir, "labels.csv")
    with open(csv, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    # quick distribution
    import collections
    labs = collections.Counter(int(np.load(f)['label']) for f in files)
    labs_i = collections.Counter(int(np.load(f)['label_incl']) for f in files)
    print(f"labels.csv -> {csv}  ({len(files)} files)  S dist: {dict(labs)}  I dist: {dict(labs_i)}")
    return csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--outdir", default="E1_data")
    ap.add_argument("--save_png", type=int, default=1)
    ap.add_argument("--csv_only", type=int, default=0)
    args = ap.parse_args()
    if not args.csv_only:
        run(args.seeds, args.seed_start, args.outdir, args.save_png)
    build_csv(args.outdir)
