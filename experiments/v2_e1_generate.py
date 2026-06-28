"""IDP v2 E1: diverse-content generation + 22-block activation capture.
Reuses v1 idp_common.generate (unchanged). NO labeling here — labels come from the
VLM judge in E1.5. Resumable. Outputs to IDP_research/v2/E1_data/.

Usage: v2_e1_generate.py [--seeds 70] [--seed_start 0] [--subdir E1_data] [--save_png 1]
"""
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C
from idp_v2_prompts import PROMPTS_V2, TYPES

V2ROOT = os.path.join(C.ROOT, "v2")


def run(seeds, seed_start, subdir, save_png):
    raw = os.path.join(V2ROOT, subdir, "raw")
    png = os.path.join(V2ROOT, subdir, "png")
    os.makedirs(raw, exist_ok=True)
    if save_png:
        os.makedirs(png, exist_ok=True)
    C.get_pipe()
    total = sum(len(v) for v in PROMPTS_V2.values()) * seeds
    done = 0; t0 = time.time()
    for ptype in TYPES:
        for pidx, prompt in enumerate(PROMPTS_V2[ptype]):
            for s in range(seed_start, seed_start + seeds):
                fn = os.path.join(raw, f"{ptype}_{pidx:02d}_{s:04d}.npz")
                done += 1
                if os.path.exists(fn):
                    continue
                pil, latent, feats = C.generate(prompt, seed=s, capture=True)
                if save_png:
                    pil.save(os.path.join(png, f"{ptype}_{pidx:02d}_{s:04d}.png"))
                np.savez_compressed(
                    fn, seed=s, prompt=prompt, prompt_idx=pidx, prompt_type=ptype,
                    final_latent=latent,
                    **{f"feat_{k}": feats[k] for k in C.BLOCK_NAMES},
                )
                if done % 50 == 0 or done == total:
                    el = time.time() - t0
                    print(f"[{done}/{total}] {el:.0f}s ({el/max(done,1):.2f}s/img) "
                          f"last={ptype}_{pidx:02d}_{s:04d}", flush=True)
    print(f"v2 E1 gen done in {time.time()-t0:.0f}s -> {raw}")


def manifest(subdir):
    raw = os.path.join(V2ROOT, subdir, "raw")
    files = sorted(glob.glob(os.path.join(raw, "*.npz")))
    rows = ["file,prompt_type,prompt_idx,prompt_text,seed"]
    for f in files:
        d = np.load(f, allow_pickle=True)
        rows.append(f'{os.path.basename(f)},{str(d["prompt_type"])},{int(d["prompt_idx"])},'
                    f'"{str(d["prompt"])}",{int(d["seed"])}')
    out = os.path.join(V2ROOT, subdir, "manifest.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"manifest -> {out} ({len(files)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=70)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--subdir", default="E1_data")
    ap.add_argument("--save_png", type=int, default=1)
    ap.add_argument("--manifest_only", type=int, default=0)
    a = ap.parse_args()
    if not a.manifest_only:
        run(a.seeds, a.seed_start, a.subdir, a.save_png)
    manifest(a.subdir)
