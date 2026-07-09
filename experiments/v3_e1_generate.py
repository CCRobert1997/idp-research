"""IDP v3 E1: generate diverse-content images + DiT activations for a given model.
Reuses the v2 6-type prompt pool (72 prompts) for cross-model comparability.
Resumable. Outputs to v3/{model}/E1_data/{raw,png}/ + manifest.csv.

Usage: v3_e1_generate.py --model M2 --seeds 120 [--seed_start 0] [--types HAND,FACE,...]
"""
import os, sys, time, argparse, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from idp_v2_prompts import PROMPTS_V2, TYPES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "v3")

# model backends: M2=SD3.5, M3=FLUX (added later)
def get_backend(model):
    if model == "M2":
        import v3_sd35 as B
        return B
    if model == "M3":
        import v3_flux as B
        return B
    raise ValueError(model)


def run(model, seeds, seed_start, types):
    B = get_backend(model)
    raw = os.path.join(V3, model, "E1_data", "raw")
    png = os.path.join(V3, model, "E1_data", "png")
    os.makedirs(raw, exist_ok=True); os.makedirs(png, exist_ok=True)
    B.get_pipe()
    todo_types = types or TYPES
    total = sum(len(PROMPTS_V2[t]) for t in todo_types) * seeds
    done = 0; t0 = time.time()
    for ptype in todo_types:
        for pidx, prompt in enumerate(PROMPTS_V2[ptype]):
            for s in range(seed_start, seed_start + seeds):
                fn = os.path.join(raw, f"{ptype}_{pidx:02d}_{s:04d}.npz")
                done += 1
                if os.path.exists(fn):
                    continue
                pil, feats, fl = B.generate(prompt, seed=s, capture=True)
                pil.save(os.path.join(png, f"{ptype}_{pidx:02d}_{s:04d}.png"))
                np.savez_compressed(
                    fn, seed=s, prompt=prompt, prompt_idx=pidx, prompt_type=ptype, model=model,
                    final_latent_gap=fl, **{f"feat_{k}": feats[k] for k in B.BLOCK_NAMES})
                if done % 50 == 0 or done == total:
                    el = time.time() - t0
                    print(f"[{done}/{total}] {el:.0f}s ({el/max(done,1):.2f}s/img) "
                          f"last={ptype}_{pidx:02d}_{s:04d}", flush=True)
    print(f"v3 {model} E1 gen done in {time.time()-t0:.0f}s -> {raw}")


def manifest(model):
    raw = os.path.join(V3, model, "E1_data", "raw")
    files = sorted(glob.glob(os.path.join(raw, "*.npz")))
    rows = ["file,prompt_type,prompt_idx,prompt_text,seed,model"]
    for f in files:
        d = np.load(f, allow_pickle=True)
        rows.append(f'{os.path.basename(f)},{d["prompt_type"]},{int(d["prompt_idx"])},'
                    f'"{d["prompt"]}",{int(d["seed"])},{model}')
    out = os.path.join(V3, model, "E1_data", "manifest.csv")
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"manifest -> {out} ({len(files)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--types", default="")
    ap.add_argument("--manifest_only", type=int, default=0)
    a = ap.parse_args()
    types = [t for t in a.types.split(",") if t] if a.types else None
    if not a.manifest_only:
        run(a.model, a.seeds, a.seed_start, types)
    manifest(a.model)
