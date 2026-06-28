"""IDP v2 E1.5: VLM visual-defect judge using on-server Qwen2.5-VL-7B-Instruct.
Fixed rubric (spec). Resumable, batched. Also a --reliability mode = second pass on
a random 500 to report % agreement + Cohen's kappa.

Outputs (under v2/<subdir>/): labels_vlm.csv, reliability.json (full run only).
Usage: v2_e15_label.py [--subdir E1_data] [--batch 8] [--reliability 1]
"""
import os, sys, json, glob, time, argparse, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import idp_common as C

V2ROOT = os.path.join(C.ROOT, "v2")
MODEL_DIR = None  # resolved from hf_cache snapshot
TYPES6 = ["hand", "face", "limb", "count", "physics", "text"]

# NOTE (2026-06-28): the spec's original "be conservative, only if a non-expert would
# clearly notice" rubric made Qwen2.5-VL-7B flag ~0% defects on a calibration set
# (it called a blatantly garbled OPEN sign and a melted crowd "clean"). Replaced with
# this checklist + reason-then-JSON rubric, which restores discrimination
# (garbled text/melted face/fused objects/noodle-arm all caught, clean stays clean).
# Quality fix, not results-biasing; validated by E1.5 reliability (kappa) + human spot-check.
RUBRIC = """You are an expert inspector of AI-generated images. Look carefully for STRUCTURAL failures typical of AI generation:
- hands/fingers: wrong count, fused, extra, bent unnaturally
- faces: asymmetry, extra/missing/melted features, distorted eyes
- limbs/body: extra or missing arms/legs, impossible joints
- object count: count the main objects vs what the scene implies
- physics/objects: fused/merged objects, floating, impossible geometry
- text: do the letters spell REAL, correctly-spelled, readable words? Garbled or nonsense letters = a text defect.
Briefly describe what you see (1-2 sentences), THEN on the final line output ONLY this JSON:
{"defective": true/false, "defect_types": [subset of hand,face,limb,count,physics,text], "severity": 0-3, "confidence": 0.0-1.0, "note": "short"}
Mark defective=true if at least one clear structural defect is present that a careful viewer would notice. Ignore pure style, lighting, or blur."""


def resolve_model():
    global MODEL_DIR
    hits = glob.glob("/home/ubuntu/shangyu_comfyui/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*")
    hits = [h for h in hits if os.path.exists(os.path.join(h, "config.json"))]
    assert hits, "Qwen2.5-VL snapshot not found (download incomplete)"
    MODEL_DIR = hits[0]
    return MODEL_DIR


_M = {}


def get_model():
    if not _M:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        md = resolve_model()
        _M["proc"] = AutoProcessor.from_pretrained(md, use_fast=True)
        _M["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            md, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    return _M["model"], _M["proc"]


def parse_json(resp):
    # robust to a reasoning preamble: take the LAST balanced (non-nested) JSON object
    cands = re.findall(r"\{[^{}]*\}", resp, re.DOTALL)
    d = None
    for c in reversed(cands):
        try:
            j = json.loads(c)
            if "defective" in j:
                d = j; break
        except Exception:
            continue
    if d is None:
        return None
    dt = [t for t in (d.get("defect_types") or []) if t in TYPES6]
    try:
        return {"defective": bool(d.get("defective")),
                "defect_types": dt,
                "severity": int(d.get("severity", 0)),
                "confidence": float(d.get("confidence", 0.0)),
                "note": str(d.get("note", ""))[:80]}
    except Exception:
        return None


@torch.no_grad()
def judge_batch(pngs, do_sample=False, temp=0.0):
    from PIL import Image
    model, proc = get_model()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": RUBRIC}]}]
    texts = [proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for _ in pngs]
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
    gen = model.generate(**inp, max_new_tokens=220, do_sample=do_sample,
                         temperature=(temp if do_sample else None),
                         top_p=(0.9 if do_sample else None))
    trimmed = gen[:, inp.input_ids.shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)


def png_path(subdir, fn_npz):
    return os.path.join(V2ROOT, subdir, "png", fn_npz.replace(".npz", ".png"))


def run_full(subdir, batch):
    raw = os.path.join(V2ROOT, subdir, "raw")
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(raw, "*.npz")))
    csv = os.path.join(V2ROOT, subdir, "labels_vlm.csv")
    done = set()
    if os.path.exists(csv):
        with open(csv) as f:
            next(f, None)
            done = {ln.split(",")[0] for ln in f if ln.strip()}
    todo = [f for f in files if f not in done]
    print(f"labeling {len(todo)} / {len(files)} ({len(done)} cached)", flush=True)
    new = not os.path.exists(csv)
    fh = open(csv, "a")
    if new:
        fh.write("file,defective,defect_types,severity,confidence,note\n")
    t0 = time.time(); n = 0; nbad = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        pngs = [png_path(subdir, f) for f in chunk]
        try:
            resps = judge_batch(pngs)
        except Exception as e:
            print("batch err, falling back to singles:", repr(e)); resps = []
            for p in pngs:
                try: resps.append(judge_batch([p])[0])
                except Exception: resps.append("")
        for f, r in zip(chunk, resps):
            d = parse_json(r)
            if d is None:
                nbad += 1
                d = {"defective": False, "defect_types": [], "severity": 0, "confidence": 0.0, "note": "PARSE_FAIL"}
            dts = "|".join(d["defect_types"])
            note = d["note"].replace(",", ";").replace('"', "'")
            fh.write(f'{f},{int(d["defective"])},{dts},{d["severity"]},{d["confidence"]:.3f},"{note}"\n')
            n += 1
        fh.flush()
        if (i // batch) % 10 == 0:
            el = time.time() - t0
            print(f"[{n}/{len(todo)}] {el:.0f}s ({el/max(n,1):.2f}s/img) parse_fail={nbad}", flush=True)
    fh.close()
    print(f"E1.5 full done: {n} new, {nbad} parse-fail -> {csv}")


def run_reliability(subdir, n_sample=500):
    """Second pass (do_sample temp 0.8) on a random subsample; report agreement + kappa
    vs the deterministic primary labels."""
    csv = os.path.join(V2ROOT, subdir, "labels_vlm.csv")
    prim = {}
    with open(csv) as f:
        next(f)
        for ln in f:
            p = ln.split(","); prim[p[0]] = int(p[1])
    rng = random.Random(42)
    sample = rng.sample(list(prim), min(n_sample, len(prim)))
    a, b = [], []
    for i in range(0, len(sample), 8):
        chunk = sample[i:i + 8]
        resps = judge_batch([png_path(subdir, f) for f in chunk], do_sample=True, temp=0.8)
        for f, r in zip(chunk, resps):
            d = parse_json(r)
            if d is None:
                continue
            a.append(prim[f]); b.append(int(d["defective"]))
    a, b = np.array(a), np.array(b)
    agree = float((a == b).mean())
    # Cohen's kappa
    po = agree
    pe = sum(((a == k).mean() * (b == k).mean()) for k in (0, 1))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    rel = {"n": int(len(a)), "pct_agreement": round(agree, 4), "cohen_kappa": round(float(kappa), 4),
           "primary_defect_rate": round(float(a.mean()), 4), "second_defect_rate": round(float(b.mean()), 4),
           "method": "primary greedy vs second-pass temp=0.8"}
    out = os.path.join(V2ROOT, subdir, "reliability.json")
    json.dump(rel, open(out, "w"), indent=2)
    print("reliability ->", json.dumps(rel))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="E1_data")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reliability", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        raw = sorted(glob.glob(os.path.join(V2ROOT, a.subdir, "raw", "*.npz")))[:a.smoke]
        for f in raw:
            r = judge_batch([png_path(a.subdir, os.path.basename(f))])[0]
            print(os.path.basename(f), "->", parse_json(r), "|raw:", r[:120].replace("\n", " "))
    else:
        run_full(a.subdir, a.batch)
        if a.reliability:
            run_reliability(a.subdir)
