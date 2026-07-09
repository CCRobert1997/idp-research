"""IDP v3 E1.5: VLM judge = Qwen2.5-VL-32B-Instruct (4-bit, on-server).
Judge SEES the generation prompt (fixes v2's COUNT failure) + refined rubric (spec §3.2).
Resumable, batched. --reliability runs a 2nd pass for Cohen's kappa.

Outputs v3/{model}/labels.csv (+ reliability.json). DiT-agnostic (operates on PNGs).
Usage: v3_label.py --model M2 [--batch 4] [--reliability 1]
"""
import os, sys, json, glob, time, argparse, re, random, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "v3")
TYPES6 = ["hand", "face", "limb", "count", "physics", "text"]

RUBRIC = """You are a strict structural-defect judge for AI-generated images.
The image was generated from this prompt: "{PROMPT}".
Answer JSON:
{{
 "defective": true/false,
 "defect_types": [],
 "severity": 0,
 "confidence": 0.0,
 "note": ""
}}
"defective" = ANY clear STRUCTURAL defect: malformed/abnormal-count anatomy (hands/fingers/limbs/eyes), fused or merged objects, physically impossible geometry, garbled/unreadable text, OR object count not matching the prompt's stated number. IGNORE pure style, aesthetics, blur, lighting, cropping.
"defect_types" = subset of ["hand","face","limb","count","physics","text"].
"severity" 0-3 (0 none, 3 catastrophic). "confidence" 0.0-1.0. "note" = short phrase.
Mark defective=true ONLY if a careful human would clearly agree. When the prompt states a number (e.g. "five wine glasses"), count the rendered objects and flag a mismatch as "count".
Output ONLY the JSON object."""

_M = {}


def resolve_model():
    hits = glob.glob("/mnt/models/hf_cache_v3/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/*")
    hits = [h for h in hits if os.path.exists(os.path.join(h, "config.json"))]
    assert hits, "Qwen2.5-VL-32B snapshot not found (download incomplete)"
    return hits[0]


def get_model():
    if not _M:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        md = resolve_model()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        _M["proc"] = AutoProcessor.from_pretrained(md, use_fast=True)
        _M["proc"].tokenizer.padding_side = "left"  # decoder-only batched gen needs left pad
        _M["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            md, quantization_config=bnb, torch_dtype=torch.float16, device_map="cuda").eval()
    return _M["model"], _M["proc"]


def parse_json(resp):
    for c in reversed(re.findall(r"\{[^{}]*\}", resp, re.DOTALL)):
        try:
            d = json.loads(c)
            if "defective" in d:
                dt = [t for t in (d.get("defect_types") or []) if t in TYPES6]
                return {"defective": bool(d.get("defective")), "defect_types": dt,
                        "severity": int(d.get("severity", 0)), "confidence": float(d.get("confidence", 0.0)),
                        "note": str(d.get("note", ""))[:80]}
        except Exception:
            continue
    return None


@torch.no_grad()
def judge_batch(items, do_sample=False, temp=0.0):
    """items = [(png_path, prompt), ...]"""
    from PIL import Image
    model, proc = get_model()
    msgs_l, imgs = [], []
    for path, prompt in items:
        msgs_l.append(proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": RUBRIC.format(PROMPT=prompt)}]}],
            tokenize=False, add_generation_prompt=True))
        imgs.append(Image.open(path).convert("RGB"))
    inp = proc(text=msgs_l, images=imgs, padding=True, return_tensors="pt").to("cuda")
    gen = model.generate(**inp, max_new_tokens=200, do_sample=do_sample,
                         temperature=(temp if do_sample else None), top_p=(0.9 if do_sample else None))
    return proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)


def load_prompts(model):
    """file -> prompt, from manifest.csv."""
    mp = {}
    with open(os.path.join(V3, model, "E1_data", "manifest.csv")) as f:
        for r in csv.DictReader(f):
            mp[r["file"]] = r["prompt_text"]
    return mp


def run_full(model, batch):
    base = os.path.join(V3, model, "E1_data")
    prompts = load_prompts(model)
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(base, "raw", "*.npz")))
    csvp = os.path.join(base, "labels.csv")
    done = set()
    if os.path.exists(csvp):
        with open(csvp) as f:
            next(f, None); done = {ln.split(",")[0] for ln in f if ln.strip()}
    todo = [f for f in files if f not in done]
    print(f"[{model}] labeling {len(todo)}/{len(files)} ({len(done)} cached)", flush=True)
    new = not os.path.exists(csvp)
    fh = open(csvp, "a")
    if new:
        fh.write("file,defective,defect_types,severity,confidence,note\n")
    t0 = time.time(); n = 0; nbad = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        items = [(os.path.join(base, "png", f.replace(".npz", ".png")), prompts.get(f, "")) for f in chunk]
        try:
            resps = judge_batch(items)
        except Exception as e:
            print("batch err:", repr(e)[:150]); resps = [""] * len(chunk)
        for f, r in zip(chunk, resps):
            d = parse_json(r) or {"defective": False, "defect_types": [], "severity": 0, "confidence": 0.0, "note": "PARSE_FAIL"}
            if d["note"] == "PARSE_FAIL":
                nbad += 1
            note = d["note"].replace(",", ";").replace('"', "'")
            fh.write(f'{f},{int(d["defective"])},{"|".join(d["defect_types"])},{d["severity"]},{d["confidence"]:.3f},"{note}"\n')
            n += 1
        fh.flush()
        if (i // batch) % 10 == 0:
            print(f"[{n}/{len(todo)}] {time.time()-t0:.0f}s ({(time.time()-t0)/max(n,1):.2f}s/img) bad={nbad}", flush=True)
    fh.close()
    print(f"[{model}] E1.5 done: {n} new, {nbad} parse-fail -> {csvp}")


def run_reliability(model, n_sample=500):
    base = os.path.join(V3, model, "E1_data")
    prompts = load_prompts(model)
    prim = {}
    with open(os.path.join(base, "labels.csv")) as f:
        next(f)
        for ln in f:
            p = ln.split(","); prim[p[0]] = int(p[1])
    sample = random.Random(42).sample(list(prim), min(n_sample, len(prim)))
    a, b = [], []
    for i in range(0, len(sample), 4):
        chunk = sample[i:i + 4]
        items = [(os.path.join(base, "png", f.replace(".npz", ".png")), prompts.get(f, "")) for f in chunk]
        for f, r in zip(chunk, judge_batch(items, do_sample=True, temp=0.8)):
            d = parse_json(r)
            if d:
                a.append(prim[f]); b.append(int(d["defective"]))
    a, b = np.array(a), np.array(b)
    po = float((a == b).mean())
    pe = sum(((a == k).mean() * (b == k).mean()) for k in (0, 1))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    rel = {"n": int(len(a)), "pct_agreement": round(po, 4), "cohen_kappa": round(float(kappa), 4),
           "primary_defect_rate": round(float(a.mean()), 4), "method": "primary greedy vs 2nd-pass temp0.8"}
    json.dump(rel, open(os.path.join(base, "reliability.json"), "w"), indent=2)
    print("reliability ->", json.dumps(rel))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reliability", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        base = os.path.join(V3, a.model, "E1_data"); prompts = load_prompts(a.model)
        fs = sorted(glob.glob(os.path.join(base, "raw", "*.npz")))[:a.smoke]
        items = [(os.path.join(base, "png", os.path.basename(f).replace(".npz", ".png")),
                  prompts.get(os.path.basename(f), "")) for f in fs]
        for (path, pr), r in zip(items, judge_batch(items)):
            print(os.path.basename(path), "prompt=", pr[:30], "->", parse_json(r))
    else:
        run_full(a.model, a.batch)
        if a.reliability:
            run_reliability(a.model)
