"""E1.6: build a self-contained human-annotation page for the kappa>=0.6 gate.
Stratified blind sample (oversample judge-positives) so kappa is estimable on both
classes. Shows prompt + rubric; user marks defective/clean; downloads CSV.
Outputs v3/{model}/E1.6_human/ (index.html + imgs/ + sample.json[judge labels, gitignored]).
Usage: v3_build_annotation.py --model M2 [--n_pos 50 --n_neg 50]
"""
import os, sys, json, csv, glob, shutil, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "v3")

RUBRIC_HTML = """A STRUCTURAL defect = malformed/abnormal-count anatomy (hands/fingers/limbs/eyes),
fused or merged objects, physically impossible geometry, garbled/unreadable text, OR object count
not matching the prompt's number. IGNORE pure style, aesthetics, blur, lighting, cropping.
Mark defective ONLY if you clearly see one."""


def main(model, n_pos, n_neg, seed=42):
    base = os.path.join(V3, model, "E1_data")
    out = os.path.join(V3, model, "E1.6_human")
    imgs = os.path.join(out, "imgs")
    os.makedirs(imgs, exist_ok=True)
    prompts = {}
    with open(os.path.join(base, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            prompts[r["file"]] = r["prompt_text"]
    rows = list(csv.DictReader(open(os.path.join(base, "labels.csv"))))
    pos = [r for r in rows if int(r["defective"]) == 1]
    neg = [r for r in rows if int(r["defective"]) == 0]
    rng = random.Random(seed)
    sample = rng.sample(pos, min(n_pos, len(pos))) + rng.sample(neg, min(n_neg, len(neg)))
    rng.shuffle(sample)  # blind: human can't infer judge label from order

    meta = []
    for i, r in enumerate(sample):
        fn = r["file"]; png = fn.replace(".npz", ".png")
        shutil.copy(os.path.join(base, "png", png), os.path.join(imgs, f"{i:03d}.png"))
        meta.append({"idx": i, "file": fn, "img": f"imgs/{i:03d}.png",
                     "prompt": prompts.get(fn, ""), "judge": int(r["defective"])})
    # sample.json keeps judge labels for later kappa (NOT shown in the page)
    json.dump(meta, open(os.path.join(out, "sample.json"), "w"), indent=2)

    cards = "\n".join(
        f'<div class=card><img src="{m["img"]}" loading=lazy>'
        f'<div class=info><b>#{m["idx"]:03d}</b> &nbsp; prompt: <i>{m["prompt"]}</i></div>'
        f'<label><input type=radio name="q{m["idx"]}" value=1>defective</label> '
        f'<label><input type=radio name="q{m["idx"]}" value=0>clean</label></div>'
        for m in meta)
    html = f"""<!doctype html><meta charset=utf-8><title>IDP {model} human annotation</title>
<style>body{{font-family:system-ui;max-width:760px;margin:auto;padding:16px}}
.card{{border:1px solid #ccc;border-radius:8px;padding:10px;margin:14px 0}}
img{{max-width:100%;border-radius:6px}}.info{{margin:8px 0;font-size:14px}}
label{{margin-right:16px}}#bar{{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:2px solid #333}}
button{{padding:8px 16px;font-size:15px}}</style>
<div id=bar><h3>IDP {model} — human defect annotation ({len(meta)} images)</h3>
<p style="font-size:13px">{RUBRIC_HTML}</p>
<button onclick=dl()>Download my labels (CSV)</button> <span id=cnt></span></div>
{cards}
<script>
const N={len(meta)};
function cnt(){{let d=0;for(let i=0;i<N;i++)if(document.querySelector(`input[name=q${{i}}]:checked`))d++;
document.getElementById('cnt').textContent=` ${{d}}/${{N}} done`;}}
document.addEventListener('change',cnt);
function dl(){{let s="idx,human\\n";for(let i=0;i<N;i++){{let e=document.querySelector(`input[name=q${{i}}]:checked`);
s+=`${{i}},${{e?e.value:''}}\\n`;}}
let a=document.createElement('a');a.href='data:text/csv,'+encodeURIComponent(s);
a.download='{model}_human_labels.csv';a.click();}}
cnt();
</script>"""
    open(os.path.join(out, "index.html"), "w").write(html)
    print(f"[{model}] annotation page -> {out}  ({len(meta)} imgs: {min(n_pos,len(pos))} judge-pos + {min(n_neg,len(neg))} judge-neg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_pos", type=int, default=50)
    ap.add_argument("--n_neg", type=int, default=50)
    a = ap.parse_args()
    main(a.model, a.n_pos, a.n_neg)
