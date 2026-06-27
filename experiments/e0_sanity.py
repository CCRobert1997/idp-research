"""E0: environment validation. 5 images of 'a person waving', MediaPipe diag,
hook-capture sanity. Outputs E0_sanity/."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idp_common as C

OUT = os.path.join(C.ROOT, "E0_sanity")
os.makedirs(OUT, exist_ok=True)

def main():
    lines = []
    print("loading pipe..."); C.get_pipe(); print("pipe ok")
    for i in range(5):
        pil, latent, feats = C.generate("a person waving", seed=i, capture=True)
        pil.save(os.path.join(OUT, f"sanity_{i}.png"))
        a = C.analyze_hands(pil)
        # feats sanity
        shapes = {k: feats[k].shape for k in C.BLOCK_NAMES}
        assert all(s[0] == C.STEPS for s in shapes.values()), shapes
        msg = (f"img {i}: hands={a['num_hands']} primary_fingers={a['finger_count']} "
               f"det_score={a['det_score']:.3f} best_geom={a['best_geom']:.3f} "
               f"per_hand_geom={a['per_hand_geom']} label={C.compute_label(a)}")
        print(msg); lines.append(msg)
    # report captured block channel dims (verify all 22 fire)
    lines.append("")
    lines.append(f"captured {len(feats)} / {C.N_BLOCKS} blocks")
    lines.append("block channel dims: " + json.dumps({k: int(feats[k].shape[1]) for k in C.BLOCK_NAMES}))
    lines.append(f"feats per-image float16 MB ~= {sum(feats[k].nbytes for k in feats)/1e6:.2f}")
    with open(os.path.join(OUT, "sanity_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print("E0 DONE ->", OUT)

if __name__ == "__main__":
    main()
