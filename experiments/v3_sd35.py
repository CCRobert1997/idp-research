"""IDP v3 — SD3.5-Medium (M2) generation + MMDiT image-stream activation capture.
24 JointTransformerBlocks; per block, image-stream hidden state out[-1] (B,N_tok,D),
GAP over tokens -> (D,). Capture all denoising steps. Reuses nothing from v1/v2 (DiT).
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/home/ubuntu/shangyu_comfyui/geopara_research/models/sd35m"
DEVICE, DTYPE = "cuda", torch.float16
STEPS, GUIDANCE, RES = 28, 4.5, 1024
N_BLOCKS = 24
BLOCK_NAMES = [f"B{i:02d}" for i in range(N_BLOCKS)]

_PIPE = {"p": None}


def get_pipe():
    if _PIPE["p"] is None:
        from diffusers import StableDiffusion3Pipeline
        p = StableDiffusion3Pipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
        p.to(DEVICE)
        p.set_progress_bar_config(disable=True)
        _PIPE["p"] = p
    return _PIPE["p"]


@torch.no_grad()
def generate(prompt, seed, capture=True):
    """Returns (pil, feats{name:(STEPS,D) fp16} or None, final_latent_gap(16,) fp16)."""
    pipe = get_pipe()
    blocks = pipe.transformer.transformer_blocks
    buf = {n: [] for n in BLOCK_NAMES}
    handles = []
    if capture:
        for name, blk in zip(BLOCK_NAMES, blocks):
            def mk(nm):
                def hook(m, inp, out):
                    img = out[-1] if isinstance(out, tuple) else out  # image stream
                    v = img[img.shape[0] // 2] if img.shape[0] == 2 else img[0]  # cond half
                    buf[nm].append(v.float().mean(dim=0).cpu().numpy().astype(np.float16))
                return hook
            handles.append(blk.register_forward_hook(mk(name)))

    last = {"lat": None}
    def cb(pipe, step, t, kw):
        last["lat"] = kw["latents"]
        return kw

    g = torch.Generator(DEVICE).manual_seed(int(seed))
    img = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
               height=RES, width=RES, generator=g,
               callback_on_step_end=cb,
               callback_on_step_end_tensor_inputs=["latents"]).images[0]
    for h in handles:
        h.remove()

    feats = None
    if capture:
        feats = {}
        for n in BLOCK_NAMES:
            assert len(buf[n]) == STEPS, f"{n}: {len(buf[n])} != {STEPS}"
            feats[n] = np.stack(buf[n])  # (STEPS, D)
    fl = last["lat"][0].float().mean(dim=(1, 2)).cpu().numpy().astype(np.float16)  # (16,)
    return img, feats, fl
