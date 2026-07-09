"""IDP v3 — FLUX.1-dev (M3) generation + MMDiT activation capture.
12B, runs via enable_model_cpu_offload (fits 40G, no install). Guidance-distilled =>
ONE forward per step (no CFG batch). 57 blocks (19 dual transformer_blocks +
38 single_transformer_blocks); subsample ~24 evenly. Image/hidden stream GAP over tokens.
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/home/ubuntu/shangyu_comfyui/geopara_research/models/flux-dev"
DEVICE, DTYPE = "cuda", torch.bfloat16
STEPS, GUIDANCE, RES = 28, 3.5, 1024

# subsample blocks: 12 of 19 dual + 12 of 38 single
_DUAL_IDX = [int(round(x)) for x in np.linspace(0, 18, 12)]
_SINGLE_IDX = [int(round(x)) for x in np.linspace(0, 37, 12)]
BLOCK_NAMES = [f"D{i:02d}" for i in _DUAL_IDX] + [f"S{i:02d}" for i in _SINGLE_IDX]
N_BLOCKS = len(BLOCK_NAMES)

_PIPE = {"p": None}


def get_pipe():
    # 4-bit quantized (nf4) transformer + T5 => resident on 40G GPU, ~16s/img (vs 68s offloaded).
    # NOTE: FLUX-4bit is a slightly different model than bf16 — documented caveat.
    if _PIPE["p"] is None:
        from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DBnB
        from transformers import T5EncoderModel, BitsAndBytesConfig as TBnB
        tr = FluxTransformer2DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer",
            quantization_config=DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE),
            torch_dtype=DTYPE)
        t5 = T5EncoderModel.from_pretrained(
            MODEL_PATH, subfolder="text_encoder_2",
            quantization_config=TBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE),
            torch_dtype=DTYPE)
        p = FluxPipeline.from_pretrained(MODEL_PATH, transformer=tr, text_encoder_2=t5, torch_dtype=DTYPE)
        p.to("cuda")
        p.set_progress_bar_config(disable=True)
        _PIPE["p"] = p
    return _PIPE["p"]


def _gap(out):
    h = out[-1] if isinstance(out, tuple) else out      # image / hidden stream
    return h[0].float().mean(dim=0).cpu().numpy().astype(np.float16)  # GAP over tokens -> (D,)


@torch.no_grad()
def generate(prompt, seed, capture=True):
    pipe = get_pipe()
    tr = pipe.transformer
    buf = {n: [] for n in BLOCK_NAMES}
    handles = []
    if capture:
        for i in _DUAL_IDX:
            handles.append(tr.transformer_blocks[i].register_forward_hook(
                (lambda nm: lambda m, inp, out: buf[nm].append(_gap(out)))(f"D{i:02d}")))
        for i in _SINGLE_IDX:
            handles.append(tr.single_transformer_blocks[i].register_forward_hook(
                (lambda nm: lambda m, inp, out: buf[nm].append(_gap(out)))(f"S{i:02d}")))
    last = {"lat": None}
    def cb(pipe, step, t, kw):
        last["lat"] = kw["latents"]; return kw

    g = torch.Generator("cpu").manual_seed(int(seed))
    img = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
               height=RES, width=RES, generator=g,
               callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
    for h in handles:
        h.remove()
    feats = None
    if capture:
        feats = {}
        for n in BLOCK_NAMES:
            assert len(buf[n]) == STEPS, f"{n}: {len(buf[n])} != {STEPS}"
            feats[n] = np.stack(buf[n])
    fl = last["lat"][0].float().mean(dim=0).cpu().numpy().astype(np.float16)  # packed latent GAP
    return img, feats, fl
