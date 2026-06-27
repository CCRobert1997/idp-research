"""
IDP (Intermediate Defect Prediction) — shared library.

Generates SD1.5 images with a manual DDIM loop while capturing, at every
denoising step, the global-average-pooled output of 22 UNet resnet blocks.
Also runs MediaPipe Hands and stores ALL raw signals so the binary
clean/defective label can be cheaply re-derived later (de-risks the
expensive E1 run against label-rule uncertainty).

venv: ~/shangyu_comfyui/venvs/comfyui/bin/python  (diffusers 0.31, mediapipe 0.10, sklearn 1.8, torch 2.11)
weights: IDP_research/models_sd15  (diffusers fp16, symlink to geopara sd15)
"""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models_sd15")
DEVICE = "cuda"
DTYPE = torch.float16

# Generation spec (locked, from the experiment doc)
STEPS = 50
GUIDANCE = 7.5
RES = 512
LATENT_RES = RES // 8  # 64

# ---- The 22 UNet resnet blocks (SD1.5 diffusers naming) ------------------
# (module attribute path, short name) in fixed order. AUC map column order
# follows this list (index 0..21).
BLOCK_SPECS = [
    ("down_blocks.0.resnets.0", "D0R0"), ("down_blocks.0.resnets.1", "D0R1"),
    ("down_blocks.1.resnets.0", "D1R0"), ("down_blocks.1.resnets.1", "D1R1"),
    ("down_blocks.2.resnets.0", "D2R0"), ("down_blocks.2.resnets.1", "D2R1"),
    ("down_blocks.3.resnets.0", "D3R0"), ("down_blocks.3.resnets.1", "D3R1"),
    ("mid_block.resnets.0", "MidR0"), ("mid_block.resnets.1", "MidR1"),
    ("up_blocks.0.resnets.0", "U0R0"), ("up_blocks.0.resnets.1", "U0R1"), ("up_blocks.0.resnets.2", "U0R2"),
    ("up_blocks.1.resnets.0", "U1R0"), ("up_blocks.1.resnets.1", "U1R1"), ("up_blocks.1.resnets.2", "U1R2"),
    ("up_blocks.2.resnets.0", "U2R0"), ("up_blocks.2.resnets.1", "U2R1"), ("up_blocks.2.resnets.2", "U2R2"),
    ("up_blocks.3.resnets.0", "U3R0"), ("up_blocks.3.resnets.1", "U3R1"), ("up_blocks.3.resnets.2", "U3R2"),
]
BLOCK_NAMES = [n for _, n in BLOCK_SPECS]
N_BLOCKS = len(BLOCK_SPECS)  # 22

PROMPTS = [
    "a person waving at the camera",
    "a person giving a thumbs up",
    "a person holding a coffee cup",
    "a chef holding a kitchen knife",
    "a person pointing at something",
    "a person playing guitar",
    "a musician playing piano",
    "a person counting on their fingers",
    "a portrait of a person with hands visible",
    "a person clapping",
    "a person reaching out their hand",
    "a barista making coffee",
    "a person typing on a keyboard",
    "a person holding a book open",
    "a person making a peace sign",
    "a painter holding a paintbrush",
    "a person holding flowers",
    "a person shaking hands",
    "a basketball player holding a ball",
    "a surgeon in gloves",
]

# --------------------------------------------------------------------------
_PIPE = {"pipe": None}


def get_pipe():
    if _PIPE["pipe"] is not None:
        return _PIPE["pipe"]
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, variant="fp16", use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    _PIPE["pipe"] = pipe
    return pipe


def _register_hooks(unet, buf, step_ref):
    """Hooks capture the CFG-conditional half of each block output, GAP over
    spatial dims -> (C,), into buf[name][step] (lazily allocated)."""
    named = dict(unet.named_modules())
    handles = []
    for path, name in BLOCK_SPECS:
        mod = named[path]

        def make(nm):
            def hook(module, inp, out):
                o = out[1] if out.shape[0] == 2 else out[0]  # cond half under CFG
                vec = o.float().mean(dim=(-1, -2)).cpu().numpy()  # (C,)
                if nm not in buf:
                    buf[nm] = np.zeros((STEPS, vec.shape[0]), dtype=np.float16)
                buf[nm][step_ref[0]] = vec.astype(np.float16)
            return hook
        handles.append(mod.register_forward_hook(make(name)))
    return handles


@torch.no_grad()
def decode_latent(latent_tensor):
    """Decode a (1,4,64,64) latent tensor to a PIL image."""
    pipe = get_pipe()
    img = pipe.vae.decode(latent_tensor / pipe.vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    from PIL import Image
    return Image.fromarray((img * 255).round().astype(np.uint8))


@torch.no_grad()
def generate(prompt, seed, capture=True, return_step_latent=None):
    """Manual DDIM loop. Returns (pil_image, final_latent[4,64,64] fp16,
    feats {name:(50,C) fp16} or None). If return_step_latent=k (0-based), also
    stashes a decoded PIL of the latent at end of step k in the 4th return slot."""
    pipe = get_pipe()
    unet, vae, te, tok, sched = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler

    ti = tok([prompt], padding="max_length", max_length=tok.model_max_length,
             truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    ui = tok([""], padding="max_length", max_length=tok.model_max_length,
             truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    emb = te(ti)[0]
    uemb = te(ui)[0]
    context = torch.cat([uemb, emb])  # [uncond, cond]

    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    latents = torch.randn((1, 4, LATENT_RES, LATENT_RES), generator=g, device=DEVICE, dtype=DTYPE)
    sched.set_timesteps(STEPS, device=DEVICE)
    latents = latents * sched.init_noise_sigma

    buf, step_ref, handles = {}, [0], []
    step_latent = [None]
    if capture:
        handles = _register_hooks(unet, buf, step_ref)
    try:
        for i, t in enumerate(sched.timesteps):
            step_ref[0] = i
            lm = torch.cat([latents] * 2)
            lm = sched.scale_model_input(lm, t)
            noise = unet(lm, t, encoder_hidden_states=context).sample
            nu, nc = noise.chunk(2)
            noise = nu + GUIDANCE * (nc - nu)
            latents = sched.step(noise, t, latents).prev_sample
            if return_step_latent is not None and i == return_step_latent:
                step_latent[0] = latents.clone()
    finally:
        for h in handles:
            h.remove()

    final_latent = latents[0].float().cpu().numpy().astype(np.float16)
    img = vae.decode(latents / vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    from PIL import Image
    pil = Image.fromarray((img * 255).round().astype(np.uint8))

    feats = buf if capture else None
    if capture:
        missing = [n for n in BLOCK_NAMES if n not in buf]
        if missing:
            raise RuntimeError(f"blocks never fired: {missing}")
    if return_step_latent is not None:
        step_pil = decode_latent(step_latent[0]) if step_latent[0] is not None else None
        return pil, final_latent, feats, step_pil
    return pil, final_latent, feats


# ---- MediaPipe Hands (Tasks API HandLandmarker, reused from para_carve) ---
# This mediapipe build (0.10.35) ships ONLY the Tasks API, not legacy solutions.
HAND_TASK = os.path.join(ROOT, "hand_landmarker.task")
_MP = {"lm": None, "mp": None}
# 21-landmark finger chains (MediaPipe ordering)
_FINGERS = {
    "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20],
}
_TIPS = [4, 8, 12, 16, 20]
_PIPS = [2, 6, 10, 14, 18]


def get_hands():
    if _MP["lm"] is None:
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions, RunningMode)
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_TASK),
            running_mode=RunningMode.IMAGE, num_hands=4,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
        _MP["lm"] = HandLandmarker.create_from_options(opts)
        _MP["mp"] = mp
    return _MP["lm"], _MP["mp"]


def _geom_anomaly(lms):
    """Para-carve's vetted scale-normalized geometric-implausibility score for one
    21-landmark hand. Canonical hand: phalanx segments decrease distally and are
    non-degenerate. Score = mean violation, >=0; higher = more implausible."""
    pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6  # wrist->middle_mcp
    viol = []
    for chain in _FINGERS.values():
        segs = np.array([np.linalg.norm(pts[chain[i + 1]] - pts[chain[i]]) / scale
                         for i in range(len(chain) - 1)])
        viol.append(float(np.sum(segs < 0.02)))  # degenerate/collapsed
        for i in range(len(segs) - 1):
            if segs[i + 1] > segs[i] * 1.6:       # non-monotonic distal growth
                viol.append(1.0)
    return float(np.mean(viol)) if viol else 0.0


def _count_extended(lms):
    """Extended-finger heuristic (x,y normalized, y down). Thumb via x-spread."""
    pts = [(p.x, p.y) for p in lms]
    cnt = 0
    if abs(pts[4][0] - pts[2][0]) > abs(pts[3][0] - pts[2][0]):
        cnt += 1  # thumb extended (tip further from MCP in x than IP)
    for tip, pip in zip(_TIPS[1:], _PIPS[1:]):
        if pts[tip][1] < pts[pip][1]:
            cnt += 1
    return cnt


def analyze_hands(pil_img):
    """Run HandLandmarker; return ALL raw signals (label re-derivable later)."""
    lm, mp = get_hands()
    rgb = np.ascontiguousarray(np.array(pil_img.convert("RGB")))
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = lm.detect(mpimg)
    n = len(res.hand_landmarks)
    out = {"num_hands": n, "finger_count": -1, "det_score": 0.0,
           "best_geom": 9.99, "per_hand_scores": [], "per_hand_geom": [],
           "per_hand_fingers": [], "landmarks": []}
    if n == 0:
        return out
    confs = [float(h[0].score) for h in res.handedness] if res.handedness else [0.0] * n
    geoms = [_geom_anomaly(h) for h in res.hand_landmarks]
    fingers = [_count_extended(h) for h in res.hand_landmarks]
    out["per_hand_scores"] = [round(c, 4) for c in confs]
    out["per_hand_geom"] = [round(g, 4) for g in geoms]
    out["per_hand_fingers"] = fingers
    out["landmarks"] = [[[p.x, p.y, p.z] for p in h] for h in res.hand_landmarks]
    out["det_score"] = round(max(confs), 4)
    best_i = int(np.argmin(geoms))           # least-anomalous detected hand
    out["best_geom"] = round(geoms[best_i], 4)
    out["finger_count"] = fingers[best_i]
    return out


# ---- Label rule (PROVISIONAL — finalized on the pilot) --------------------
# All raw signals are stored in the npz, so this is cheaply re-derivable in E2.
# label: -1 no confident hand | 0 clean | 1 defective (geometrically implausible)
DET_CONF = 0.5    # a hand counts as "detected" only above this handedness score
GEOM_TAU = 0.5    # best_geom above this => defective (para-carve default; pilot-tuned)


def compute_label(rec, det_conf=DET_CONF, geom_tau=GEOM_TAU):
    """Scheme S (spec-strict): -1 no confident hand | 0 clean | 1 geom-defective.
    NOTE: pilot showed class 1 is ~empty (defects live in the no-hand bucket) -> degenerate."""
    confident = [c for c in rec.get("per_hand_scores", []) if c >= det_conf]
    if not confident:
        return -1
    return 1 if rec["best_geom"] > geom_tau else 0


def compute_label_inclusive(rec, det_conf=DET_CONF, geom_tau=GEOM_TAU):
    """Scheme I (no-hand-inclusive, PRIMARY): 0 clean-detected hand |
    1 defective = (no confident hand) OR (geom-anomalous). Non-degenerate.
    This is the well-posed early-defect-prediction target (decided from pilot 72/28)."""
    confident = [c for c in rec.get("per_hand_scores", []) if c >= det_conf]
    bad = (not confident) or (rec["best_geom"] > geom_tau)
    return 1 if bad else 0
