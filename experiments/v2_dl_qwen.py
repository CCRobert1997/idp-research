import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir="/home/ubuntu/shangyu_comfyui/hf_cache",
        ignore_patterns=["*.pth","original/*","*.gguf"])
print("DOWNLOADED ->", p)
