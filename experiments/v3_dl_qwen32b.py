from huggingface_hub import snapshot_download
p = snapshot_download(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    cache_dir="/mnt/models/hf_cache_v3",
    ignore_patterns=["*.pth", "original/*", "*.gguf"],
)
print("DOWNLOADED ->", p)
