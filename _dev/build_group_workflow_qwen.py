"""Build the 4-variant group-hero workflow JSON for Qwen-Image:
  Full LoRA   |   Top-N off   |   Bot-N off   |   No LoRA

Edit TOP_BLOCKS and BOT_BLOCKS after running fetch_and_analyze.py --model qwen
on the Stage 1 sweep results — pick the N highest-MSE and N lowest-MSE blocks.

Uses LoraBlockWeightQwenCustom with 4 explicit 60-value weight strings, all
sharing the same UNet / CLIP / VAE / latent / sampler so only the LoRA
weighting differs between branches.
"""
import json
from pathlib import Path

B_COUNT = 60
B = [f"B{i:02d}" for i in range(B_COUNT)]
assert len(B) == 60

# Top-12 by MSE (most impactful, MSE 0.0021..0.0059):
TOP_BLOCKS = {"B29", "B28", "B38", "B31", "B18", "B30", "B16", "B37", "B15", "B19", "B00", "B34"}
# Bot-12 by MSE (least impactful, MSE 0.00024..0.00066):
BOT_BLOCKS = {"B50", "B07", "B06", "B05", "B24", "B56", "B10", "B12", "B09", "B21", "B25", "B11"}
TOP_LABEL = f"Top-{len(TOP_BLOCKS)} off"
BOT_LABEL = f"Bot-{len(BOT_BLOCKS)} off"


def weights(off_set, all_off=False):
    if all_off:
        return ",".join(["0"] * B_COUNT)
    return ",".join("0" if tag in off_set else "1" for tag in B)


variants = [
    ("full", weights(set()),               "Full LoRA"),
    ("top",  weights(TOP_BLOCKS),          TOP_LABEL),
    ("bot",  weights(BOT_BLOCKS),          BOT_LABEL),
    ("none", weights(set(), all_off=True), "No LoRA"),
]

for slug, w, label in variants:
    print(f"  {label:12} ({slug:5}) {len(w.split(','))} weights, "
          f"on={w.count('1')} off={w.count('0')}")

UNET_NAME = "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
CLIP_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
LORA_NAME = "qwen-image-modern-anime-lora.safetensors"
PROMPT = ("Japanese modern anime style, portrait of a young woman with brown "
          "bob hair and blue eyes, head and shoulders, neutral expression, "
          "soft natural lighting, plain pastel background")
NEG_PROMPT = "photo, cg, 3d"
SAMPLE_PARAMS = dict(seed=42, steps=20, cfg=4.0, sampler_name="euler",
                     scheduler="simple", denoise=1.0)
WIDTH, HEIGHT = 768, 768

nodes = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
    "2": {"class_type": "ModelSamplingAuraFlow",
          "inputs": {"model": ["1", 0], "shift": 4.0}},
    "3": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image"}},
    "4": {"class_type": "VAELoader",
          "inputs": {"vae_name": VAE_NAME}},
    "5": {"class_type": "CLIPTextEncode",
          "inputs": {"text": PROMPT, "clip": ["3", 0]}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
    "7": {"class_type": "EmptySD3LatentImage",
          "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
}

for i, (slug, w, label) in enumerate(variants):
    base = 10 + i * 10
    nodes[str(base + 0)] = {
        "class_type": "LoraBlockWeightQwenCustom",
        "inputs": {
            "model": ["2", 0],
            "clip": ["3", 0],
            "lora_name": LORA_NAME,
            "weights": w,
            "baseline_weight": 1.0,
            "clip_strength": 1.0,
        },
    }
    nodes[str(base + 3)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [str(base + 0), 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["7", 0],
            **SAMPLE_PARAMS,
        },
    }
    nodes[str(base + 4)] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [str(base + 3), 0], "vae": ["4", 0]},
    }
    nodes[str(base + 5)] = {
        "class_type": "SaveImage",
        "inputs": {"images": [str(base + 4), 0],
                   "filename_prefix": f"lbw_qwen_group_{slug}"},
    }

out = Path(__file__).parent / "group_hero_workflow_qwen.json"
out.write_text(json.dumps(nodes, indent=2))
print(f"\nwrote {out}")
print(f"  {len(nodes)} nodes total, 4 branches sharing UNet/CLIP/VAE/latent")
