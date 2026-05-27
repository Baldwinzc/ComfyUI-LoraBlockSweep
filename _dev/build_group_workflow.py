"""Build the 4-variant group-hero workflow JSON:
  Full LoRA   |   Top-7 off   |   Bot-7 off   |   No LoRA

Top-7 by MSE (most impactful): D00, D02, D03, D07, D08, D09, D15
Bot-7 by MSE (least impactful): D01, D10, D12, D13, D14, D17, D18

Uses LoraBlockSweepFluxCustom with 4 explicit 57-value weight strings, all
sharing the same UNet / CLIP / VAE / latent / sampler so only the LoRA
weighting differs between branches.
"""
import json
from pathlib import Path

D_COUNT = 19
S_COUNT = 38
D = [f"D{i:02d}" for i in range(D_COUNT)]
S = [f"S{i:02d}" for i in range(S_COUNT)]
ALL = D + S
assert len(ALL) == 57

TOP_7 = {"D00", "D02", "D03", "D07", "D08", "D09", "D15"}
BOT_7 = {"D01", "D10", "D12", "D13", "D14", "D17", "D18"}


def weights(off_set, all_off=False):
    if all_off:
        return ",".join(["0"] * 57)
    return ",".join("0" if tag in off_set else "1" for tag in ALL)


variants = [
    ("full",   weights(set()),    "Full LoRA"),
    ("top7",   weights(TOP_7),    "Top-7 off"),
    ("bot7",   weights(BOT_7),    "Bot-7 off"),
    ("none",   weights(set(), all_off=True), "No LoRA"),
]

for slug, w, label in variants:
    print(f"  {label:12} ({slug:5}) {len(w.split(','))} weights, "
          f"on={w.count('1')} off={w.count('0')}")

# Shared graph: load model/clip/vae, encode prompt, build latent.
nodes = {
    "1": {"class_type": "UNETLoader",
          "inputs": {"unet_name": "flux1-Dev_FP8.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "DualCLIPLoader",
          "inputs": {"clip_name1": "clip_l.safetensors",
                     "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
    "3": {"class_type": "VAELoader",
          "inputs": {"vae_name": "ae.safetensors"}},
    "5": {"class_type": "ConditioningZeroOut",
          "inputs": {"conditioning": ["4_full", 0]}},
    "6": {"class_type": "EmptySD3LatentImage",
          "inputs": {"width": 768, "height": 768, "batch_size": 1}},
}

LORA_NAME = "flux_dev_frostinglane_araminta_k.safetensors"
PROMPT = ("frstingln illustration, portrait of a young person, head and "
          "shoulders, neutral expression, soft natural lighting, plain pastel "
          "background")
SAMPLE_PARAMS = dict(seed=42, steps=20, cfg=1.0, sampler_name="euler",
                     scheduler="simple", denoise=1.0)

# Per-variant branch: Custom LoRA -> KSampler -> VAEDecode -> SaveImage.
# Each branch re-encodes with its own CLIP (which is patched by the Custom node),
# so we use one CLIPTextEncode per variant (CLIP differs after the LoRA patches it).

for i, (slug, w, label) in enumerate(variants):
    base = 10 + i * 10
    nodes[str(base + 0)] = {
        "class_type": "LoraBlockSweepFluxCustom",
        "inputs": {
            "model": ["1", 0],
            "clip": ["2", 0],
            "lora_name": LORA_NAME,
            "weights": w,
            "baseline_weight": 1.0,
            "clip_strength": 1.0,
        },
    }
    nodes[str(base + 1)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": PROMPT, "clip": [str(base + 0), 1]},
    }
    nodes[str(base + 2)] = {
        "class_type": "ConditioningZeroOut",
        "inputs": {"conditioning": [str(base + 1), 0]},
    }
    nodes[str(base + 3)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [str(base + 0), 0],
            "positive": [str(base + 1), 0],
            "negative": [str(base + 2), 0],
            "latent_image": ["6", 0],
            **SAMPLE_PARAMS,
        },
    }
    nodes[str(base + 4)] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [str(base + 3), 0], "vae": ["3", 0]},
    }
    nodes[str(base + 5)] = {
        "class_type": "SaveImage",
        "inputs": {"images": [str(base + 4), 0],
                   "filename_prefix": f"lbw_group_{slug}"},
    }

# Drop placeholder ConditioningZeroOut node 5 (it was a leftover skeleton).
nodes.pop("5", None)

out = Path(__file__).parent / "group_hero_workflow.json"
out.write_text(json.dumps(nodes, indent=2))
print(f"\nwrote {out}")
print(f"  {len(nodes)} nodes total, 4 branches sharing UNet/CLIP/VAE/latent")
