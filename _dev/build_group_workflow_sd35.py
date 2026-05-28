"""Build the 4-variant group-hero workflow JSON for SD3.5 Large:
  Full LoRA   |   Top-N off   |   Bot-N off   |   No LoRA

Edit TOP_BLOCKS and BOT_BLOCKS after running fetch_and_analyze.py --model sd35
on the Stage 1 sweep results — pick the N highest-MSE and N lowest-MSE blocks.

Uses LoraBlockSweepSD35Custom with 4 explicit 38-value weight strings, all
sharing the same checkpoint / triple CLIP / latent / sampler so only the LoRA
weighting differs between branches.
"""
import json
from pathlib import Path

J_COUNT = 38
J = [f"J{i:02d}" for i in range(J_COUNT)]
assert len(J) == 38

# Top-12 by MSE (most impactful, MSE 0.0048..0.0095):
TOP_BLOCKS = {"J07", "J24", "J26", "J21", "J30", "J22", "J00", "J20", "J25", "J09", "J01", "J19"}
# Bot-12 by MSE (least impactful, MSE 0.00021..0.0023):
BOT_BLOCKS = {"J10", "J35", "J05", "J08", "J36", "J13", "J06", "J14", "J12", "J34", "J02", "J37"}
TOP_LABEL = f"Top-{len(TOP_BLOCKS)} off"
BOT_LABEL = f"Bot-{len(BOT_BLOCKS)} off"


def weights(off_set, all_off=False):
    if all_off:
        return ",".join(["0"] * J_COUNT)
    return ",".join("0" if tag in off_set else "1" for tag in J)


variants = [
    ("full", weights(set()),               "Full LoRA"),
    ("top",  weights(TOP_BLOCKS),          TOP_LABEL),
    ("bot",  weights(BOT_BLOCKS),          BOT_LABEL),
    ("none", weights(set(), all_off=True), "No LoRA"),
]

for slug, w, label in variants:
    print(f"  {label:12} ({slug:5}) {len(w.split(','))} weights, "
          f"on={w.count('1')} off={w.count('0')}")

CKPT_NAME = "Stable Diffusion 3.5 Large.safetensors"
CLIP_L = "clip_l.safetensors"
CLIP_G = "clip_g.safetensors"
CLIP_T5 = "t5xxl_fp16.safetensors"
LORA_NAME = "sd35-large-anime.safetensors"
PROMPT = ("anime portrait of a young woman with brown bob hair and blue eyes, "
          "head and shoulders, soft lighting, pastel background")
NEG_PROMPT = "photo, blurry, low quality"
SAMPLE_PARAMS = dict(seed=42, steps=20, cfg=4.5, sampler_name="euler",
                     scheduler="sgm_uniform", denoise=1.0)
WIDTH, HEIGHT = 1024, 1024

nodes = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": CKPT_NAME}},
    "1b": {"class_type": "TripleCLIPLoader",
           "inputs": {"clip_name1": CLIP_L, "clip_name2": CLIP_G, "clip_name3": CLIP_T5}},
    "5": {"class_type": "CLIPTextEncode",
          "inputs": {"text": PROMPT, "clip": ["1b", 0]}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": NEG_PROMPT, "clip": ["1b", 0]}},
    "7": {"class_type": "EmptySD3LatentImage",
          "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
}

for i, (slug, w, label) in enumerate(variants):
    base = 10 + i * 10
    nodes[str(base + 0)] = {
        "class_type": "LoraBlockSweepSD35Custom",
        "inputs": {
            "model": ["1", 0],
            "clip": ["1b", 0],
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
        "inputs": {"samples": [str(base + 3), 0], "vae": ["1", 2]},
    }
    nodes[str(base + 5)] = {
        "class_type": "SaveImage",
        "inputs": {"images": [str(base + 4), 0],
                   "filename_prefix": f"lbw_sd35_group_{slug}"},
    }

out = Path(__file__).parent / "group_hero_workflow_sd35.json"
out.write_text(json.dumps(nodes, indent=2))
print(f"\nwrote {out}")
print(f"  {len(nodes)} nodes total, 4 branches sharing checkpoint/CLIP/latent")
