"""Standalone regex-vs-resolved-key check for the SD3.5 adapter.

Loads the SD3.5 Large checkpoint + the LoRA, runs ComfyUI's lora-key resolver,
then classifies every resolved key with our adapter's BlockSpec to confirm
every patch lands in J00..J37 (and which ones, if any, fall into `extras`).
Run on the ComfyUI server with its conda env.
"""
import sys
from pathlib import Path
from collections import Counter

# Make ComfyUI imports work
sys.path.insert(0, "<ComfyUI>")
sys.path.insert(0, "<ComfyUI>/custom_nodes/ComfyUI-LoraBlockWeight")

import comfy.lora
import comfy.lora_convert
import comfy.utils
import comfy.sd

from lora_block_weight._sd35 import SD35_SPEC
from lora_block_weight._core import classify_key

CKPT = "<ComfyUI>/models/checkpoints/Stable Diffusion 3.5 Large.safetensors"
LORA = "<ComfyUI>/models/loras/sd35-large-anime.safetensors"

print("loading checkpoint to build model_lora_keys_unet map...")
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(CKPT, output_vae=False, output_clip=False)

lora_sd = comfy.utils.load_torch_file(LORA, safe_load=True)
lora_sd = comfy.lora_convert.convert_lora(lora_sd)
print(f"LoRA file: {len(lora_sd)} tensors")

key_map = comfy.lora.model_lora_keys_unet(model.model, {})
loaded = comfy.lora.load_lora(lora_sd, key_map)
print(f"resolved patches: {len(loaded)}")

tag_counts = Counter()
samples = {}
for k in loaded.keys():
    tag = classify_key(SD35_SPEC, k)
    tag_counts[tag] += 1
    samples.setdefault(tag, k if isinstance(k, str) else k[0])

print("\ntag distribution:")
for tag in sorted(tag_counts.keys()):
    sample = samples[tag]
    sample_str = sample if isinstance(sample, str) else sample[0]
    print(f"  {tag:>8}: {tag_counts[tag]:>4} patches   sample={sample_str[:90]}")

extras = tag_counts.get("extras", 0)
total = sum(tag_counts.values())
print(f"\ntotal={total}  in-J-blocks={total-extras}  extras={extras}")
if extras:
    print("(extras = LoRA patches not classified into any J block; review the regex if non-zero)")
