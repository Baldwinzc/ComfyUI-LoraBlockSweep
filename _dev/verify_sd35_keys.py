"""Standalone regex-vs-resolved-key check for the SD3.5 adapter.

Loads the SD3.5 Large checkpoint + the LoRA, runs ComfyUI's lora-key resolver,
then classifies every resolved key with our adapter's BlockSpec to confirm
every patch lands in J00..J37 (and which ones, if any, fall into `extras`).

Run on the machine where ComfyUI lives, with its Python env. Point it at your
ComfyUI install and model files via flags or env vars:

  COMFYUI_DIR=/path/to/ComfyUI \
  python verify_sd35_keys.py \
      --ckpt "models/checkpoints/Stable Diffusion 3.5 Large.safetensors" \
      --lora "models/loras/your-sd35-lora.safetensors"

Relative --ckpt / --lora paths are resolved against COMFYUI_DIR.
"""
import argparse
import os
import sys
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui-dir",
                        default=os.environ.get("COMFYUI_DIR",
                                               os.path.expanduser("~/ComfyUI")),
                        help="ComfyUI install dir (or set COMFYUI_DIR). "
                             "Default: ~/ComfyUI")
    parser.add_argument("--ckpt", required=True,
                        help="SD3.5 Large checkpoint (abs path, or relative "
                             "to --comfyui-dir)")
    parser.add_argument("--lora", required=True,
                        help="LoRA file (abs path, or relative to --comfyui-dir)")
    args = parser.parse_args()

    comfyui_dir = os.path.abspath(os.path.expanduser(args.comfyui_dir))
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Make ComfyUI + this custom node importable
    sys.path.insert(0, comfyui_dir)
    sys.path.insert(0, repo_dir)

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(comfyui_dir, p)

    ckpt = _resolve(args.ckpt)
    lora = _resolve(args.lora)

    import comfy.lora
    import comfy.lora_convert
    import comfy.utils
    import comfy.sd

    from lora_block_weight._sd35 import SD35_SPEC
    from lora_block_weight._core import classify_key

    print(f"loading checkpoint to build model_lora_keys_unet map...\n  {ckpt}")
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ckpt, output_vae=False, output_clip=False)

    lora_sd = comfy.utils.load_torch_file(lora, safe_load=True)
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
    print(f"\ntotal={total}  in-J-blocks={total - extras}  extras={extras}")
    if extras:
        print("(extras = LoRA patches not classified into any J block; "
              "review the regex if non-zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
