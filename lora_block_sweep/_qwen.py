"""Qwen-Image adapter for LoRA block sweeps.

Qwen-Image is a single-stream MMDiT with 60 `transformer_blocks` (joint
image+text attention inside each block). LoRA keys live under
`diffusion_model.transformer_blocks.{N}.` — we tag them B00..B59.

ComfyUI's QwenImage LoRA key mapping (see `comfy/lora.py:320`) strips the
`diffusion_model.` / `.weight` framing and exposes the inner key as a LoRA
target, so the same regex that classifies state_dict keys here also matches
LoRA file keys after `convert_lora` + `load_lora` resolution.
"""
from ._core import (
    BlockSpec,
    _BatchSweepBase,
    _CustomSweepBase,
    _GroupSweepBase,
    _SingleBlockSweepBase,
)


QWEN_SPEC = BlockSpec(
    model_key="qwen",
    display_name="Qwen-Image",
    tag_groups={
        "B": (60, r"diffusion_model\.transformer_blocks\.(\d+)\."),
    },
    default_groups=(
        "B00-B09\n"
        "B10-B19\n"
        "B20-B29\n"
        "B30-B39\n"
        "B40-B49\n"
        "B50-B59\n"
        "B00-B29\n"
        "B30-B59"
    ),
    default_first_round_blocks=",".join(f"B{i:02d}" for i in range(60)),
)


class LoraBlockSweepQwen(_SingleBlockSweepBase):
    SPEC = QWEN_SPEC


class LoraBlockSweepQwenCustom(_CustomSweepBase):
    SPEC = QWEN_SPEC


class LoraBlockSweepQwenBatch(_BatchSweepBase):
    SPEC = QWEN_SPEC


class LoraBlockSweepQwenGroup(_GroupSweepBase):
    SPEC = QWEN_SPEC


NODE_CLASS_MAPPINGS = {
    "LoraBlockSweepQwen": LoraBlockSweepQwen,
    "LoraBlockSweepQwenCustom": LoraBlockSweepQwenCustom,
    "LoraBlockSweepQwenBatch": LoraBlockSweepQwenBatch,
    "LoraBlockSweepQwenGroup": LoraBlockSweepQwenGroup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraBlockSweepQwen": "LoRA Block Sweep (Qwen-Image)",
    "LoraBlockSweepQwenCustom": "LoRA Block Sweep Custom (Qwen-Image)",
    "LoraBlockSweepQwenBatch": "LoRA Block Sweep Batch (Qwen-Image)",
    "LoraBlockSweepQwenGroup": "LoRA Block Sweep Group (Qwen-Image)",
}
