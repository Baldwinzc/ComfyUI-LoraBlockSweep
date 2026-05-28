"""Stable Diffusion 3.5 Large adapter for LoRA block sweeps.

SD3.5 Large is an MMDiT with 38 `joint_blocks` (joint image+text attention
inside each block, split into `context_block` / `x_block` halves). LoRA keys
live under `diffusion_model.joint_blocks.{N}.` — we tag them J00..J37.

The internal key prefix `joint_blocks` is unique to SD3 / SD3.5 (vs FLUX's
`double_blocks`/`single_blocks` or Qwen's `transformer_blocks`). ComfyUI's
SD3 LoRA key mapping in `comfy/lora.py` strips the `diffusion_model.` /
`.weight` framing, so the same regex used for state-dict key classification
also matches LoRA file keys after `convert_lora` + `load_lora` resolution.

SD3.5 Medium (24 blocks) is not currently exposed as a separate node — its
block count differs but the key layout is identical, so the same SPEC could
be reused if `count` were parameterized.
"""
from ._core import (
    BlockSpec,
    _BatchSweepBase,
    _CustomSweepBase,
    _GroupSweepBase,
    _SingleBlockSweepBase,
)


SD35_SPEC = BlockSpec(
    model_key="sd35",
    display_name="SD3.5 Large",
    tag_groups={
        "J": (38, r"diffusion_model\.joint_blocks\.(\d+)\."),
    },
    default_groups=(
        "J00-J09\n"
        "J10-J19\n"
        "J20-J29\n"
        "J30-J37\n"
        "J00-J18\n"
        "J19-J37"
    ),
    default_first_round_blocks=",".join(f"J{i:02d}" for i in range(38)),
)


class LoraBlockSweepSD35(_SingleBlockSweepBase):
    SPEC = SD35_SPEC


class LoraBlockSweepSD35Custom(_CustomSweepBase):
    SPEC = SD35_SPEC


class LoraBlockSweepSD35Batch(_BatchSweepBase):
    SPEC = SD35_SPEC


class LoraBlockSweepSD35Group(_GroupSweepBase):
    SPEC = SD35_SPEC


NODE_CLASS_MAPPINGS = {
    "LoraBlockSweepSD35": LoraBlockSweepSD35,
    "LoraBlockSweepSD35Custom": LoraBlockSweepSD35Custom,
    "LoraBlockSweepSD35Batch": LoraBlockSweepSD35Batch,
    "LoraBlockSweepSD35Group": LoraBlockSweepSD35Group,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraBlockSweepSD35": "LoRA Block Sweep (SD3.5 Large)",
    "LoraBlockSweepSD35Custom": "LoRA Block Sweep Custom (SD3.5 Large)",
    "LoraBlockSweepSD35Batch": "LoRA Block Sweep Batch (SD3.5 Large)",
    "LoraBlockSweepSD35Group": "LoRA Block Sweep Group (SD3.5 Large)",
}
