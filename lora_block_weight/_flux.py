"""FLUX adapter for LoRA block sweeps.

FLUX's transformer has 19 double-stream blocks then 38 single-stream blocks
(57 total). LoRA keys live under `diffusion_model.double_blocks.{N}.` and
`diffusion_model.single_blocks.{N}.` — we tag them D00..D18 and S00..S37.
"""
from ._core import (
    BlockSpec,
    _BatchSweepBase,
    _CustomSweepBase,
    _GroupSweepBase,
    _SingleBlockSweepBase,
)


FLUX_SPEC = BlockSpec(
    model_key="flux",
    display_name="FLUX",
    tag_groups={
        "D": (19, r"diffusion_model\.double_blocks\.(\d+)\."),
        "S": (38, r"diffusion_model\.single_blocks\.(\d+)\."),
    },
    default_groups=(
        "D00-D06\n"
        "D07-D12\n"
        "D13-D18\n"
        "S00-S12\n"
        "S13-S25\n"
        "S26-S37\n"
        "D00-D18\n"
        "S00-S37"
    ),
    default_first_round_blocks=",".join(f"D{i:02d}" for i in range(19)),
)


class LoraBlockWeightFlux(_SingleBlockSweepBase):
    SPEC = FLUX_SPEC


class LoraBlockWeightFluxCustom(_CustomSweepBase):
    SPEC = FLUX_SPEC


class LoraBlockWeightFluxBatch(_BatchSweepBase):
    SPEC = FLUX_SPEC


class LoraBlockWeightFluxGroup(_GroupSweepBase):
    SPEC = FLUX_SPEC


NODE_CLASS_MAPPINGS = {
    "LoraBlockWeightFlux": LoraBlockWeightFlux,
    "LoraBlockWeightFluxCustom": LoraBlockWeightFluxCustom,
    "LoraBlockWeightFluxBatch": LoraBlockWeightFluxBatch,
    "LoraBlockWeightFluxGroup": LoraBlockWeightFluxGroup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraBlockWeightFlux": "LoRA Block Weight (FLUX)",
    "LoraBlockWeightFluxCustom": "LoRA Block Weight Custom (FLUX)",
    "LoraBlockWeightFluxBatch": "LoRA Block Weight Batch (FLUX)",
    "LoraBlockWeightFluxGroup": "LoRA Block Weight Group (FLUX)",
}
