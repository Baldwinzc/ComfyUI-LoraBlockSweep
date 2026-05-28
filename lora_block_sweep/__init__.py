"""LoRA Block Sweep — per-block LoRA weighting for DiT-family models.

Assembles NODE mappings from each model adapter plus the model-agnostic
Save Grid node.
"""
from ._core import LoraBlockSweepSaveGrid
from . import _flux, _qwen, _sd35


NODE_CLASS_MAPPINGS = {
    **_flux.NODE_CLASS_MAPPINGS,
    **_qwen.NODE_CLASS_MAPPINGS,
    **_sd35.NODE_CLASS_MAPPINGS,
    "LoraBlockSweepSaveGrid": LoraBlockSweepSaveGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_flux.NODE_DISPLAY_NAME_MAPPINGS,
    **_qwen.NODE_DISPLAY_NAME_MAPPINGS,
    **_sd35.NODE_DISPLAY_NAME_MAPPINGS,
    "LoraBlockSweepSaveGrid": "LoRA Block Sweep Save Grid",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
