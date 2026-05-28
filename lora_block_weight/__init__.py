"""LoRA Block Weight — per-block LoRA weighting for DiT-family models.

Assembles NODE mappings from each model adapter plus the model-agnostic
Save Grid node.
"""
from ._core import LoraBlockWeightSaveGrid
from . import _flux, _qwen, _sd35


NODE_CLASS_MAPPINGS = {
    **_flux.NODE_CLASS_MAPPINGS,
    **_qwen.NODE_CLASS_MAPPINGS,
    **_sd35.NODE_CLASS_MAPPINGS,
    "LoraBlockWeightSaveGrid": LoraBlockWeightSaveGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_flux.NODE_DISPLAY_NAME_MAPPINGS,
    **_qwen.NODE_DISPLAY_NAME_MAPPINGS,
    **_sd35.NODE_DISPLAY_NAME_MAPPINGS,
    "LoraBlockWeightSaveGrid": "LoRA Block Weight Save Grid",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
