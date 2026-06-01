try:
    # Normal ComfyUI load: this dir is imported as a package.
    from .lora_block_weight import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Imported standalone (e.g. by a test collector) with no parent package.
    from lora_block_weight import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
