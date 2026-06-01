"""Repo-root pytest setup.

Loaded before collection, so it runs before pytest imports the repo-root
`__init__.py` (ComfyUI's entry point) as a Package. Two jobs:

1. Put the repo root on sys.path so `import lora_block_weight...` resolves.
2. Stub the modules `_core.py` imports that only exist inside a ComfyUI install
   (folder_paths, comfy.*) or are heavy optional deps (numpy, torch, PIL). The
   pure logic under test never calls into them, so empty stubs are enough —
   except for the few names referenced in type annotations, which we provide.

This keeps the smoke tests dependency-free and fast in CI.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_stub(name, configure=None):
    """Install a stub module under `name` if it can't be imported for real."""
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        if configure:
            configure(mod)
        return mod


def _cfg_folder_paths(m):
    m.get_filename_list = lambda *a, **k: []
    m.get_full_path = lambda *a, **k: ""
    m.get_output_directory = lambda *a, **k: "."
    m.get_save_image_path = lambda *a, **k: (".", "x", 0, "", "")


_ensure_stub("folder_paths", _cfg_folder_paths)

_ensure_stub("comfy")
for sub in ("lora", "lora_convert", "sample", "utils"):
    _ensure_stub(f"comfy.{sub}")
    setattr(sys.modules["comfy"], sub, sys.modules[f"comfy.{sub}"])


def _cfg_samplers(m):
    class _KSampler:
        SAMPLERS = ["euler"]
        SCHEDULERS = ["simple"]
    m.KSampler = _KSampler


_ensure_stub("comfy.samplers", _cfg_samplers)
sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]

# Heavy deps: real if installed, else a bare stub. `_core.py` references
# `torch.Tensor` / `Image.Image` in annotations (evaluated at def time).
_ensure_stub("numpy")
_ensure_stub("torch", lambda m: setattr(m, "Tensor", object))


def _cfg_pil(m):
    m.Image = types.SimpleNamespace(Image=object)
    m.ImageDraw = types.SimpleNamespace()
    m.ImageFont = types.SimpleNamespace()


_ensure_stub("PIL", _cfg_pil)

# Keep the repo-root __init__.py (ComfyUI entry point) out of collection.
collect_ignore = ["__init__.py"]
