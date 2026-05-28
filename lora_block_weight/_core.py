"""Model-agnostic core for LoRA block sweeps.

Defines a `BlockSpec` describing one model family's transformer block layout
(name, count, key regex), plus base node classes that each model module
subclasses by attaching its own `SPEC`.

FLUX, Qwen-Image, and SD3.5 differ in their block layout but share the same
mechanics: parse a LoRA file, group its keys by transformer block, apply each
group with its own scalar strength via `model_patcher.add_patches`.
"""
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.sample
import comfy.samplers
import comfy.utils


@dataclass(frozen=True)
class BlockSpec:
    """Per-model block layout.

    `tag_groups` defines the per-prefix (\\d+) regex against ComfyUI
    state_dict keys. Each prefix letter (`D`, `S`, `B`, `J`, ...) maps to a
    (count, regex_pattern) pair. A key matching `regex` with group(1) == idx
    is classified as `f"{prefix}{idx:02d}"` provided idx < count.

    Block names are produced by `_build_block_names` in the order given by
    `tag_groups` insertion order. FLUX yields D00..D18,S00..S37; Qwen
    yields B00..B59; SD3.5 yields J00..J37.
    """

    model_key: str                          # "flux" | "qwen" | "sd35" — used in node id / category
    display_name: str                       # "FLUX" | "Qwen-Image" | "SD3.5" — user-facing
    tag_groups: dict                        # {"D": (19, r"diffusion_model\.double_blocks\.(\d+)\."), ...}
    default_groups: str                     # multiline string for Group node default
    default_first_round_blocks: str = ""    # comma list of blocks for SaveGrid default (empty -> all)

    # Computed
    block_names: tuple = field(init=False)
    block_names_set: frozenset = field(init=False)
    compiled_regexes: tuple = field(init=False)

    def __post_init__(self):
        names = []
        compiled = []
        for prefix, (count, pattern) in self.tag_groups.items():
            if len(prefix) != 1 or not prefix.isalpha():
                raise ValueError(f"tag prefix must be a single letter, got {prefix!r}")
            for i in range(count):
                names.append(f"{prefix}{i:02d}")
            compiled.append((prefix, count, re.compile(pattern)))
        object.__setattr__(self, "block_names", tuple(names))
        object.__setattr__(self, "block_names_set", frozenset(names))
        object.__setattr__(self, "compiled_regexes", tuple(compiled))
        if not self.default_first_round_blocks:
            object.__setattr__(self, "default_first_round_blocks", ",".join(names))


def classify_key(spec: BlockSpec, key) -> str:
    """Return block tag for a state_dict key, or 'extras' if unmatched.

    `comfy.lora.load_lora` returns a dict whose keys may be a plain string
    or a (key_name, offset, function) tuple; we normalize to the string form
    first (mirrors comfy/model_patcher.py add_patches handling).
    """
    key_str = key if isinstance(key, str) else key[0]
    for prefix, count, regex in spec.compiled_regexes:
        m = regex.search(key_str)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < count:
                return f"{prefix}{idx:02d}"
    return "extras"


_RE_TAG_GENERIC = re.compile(r"^([A-Z])(\d{1,3})$")


def parse_group(spec: BlockSpec, group_spec: str) -> list:
    """Parse 'D00-D06,S15-S20,D10' into a list of block tags.

    Ranges may not cross prefixes (e.g. 'D18-S00' is rejected). All resulting
    tags must exist in spec.block_names_set.
    """
    tags = []
    for raw in group_spec.split(","):
        part = raw.strip().upper()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ma, mb = _RE_TAG_GENERIC.match(a), _RE_TAG_GENERIC.match(b)
            if not (ma and mb):
                raise ValueError(f"Bad range '{part}' in group spec")
            if ma.group(1) != mb.group(1):
                raise ValueError(
                    f"Range '{part}' mixes {ma.group(1)} and {mb.group(1)} "
                    f"blocks (split into two)"
                )
            prefix = ma.group(1)
            start = int(ma.group(2))
            end = int(mb.group(2))
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                tags.append(f"{prefix}{i:02d}")
        else:
            if not _RE_TAG_GENERIC.match(part):
                raise ValueError(f"Bad block tag '{part}' in group spec")
            tags.append(part)
    unknown = [t for t in tags if t not in spec.block_names_set]
    if unknown:
        raise ValueError(
            f"Out-of-range tags in group spec: {unknown}. "
            f"Valid range: {spec.block_names[0]}..{spec.block_names[-1]}"
        )
    return tags


def build_block_strengths(spec: BlockSpec, target_block: str,
                          target_value: float, baseline_weight: float) -> dict:
    """{tag: strength} with target_block at target_value, others at baseline.

    `extras` (input/output layers) follows baseline_weight so the LoRA's
    embedding-side modifications stay consistent with the experiment baseline.
    """
    strengths = {tag: baseline_weight for tag in spec.block_names}
    strengths["extras"] = baseline_weight
    if target_block in strengths:
        strengths[target_block] = target_value
    return strengths


def build_group_strengths(spec: BlockSpec, group_tags: list,
                          group_value: float, baseline_weight: float) -> dict:
    """Same shape as build_block_strengths but for a list of tags forming one group."""
    strengths = {tag: baseline_weight for tag in spec.block_names}
    strengths["extras"] = baseline_weight
    for tag in group_tags:
        if tag in strengths:
            strengths[tag] = group_value
    return strengths


def apply_blockwise_patches(spec: BlockSpec, model_patcher,
                            loaded_patches: dict, block_strengths: dict,
                            debug: bool = False):
    """Group loaded patches by block tag, then `add_patches` once per group
    with the per-block strength. Empty-strength groups are skipped to avoid
    no-op work.
    """
    by_block = defaultdict(dict)
    for k, v in loaded_patches.items():
        tag = classify_key(spec, k)
        by_block[tag][k] = v

    if debug:
        print(f"[LBW {spec.model_key} debug] patches grouped by block tag:")
        for tag in sorted(by_block.keys()):
            count = len(by_block[tag])
            strength = block_strengths.get(tag, 0.0)
            sample_key = next(iter(by_block[tag]))
            sample_str = sample_key if isinstance(sample_key, str) else sample_key[0]
            print(f"  {tag:>8}: {count:>4} patches, strength={strength:.3f}, "
                  f"sample={sample_str[:80]}")

    applied_keys = set()
    for tag, patches in by_block.items():
        if not patches:
            continue
        strength = block_strengths.get(tag, 0.0)
        keys = model_patcher.add_patches(patches, strength)
        applied_keys.update(keys)
    return applied_keys


def load_lora_for_sweep(model, clip, lora_name):
    """Load a LoRA file and resolve its keys against the model + clip state dicts.
    Returns the loaded_patches dict ready for blockwise patching.
    """
    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)

    key_map = {}
    key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return comfy.lora.load_lora(lora_sd, key_map)


def sample_one(model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent, denoise):
    """Run one sampling pass. Mirrors nodes.py:common_ksampler but skips the
    UI preview callback (we have our own progress bar across the sweep).
    """
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    batch_inds = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask")

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, noise_mask=noise_mask,
        disable_pbar=True, seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return out


# Mode presets for the Group sweep: name -> (target_value, baseline_weight)
GROUP_MODES = {
    "knockout": (0.0, 1.0),  # group off, others on -> "what does dropping it do"
    "solo":     (1.0, 0.0),  # group on, others off -> "what does it alone do"
    "full":     (1.0, 1.0),  # everything on        -> "complete LoRA reference"
    "off":      (0.0, 0.0),  # everything off       -> "no LoRA reference"
}
DEFAULT_MODES = "knockout,solo,full"


def parse_modes(spec_str: str):
    """[(label, value, baseline), ...] from a comma list of mode names."""
    out = []
    for raw in spec_str.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in GROUP_MODES:
            valid = ",".join(GROUP_MODES.keys())
            raise ValueError(f"Unknown mode '{name}'. Valid: {valid}")
        value, baseline = GROUP_MODES[name]
        out.append((name, value, baseline))
    if not out:
        raise ValueError("modes list is empty")
    return out


# ---------------------------------------------------------------------------
# Base node classes — subclasses set SPEC, NAME, DISPLAY_NAME class attrs.
# ---------------------------------------------------------------------------


class _SweepBase:
    SPEC: BlockSpec = None  # subclass attaches
    CATEGORY = "LoraBlockWeight"


class _SingleBlockSweepBase(_SweepBase):
    """Single-block sweep. Pair with Efficiency Nodes XY Plot for grids."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = cls.SPEC
        valid_range = f"{spec.block_names[0]}..{spec.block_names[-1]}"
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "target_block": ("STRING",
                                 {"default": spec.block_names[0],
                                  "tooltip": f"Block tag: {valid_range}. "
                                             "Empty / unknown -> no target, all blocks at baseline_weight."}),
                "target_value": ("FLOAT",
                                 {"default": 1.0, "min": 0.0, "max": 2.0,
                                  "step": 0.05}),
                "baseline_weight": ("FLOAT",
                                    {"default": 1.0, "min": 0.0, "max": 2.0,
                                     "step": 0.05,
                                     "tooltip": "Knock-out: 1.0 (others stay full, target varies). "
                                                "Solo: 0.0 (others off, only target varies)."}),
                "clip_strength": ("FLOAT",
                                  {"default": 1.0, "min": -2.0, "max": 2.0,
                                   "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "apply"

    def apply(self, model, clip, lora_name, target_block, target_value,
              baseline_weight, clip_strength):
        spec = self.SPEC
        target_block = (target_block or "").strip().upper()
        loaded = load_lora_for_sweep(model, clip, lora_name)

        strengths = build_block_strengths(spec, target_block, target_value,
                                          baseline_weight)
        new_model = model.clone()
        applied = apply_blockwise_patches(spec, new_model, loaded, strengths)

        new_clip = clip.clone()
        clip_applied = new_clip.add_patches(loaded, clip_strength)
        applied.update(clip_applied)

        not_loaded = [k for k in loaded if k not in applied]
        info = (f"[{spec.model_key}] target={target_block}={target_value:.3f} "
                f"baseline={baseline_weight:.3f} clip={clip_strength:.3f} "
                f"patched={len(applied)} skipped={len(not_loaded)}")
        if not_loaded:
            first_skip = not_loaded[0]
            first_skip_str = first_skip if isinstance(first_skip, str) else first_skip[0]
            info += f" first_skip={first_skip_str}"
        return (new_model, new_clip, info)


class _CustomSweepBase(_SweepBase):
    """Manual per-block weights for fine tuning after the sweep narrows things down."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = cls.SPEC
        default_weights = ",".join(["1.0"] * len(spec.block_names))
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "weights": ("STRING",
                            {"default": default_weights, "multiline": True,
                             "tooltip": f"{len(spec.block_names)} comma-separated "
                                        f"weights in order: "
                                        f"{spec.block_names[0]},...,{spec.block_names[-1]}"}),
                "baseline_weight": ("FLOAT",
                                    {"default": 1.0, "min": 0.0, "max": 2.0,
                                     "step": 0.05}),
                "clip_strength": ("FLOAT",
                                  {"default": 1.0, "min": -2.0, "max": 2.0,
                                   "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "apply"

    def apply(self, model, clip, lora_name, weights, baseline_weight,
              clip_strength):
        spec = self.SPEC
        parts = [w.strip() for w in weights.split(",")]
        per_block = {}
        for i, tag in enumerate(spec.block_names):
            if i < len(parts):
                try:
                    per_block[tag] = float(parts[i])
                    continue
                except ValueError:
                    pass
            per_block[tag] = baseline_weight
        per_block["extras"] = baseline_weight

        loaded = load_lora_for_sweep(model, clip, lora_name)
        new_model = model.clone()
        applied = apply_blockwise_patches(spec, new_model, loaded, per_block)

        new_clip = clip.clone()
        clip_applied = new_clip.add_patches(loaded, clip_strength)
        applied.update(clip_applied)

        info = (f"[{spec.model_key}] custom weights, baseline={baseline_weight:.3f} "
                f"clip={clip_strength:.3f} patched={len(applied)}")
        return (new_model, new_clip, info)


class _BatchSweepBase(_SweepBase):
    """All-in-one sweep: for every (block, value) combination, patch + sample +
    decode into a batched IMAGE.

    CLIP-side LoRA modifications cannot take effect here because the
    positive/negative CONDITIONING is already encoded upstream. If you need
    the LoRA's text-encoder contribution, encode prompts AFTER a regular
    LoRA loader instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        spec = cls.SPEC
        default_blocks = ",".join(spec.block_names)
        valid_range = f"{spec.block_names[0]}..{spec.block_names[-1]}"
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                  "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                                 {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
                "block_list": ("STRING",
                               {"default": default_blocks, "multiline": True,
                                "tooltip": f"Comma-separated block tags. "
                                           f"Defaults to all {len(spec.block_names)} "
                                           f"({valid_range}). Trim for faster first round."}),
                "value_list": ("STRING",
                               {"default": "0,0.25,0.5,0.75,1.0",
                                "tooltip": "Comma-separated strength values."}),
                "baseline_weight": ("FLOAT",
                                    {"default": 1.0, "min": 0.0, "max": 2.0,
                                     "step": 0.05,
                                     "tooltip": "Knock-out: 1.0. Solo: 0.0."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "info", "blocks_used", "values_used")
    FUNCTION = "sweep"

    def sweep(self, model, vae, lora_name, positive, negative, latent_image,
              seed, steps, cfg, sampler_name, scheduler, denoise,
              block_list, value_list, baseline_weight):
        spec = self.SPEC
        blocks = [b.strip().upper() for b in block_list.split(",") if b.strip()]
        try:
            values = [float(v.strip()) for v in value_list.split(",") if v.strip()]
        except ValueError as e:
            raise ValueError(f"value_list must be comma-separated numbers, got: {e}")

        if not blocks:
            raise ValueError("block_list is empty")
        if not values:
            raise ValueError("value_list is empty")

        unknown = [b for b in blocks if b not in spec.block_names_set]
        if unknown:
            valid_range = f"{spec.block_names[0]}..{spec.block_names[-1]}"
            raise ValueError(f"Unknown block tags: {unknown}. Valid: {valid_range}")

        loaded = load_lora_for_sweep(model, None, lora_name)
        total = len(blocks) * len(values)
        pbar = comfy.utils.ProgressBar(total)
        all_images = []
        log = []

        print(f"[LBW {spec.model_key}] Loaded {len(loaded)} LoRA patches from {lora_name}")
        sample_keys = list(loaded.keys())[:3]
        for sk in sample_keys:
            sk_str = sk if isinstance(sk, str) else sk[0]
            print(f"[LBW {spec.model_key}]   sample loaded key: {sk_str}")

        first_iter = True
        for block in blocks:
            for value in values:
                strengths = build_block_strengths(spec, block, value, baseline_weight)
                new_model = model.clone()
                apply_blockwise_patches(spec, new_model, loaded, strengths,
                                        debug=first_iter)
                first_iter = False

                latent_out = sample_one(
                    new_model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image, denoise,
                )
                image = vae.decode(latent_out["samples"])
                if image.ndim == 5:
                    image = image.reshape(-1, image.shape[-3],
                                          image.shape[-2], image.shape[-1])
                all_images.append(image)
                log.append(f"{block}={value:.3f}")
                pbar.update(1)
                del new_model

        images_batch = torch.cat(all_images, dim=0)
        info = (f"[{spec.model_key}] sweep done: {total} images, "
                f"{len(blocks)} blocks x {len(values)} values, "
                f"baseline={baseline_weight:.3f}")
        blocks_used = ",".join(blocks)
        values_used = ",".join(f"{v:g}" for v in values)
        return (images_batch, info, blocks_used, values_used)


class _GroupSweepBase(_SweepBase):
    """Group sweep: each iteration treats a *range* of blocks as one unit.

    groups: one group per line, e.g.

        D00-D06
        S26-S37
        D00-D18,S00-S05      (commas combine ranges into one group)

    For each group the chosen blocks are set to value (looped over
    value_list); every other block is set to baseline_weight.
    """

    @classmethod
    def INPUT_TYPES(cls):
        spec = cls.SPEC
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                  "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                                 {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
                "groups": ("STRING",
                           {"default": spec.default_groups, "multiline": True,
                            "tooltip": "One group per line. Use D00-D06 for "
                                       "a range, comma to combine ranges "
                                       "(e.g. D00-D06,S20-S25). "
                                       "Default is a naive even split with "
                                       "no prior knowledge - edit per LoRA "
                                       "once a first sweep shows where the "
                                       "effect concentrates."}),
                "modes": ("STRING",
                          {"default": DEFAULT_MODES,
                           "tooltip": "Comma list of column types. "
                                      "knockout = group off, others on. "
                                      "solo = group on, others off. "
                                      "full = everything on (LoRA reference). "
                                      "off = everything off (no-LoRA "
                                      "reference). Each entry becomes one "
                                      "column in the output grid."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "info", "groups_used", "modes_used")
    FUNCTION = "sweep"

    def sweep(self, model, vae, lora_name, positive, negative, latent_image,
              seed, steps, cfg, sampler_name, scheduler, denoise,
              groups, modes):
        spec = self.SPEC
        group_lines = [g.strip() for g in groups.splitlines() if g.strip()]
        if not group_lines:
            raise ValueError("groups is empty")
        mode_specs = parse_modes(modes)

        group_tags = [parse_group(spec, g) for g in group_lines]
        loaded = load_lora_for_sweep(model, None, lora_name)
        total = len(group_lines) * len(mode_specs)
        pbar = comfy.utils.ProgressBar(total)
        all_images = []

        mode_summary = ", ".join(f"{n}(v={v:g},base={b:g})"
                                 for n, v, b in mode_specs)
        print(f"[LBW {spec.model_key} Group] {total} runs: {len(group_lines)} "
              f"groups x {len(mode_specs)} modes [{mode_summary}]")
        for i, (label, tags) in enumerate(zip(group_lines, group_tags)):
            print(f"[LBW {spec.model_key} Group]   group {i}: '{label}' -> {len(tags)} blocks")

        first_iter = True
        for label, tags in zip(group_lines, group_tags):
            for mode_name, value, baseline in mode_specs:
                strengths = build_group_strengths(spec, tags, value, baseline)
                new_model = model.clone()
                apply_blockwise_patches(spec, new_model, loaded, strengths,
                                        debug=first_iter)
                first_iter = False

                latent_out = sample_one(
                    new_model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image, denoise,
                )
                image = vae.decode(latent_out["samples"])
                if image.ndim == 5:
                    image = image.reshape(-1, image.shape[-3],
                                          image.shape[-2], image.shape[-1])
                all_images.append(image)
                pbar.update(1)
                del new_model

        images_batch = torch.cat(all_images, dim=0)
        info = (f"[{spec.model_key}] group sweep done: {total} images, "
                f"{len(group_lines)} groups x {len(mode_specs)} modes")
        groups_used = ",".join(group_lines)
        modes_used = ",".join(n for n, _, _ in mode_specs)
        return (images_batch, info, groups_used, modes_used)


# ---------------------------------------------------------------------------
# Model-agnostic Save Grid node — labels rows/cols from user-supplied strings.
# ---------------------------------------------------------------------------


def _load_font(size: int):
    """Pick the first available system font, fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE tensor (H,W,C) float [0,1] -> PIL RGB."""
    arr = (255.0 * image_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class LoraBlockWeightSaveGrid:
    """Compose a sweep IMAGE batch into a single labeled grid PNG.

    Cells stay at original resolution. Labels live in dedicated header
    columns / rows so nothing is occluded. The IMAGE batch is interpreted
    as row-major: outer loop = block_list, inner = value_list (matches the
    Batch sweep node's iteration order).
    """

    CATEGORY = "LoraBlockWeight"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blocks": ("STRING",
                           {"default": "", "multiline": True,
                            "tooltip": "Block tags in row order, must match "
                                       "the order used in the sweep."}),
                "values": ("STRING",
                           {"default": "0,0.5,1.0",
                            "tooltip": "Values in column order, must match "
                                       "the order used in the sweep."}),
                "filename_prefix": ("STRING", {"default": "lbw_grid"}),
                "label_size": ("INT", {"default": 36, "min": 8, "max": 256}),
                "pad": ("INT", {"default": 16, "min": 0, "max": 256}),
                "compress_level": ("INT", {"default": 4, "min": 0, "max": 9,
                                           "tooltip": "PNG deflate level. "
                                                      "PNG is always lossless; "
                                                      "0 = no compression (fastest, "
                                                      "biggest file), 9 = max."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_grid"
    OUTPUT_NODE = True

    def save_grid(self, images, blocks, values, filename_prefix,
                  label_size, pad, compress_level,
                  prompt=None, extra_pnginfo=None):
        block_list = [b.strip() for b in blocks.split(",") if b.strip()]
        value_list = [v.strip() for v in values.split(",") if v.strip()]
        rows = len(block_list)
        cols = len(value_list)
        expected = rows * cols

        if images.shape[0] != expected:
            raise ValueError(
                f"images batch size {images.shape[0]} does not match "
                f"{rows} blocks x {cols} values = {expected}"
            )

        cells = [_tensor_to_pil(images[i]) for i in range(expected)]
        cell_w = max(c.width for c in cells)
        cell_h = max(c.height for c in cells)

        font = _load_font(label_size)
        dummy = Image.new("RGB", (10, 10))
        dctx = ImageDraw.Draw(dummy)

        def text_size(s):
            bbox = dctx.textbbox((0, 0), s, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]

        def _fmt_col(v):
            try:
                float(v)
                return f"v={v}"
            except ValueError:
                return v

        col_label_strings = [_fmt_col(v) for v in value_list]
        row_label_strings = list(block_list)

        col_label_h = max(text_size(s)[1] for s in col_label_strings) + 2 * pad
        row_label_w = max(text_size(s)[0] for s in row_label_strings) + 2 * pad

        grid_w = row_label_w + cols * cell_w + (cols + 1) * pad
        grid_h = col_label_h + rows * cell_h + (rows + 1) * pad

        grid = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(grid)

        for col_idx, txt in enumerate(col_label_strings):
            x0 = row_label_w + pad + col_idx * (cell_w + pad)
            tw, th = text_size(txt)
            draw.text((x0 + (cell_w - tw) // 2, (col_label_h - th) // 2),
                      txt, fill="black", font=font)

        for row_idx, txt in enumerate(row_label_strings):
            y0 = col_label_h + pad + row_idx * (cell_h + pad)
            tw, th = text_size(txt)
            draw.text(((row_label_w - tw) // 2, y0 + (cell_h - th) // 2),
                      txt, fill="black", font=font)

            for col_idx in range(cols):
                cell = cells[row_idx * cols + col_idx]
                x = row_label_w + pad + col_idx * (cell_w + pad)
                grid.paste(cell, (x, y0))

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, grid_w, grid_h)

        out_name = f"{filename}_{counter:05}_.png"
        out_path = os.path.join(full_output_folder, out_name)
        grid.save(out_path, compress_level=compress_level)

        return {"ui": {"images": [{"filename": out_name,
                                    "subfolder": subfolder,
                                    "type": "output"}]}}
