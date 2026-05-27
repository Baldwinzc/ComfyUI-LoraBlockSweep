# ComfyUI-LoraBlockSweep

**English** | [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-success)](https://github.com/comfyanonymous/ComfyUI)
[![Model: FLUX](https://img.shields.io/badge/model-FLUX-orange)](https://blackforestlabs.ai/)

Per-block LoRA weighting for **FLUX**. Scan every transformer block
(19 double + 38 single = 57), sweep their strengths, and read the impact
of each block straight off a labeled grid.

![Group knockout: Full / Top-7 off / Bot-7 off / No LoRA](docs/hero_group.png)

> Same LoRA, same seed, same prompt. Knocking out the **7 highest-MSE blocks**
> (Top-7 off) strips most of the LoRA's illustration style — the result is
> close to the No-LoRA baseline. Knocking out the **7 lowest-MSE blocks**
> (Bot-7 off) is visually indistinguishable from Full. The other 12 blocks
> apparently do most of the work; those 7 are dead weight you can drop for
> compatibility / stacking / speed gains, with no visual cost. **This node
> exists to find that split.**

## What it does

Most LoRA loaders take a single `strength` scalar that applies to the whole
adapter. Block-wise loaders for SD1.5/SDXL (LBW, Bobs Lora Loader) expose ~14
conceptual block groups. **This exposes FLUX's actual 57 transformer blocks**,
so a single-block sweep gives you a clean per-layer signal — which blocks
matter, which are dead weight, which you can dial down without losing the
style.

Pairs with [Efficiency Nodes' XY Plot](https://github.com/jags111/efficiency-nodes-comfyui)
or runs standalone via the batch node — no external orchestration required.

## Demo: how the hero was produced

The hero is built from a two-stage experiment with
[`alvdansen/frosting_lane_flux`](https://huggingface.co/alvdansen/frosting_lane_flux),
a stylized illustration LoRA.

**Stage 1 — per-block sweep.** Sweep all 19 double blocks at weights
`{0, 0.25, 0.5, 0.75, 1.0}` while every other block stays at `1.0`. That's
95 images. For each block I compute MSE between the `weight=0` (knockout)
and `weight=1.0` (full) result. The higher the MSE, the more that block
carries the LoRA's effect.

**Stage 2 — group knockout.** Take the 7 highest-MSE blocks and zero them
together (Top-7 off); take the 7 lowest-MSE blocks and zero them together
(Bot-7 off). Compare against Full LoRA and No LoRA — the four-up image
above. The MSE ranking from Stage 1 turns out to predict the visual outcome
of Stage 2 cleanly: kill the top 7, lose the style; kill the bottom 7, lose
nothing.

![Per-block impact bar chart](docs/impact_chart_d.png)

|         | Block | MSE       |
|---------|-------|-----------|
| **Critical**  | D00   | 0.00778  |
|         | D09   | 0.00694  |
|         | D15   | 0.00612  |
|         | D08   | 0.00592  |
|         | D02   | 0.00488  |
|         | D07   | 0.00470  |
|         | D03   | 0.00464  |
| **Mid**       | D11   | 0.00456  |
|         | D06   | 0.00440  |
|         | D16   | 0.00430  |
|         | D05   | 0.00426  |
|         | D04   | 0.00417  |
|         | D12   | 0.00283  |
| **Negligible** | D17   | 0.00221  |
|         | D18   | 0.00220  |
|         | D14   | 0.00218  |
|         | D13   | 0.00179  |
|         | D10   | 0.00091  |
|         | D01   | 0.00057  |

The 14× gap from D00 → D01 explains why the hero's Bot-7 off panel looks
identical to Full: those bottom blocks barely contribute. The full 19×5
labeled grid lives in [docs/grid_preview_d.png](docs/grid_preview_d.png).

Reproduce:
- [`_dev/full_sweep_D.json`](_dev/full_sweep_D.json) — Stage 1 API workflow
- [`_dev/fetch_and_analyze.py`](_dev/fetch_and_analyze.py) — downloader + MSE ranking
- [`_dev/build_group_workflow.py`](_dev/build_group_workflow.py) — generate the Stage 2 workflow
- [`_dev/make_hero.py`](_dev/make_hero.py) — compose the 4-up hero

## Install

**Via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)** (when listed): search "LoraBlockSweep" → Install.

**Manual:**

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/Baldwinzc/ComfyUI-LoraBlockSweep.git
```

Restart ComfyUI.

Dependencies (`numpy`, `Pillow`, `torch`) are already pulled in by ComfyUI.

## Nodes

| Node | Use when |
|------|----------|
| **LoRA Block Sweep (FLUX)** | Drop-in `LoraLoader` replacement, one block × one value. Wire to Efficiency XY Plot for grid sweeps. |
| **LoRA Block Sweep Batch (FLUX)** | All-in-one: loops over `(block, value)` internally, samples each, returns a batched IMAGE. No XY plot needed. Used in the demo above. |
| **LoRA Block Sweep Group (FLUX)** | Sweep *grouped* blocks (e.g. `D00-D06`, `S15-S20`) once you've narrowed down where the action is. |
| **LoRA Block Sweep Custom (FLUX)** | Final pass: set all 57 blocks individually via a comma-separated list. |
| **LoRA Block Sweep Save Grid** | Renders the batched IMAGE output into a labeled grid PNG (block names on Y axis, weights on X axis). |

`baseline_weight` flips the experiment mode:

- `1.0` → **Knock-out**: every other block at full; target varies. *"What breaks when this block is removed?"*
- `0.0` → **Solo**: every other block at zero; target alone. *"What does this block contribute by itself?"*

Input/output layers (`img_in` / `txt_in` / `time_in` / `vector_in` /
`guidance_in` / `final_layer`) always follow `baseline_weight` and are never
sweep targets — they don't have block indices.

See [USAGE.md](USAGE.md) for full workflow recipes.

## Why FLUX-specific?

FLUX's transformer is structurally different from SDXL: 19 double-stream
blocks then 38 single-stream blocks. Block-wise LoRA loaders built for SDXL's
U-Net layout don't map cleanly. This node groups LoRA keys by the actual
FLUX block index using regex on `diffusion_model.double_blocks.{N}.` /
`diffusion_model.single_blocks.{N}.` so the weights you set match the
transformer the model actually runs.

## Citation / inspiration

Block-wise LoRA weighting as a technique comes from
[hako-mikan/sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)
(SD1.5/SDXL, A1111). This node ports the idea to ComfyUI and adapts it to
FLUX's transformer layout.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Bug reports and LoRA-specific findings welcome via
[Issues](https://github.com/Baldwinzc/ComfyUI-LoraBlockSweep/issues) — if you
sweep a popular LoRA and find an interesting block ranking, share it.
