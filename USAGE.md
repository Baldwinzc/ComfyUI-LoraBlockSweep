# Usage Guide

**English** | [简体中文](USAGE.zh-CN.md)

## Install

1. Find your ComfyUI install (the folder that has `custom_nodes/`)
2. Clone or copy this folder there:

   ```bash
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/Baldwinzc/ComfyUI-LoraBlockWeight.git
   ```

3. Restart ComfyUI

## Quickstart: knock-out sweep with Efficiency XY Plot

Drop in `LoRA Block Weight (FLUX.1)` where you'd normally use `LoraLoader`:

```
UNETLoader      ──┐
                  ├──▶ LoRA Block Weight (FLUX.1) ──▶ KSampler
DualCLIPLoader ──┘
                       lora_name = <your_lora>.safetensors
                       baseline_weight = 1.0
                       target_block    = (overridden by XY plot)
                       target_value    = (overridden by XY plot)
                       clip_strength   = 1.0
```

Wire **XY Plot** (from
[efficiency-nodes-comfyui](https://github.com/jags111/efficiency-nodes-comfyui)):

- **X axis** — `XY Input: String`, override `target_block`, paste:

   ```
   D00,D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18,S00,S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37
   ```

- **Y axis** — `XY Input: Number`, override `target_value`, values:
  `0,0.25,0.5,0.75,1.0`

Result: 57 × 5 = 285 image grid. Each cell shows the output when **one** block
is dialed to that strength while the other 56 stay at `baseline_weight = 1.0`.

### Reading the grid

- Column-wise nearly identical → block is non-critical, can be lowered freely
- Column-wise visible change → block carries weight, keep it high
- Column-wise improving as value drops → block introduces unwanted artifacts;
  consider keeping it permanently lower

## Solo sweep (optional second round)

Same setup, flip `baseline_weight = 0.0`. Now each cell shows what the target
block contributes **alone** (everything else off). Useful for understanding
what each block independently "knows".

## All-in-one: Batch node (no XY plot needed)

If you don't want to install Efficiency Nodes, use **LoRA Block Weight Batch
(FLUX.1)** instead. It internally loops over `(block, value)`, samples each, and
returns a batched IMAGE.

```
UNETLoader            ─→ model
VAELoader             ─→ vae
CLIPTextEncode        ─→ positive
ConditioningZeroOut   ─→ negative
EmptySD3LatentImage   ─→ latent_image
                         lora_name        = <your_lora>.safetensors
                         block_list       = D00,D01,...,D18,S00,...,S37  (or a subset)
                         value_list       = 0,0.25,0.5,0.75,1.0
                         baseline_weight  = 1.0  (knock-out)  |  0.0 (solo)
                         seed/steps/cfg/sampler/scheduler/denoise = same as KSampler

images out ─→ LoRA Block Weight Save Grid   (labeled grid PNG)
           └→ SaveImage                    (also keeps individual cells)
```

The same graph in ComfyUI:

![All-in-one Batch sweep graph in ComfyUI](docs/workflow_batch_flux.png)

> Drag [`example_workflows/flux_batch_sweep.json`](example_workflows/flux_batch_sweep.json)
> onto the canvas to load it, then swap in your own model / LoRA names.

`CLIP` input is intentionally absent — positive/negative are already encoded
upstream, so CLIP-side LoRA patches would have no effect. If you need
CLIP-side LoRA, use the regular Block Weight node + Efficiency XY Plot.

### First-round recipe (recommended)

For a faster first pass, scan only the 19 double blocks:

```
D00,D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18
```

That's 19 × 5 = 95 images. After reviewing, expand to the 38 single blocks
in a second run if needed. (The README demo follows exactly this recipe.)

## Group sweeps (after narrowing down)

Use **LoRA Block Weight Group (FLUX.1)** to sweep contiguous block ranges as a
single unit. Groups can be ranges (`D00-D06`), individual blocks (`S15`),
or mixed comma-lists (`D00-D03,S20`). Ranges may not cross D and S.

The default 8-group split is a naive even partition — three thirds of the
double blocks plus three thirds of the single blocks, then both whole halves
as anchors. **Treat it as a starting point and edit per LoRA once the
single-block sweep reveals where the action concentrates.**

## Fine-tune all 57 blocks

Use **LoRA Block Weight Custom (FLUX.1)**. Paste a 57-value comma list in the
order:

    D00,D01,...,D18,S00,S01,...,S37

Example — keep all double blocks at full, taper late single blocks:

```
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0
```

## Qwen-Image

Qwen-Image has 60 single-stream transformer blocks (tags `B00..B59`).
Every Qwen node mirrors the FLUX.1 node above — just swap `(FLUX.1)` for
`(Qwen-Image)` in the node name and use `B00..B59` instead of
`D00-D18,S00-S37`.

```
UNETLoader            ─→ model (qwen_image_fp8_e4m3fn.safetensors)
ModelSamplingAuraFlow ─→ model (shift = 4.0)
VAELoader             ─→ vae   (qwen_image_vae.safetensors)
CLIPLoader            ─→ clip  (qwen_2.5_vl_7b_fp8_scaled.safetensors, type=qwen_image)
CLIPTextEncode        ─→ positive / negative (empty)
EmptySD3LatentImage   ─→ latent_image
                         lora_name = <your_qwen_lora>.safetensors
                         block_list = B00,B01,...,B59  (or a subset)
                         value_list = 0,0.25,0.5,0.75,1.0
                         baseline_weight = 1.0 (knock-out) | 0.0 (solo)
                         seed/steps/cfg = workflow-appropriate
                         (Qwen-Image Lightning: 8 steps, cfg=1.0)

images out ─→ LoRA Block Weight Save Grid   (labeled grid PNG)
           └→ SaveImage                    (also keeps individual cells)
```

For an Efficiency XY Plot setup, paste the 60 Qwen block tags as the
X-axis values:

```
B00,B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,B33,B34,B35,B36,B37,B38,B39,B40,B41,B42,B43,B44,B45,B46,B47,B48,B49,B50,B51,B52,B53,B54,B55,B56,B57,B58,B59
```

### First-round recipe (Qwen-Image)

60 × 5 = 300 images is a lot. For a faster signal, pick every 5th block:

```
B00,B05,B10,B15,B20,B25,B30,B35,B40,B45,B50,B55
```

12 × 5 = 60 images. Once you see which neighbourhood the action is in,
zoom in with the Group node on contiguous ranges (e.g. `B20-B29`).

## SD3.5 Large

SD3.5 Large has 38 MMDiT joint blocks (tags `J00..J37`). Every SD3.5 node
mirrors the FLUX.1/Qwen node above — swap to `(SD3.5 Large)` and use
`J00..J37`.

```
CheckpointLoaderSimple ─→ model + vae (Stable Diffusion 3.5 Large.safetensors)
TripleCLIPLoader       ─→ clip  (clip_l + clip_g + t5xxl_fp16)
CLIPTextEncode         ─→ positive / negative
EmptySD3LatentImage    ─→ latent_image  (1024×1024 recommended)
                          lora_name = <your_sd35_lora>.safetensors
                          block_list = J00,J01,...,J37  (or a subset)
                          value_list = 0,0.25,0.5,0.75,1.0
                          baseline_weight = 1.0 (knock-out) | 0.0 (solo)
                          cfg = 4.5, sampler = euler, scheduler = sgm_uniform, steps = 20

images out ─→ LoRA Block Weight Save Grid   (labeled grid PNG)
           └→ SaveImage                    (also keeps individual cells)
```

For an Efficiency XY Plot setup, paste all 38 J tags as the X-axis values:

```
J00,J01,J02,J03,J04,J05,J06,J07,J08,J09,J10,J11,J12,J13,J14,J15,J16,J17,J18,J19,J20,J21,J22,J23,J24,J25,J26,J27,J28,J29,J30,J31,J32,J33,J34,J35,J36,J37
```

### First-round recipe (SD3.5)

38 × 5 = 190 images runs in ~50 min on an H-class GPU at 1024². For a
faster first pass, pick every 4th block:

```
J00,J04,J08,J12,J16,J20,J24,J28,J32,J36
```

10 × 5 = 50 images. After spotting the active neighbourhood, expand with
the Group node (e.g. `J20-J30`).

## Tips

- **Fix the seed.** The grid is meaningless if the only variable isn't block weight.
- **Sweep at low res** (768×768, ~20 steps) to stay under an hour. Re-test the
  best settings at full resolution after.
- **285 images at 20–25 steps takes 1–3 h** on a single consumer GPU. Always
  start with the 19 double blocks (95 images, ~20 min on an H800) before
  committing to the full sweep.
- Wire the `info` STRING output to a `ShowText` node to see each cell's
  parameters in the UI.
