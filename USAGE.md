# Usage Guide

## Install

1. Find your ComfyUI install (the folder that has `custom_nodes/`)
2. Clone or copy this folder there:

   ```bash
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/Baldwinzc/ComfyUI-LoraBlockSweep.git
   ```

3. Restart ComfyUI

## Quickstart: knock-out sweep with Efficiency XY Plot

Drop in `LoRA Block Sweep (FLUX)` where you'd normally use `LoraLoader`:

```
UNETLoader      ──┐
                  ├──▶ LoRA Block Sweep (FLUX) ──▶ KSampler
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

If you don't want to install Efficiency Nodes, use **LoRA Block Sweep Batch
(FLUX)** instead. It internally loops over `(block, value)`, samples each, and
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

images out ─→ LoRA Block Sweep Save Grid   (labeled grid PNG)
           └→ SaveImage                    (also keeps individual cells)
```

`CLIP` input is intentionally absent — positive/negative are already encoded
upstream, so CLIP-side LoRA patches would have no effect. If you need
CLIP-side LoRA, use the regular Block Sweep node + Efficiency XY Plot.

### First-round recipe (recommended)

For a faster first pass, scan only the 19 double blocks:

```
D00,D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18
```

That's 19 × 5 = 95 images. After reviewing, expand to the 38 single blocks
in a second run if needed. (The README demo follows exactly this recipe.)

## Group sweeps (after narrowing down)

Use **LoRA Block Sweep Group (FLUX)** to sweep contiguous block ranges as a
single unit. Groups can be ranges (`D00-D06`), individual blocks (`S15`),
or mixed comma-lists (`D00-D03,S20`). Ranges may not cross D and S.

The default 8-group split is a naive even partition — three thirds of the
double blocks plus three thirds of the single blocks, then both whole halves
as anchors. **Treat it as a starting point and edit per LoRA once the
single-block sweep reveals where the action concentrates.**

## Fine-tune all 57 blocks

Use **LoRA Block Sweep Custom (FLUX)**. Paste a 57-value comma list in the
order:

    D00,D01,...,D18,S00,S01,...,S37

Example — keep all double blocks at full, taper late single blocks:

```
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0
```

## Tips

- **Fix the seed.** The grid is meaningless if the only variable isn't block weight.
- **Sweep at low res** (768×768, ~20 steps) to stay under an hour. Re-test the
  best settings at full resolution after.
- **285 images at 20–25 steps takes 1–3 h** on a single consumer GPU. Always
  start with the 19 double blocks (95 images, ~20 min on an H800) before
  committing to the full sweep.
- Wire the `info` STRING output to a `ShowText` node to see each cell's
  parameters in the UI.
