# 使用指南

[English](USAGE.md) | **简体中文**

## 安装

1. 找到你的 ComfyUI 安装目录（含 `custom_nodes/` 的那个文件夹）
2. 把本仓库 clone 或复制进去：

   ```bash
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/Baldwinzc/ComfyUI-LoraBlockWeight.git
   ```

3. 重启 ComfyUI

## 快速开始：用 Efficiency XY Plot 跑 knock-out 扫描

把 `LoRA Block Weight (FLUX)` 接到原本 `LoraLoader` 的位置：

```
UNETLoader      ──┐
                  ├──▶ LoRA Block Weight (FLUX) ──▶ KSampler
DualCLIPLoader ──┘
                       lora_name = <your_lora>.safetensors
                       baseline_weight = 1.0
                       target_block    = （由 XY plot 覆盖）
                       target_value    = （由 XY plot 覆盖）
                       clip_strength   = 1.0
```

接 **XY Plot**（来自
[efficiency-nodes-comfyui](https://github.com/jags111/efficiency-nodes-comfyui)）：

- **X 轴** —— `XY Input: String`，覆盖 `target_block`，粘贴：

   ```
   D00,D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18,S00,S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37
   ```

- **Y 轴** —— `XY Input: Number`，覆盖 `target_value`，值：
  `0,0.25,0.5,0.75,1.0`

结果：57 × 5 = 285 张图的网格。每个格子展示**单块**被调到某个强度、
其他 56 块保持 `baseline_weight = 1.0` 时的输出。

### 怎么读这张网格

- 整列几乎没差异 → 这块不关键，可以放心调低
- 整列有明显变化 → 这块有分量，保持高权重
- 整列在权重下降时反而变好 → 这块引入了不想要的伪影，考虑长期保持低权重

## Solo 扫描（可选的第二轮）

同样的连法，把 `baseline_weight` 改成 `0.0`。这时每个格子展示的是该块
**单独**贡献什么（其他全关）。用于理解每块单独"知道"什么。

## 一体式：Batch 节点（不需要 XY plot）

如果你不想装 Efficiency Nodes，用 **LoRA Block Weight Batch (FLUX)** 即可。
它内部循环 `(block, value)`、逐个采样，返回 batched IMAGE。

```
UNETLoader            ─→ model
VAELoader             ─→ vae
CLIPTextEncode        ─→ positive
ConditioningZeroOut   ─→ negative
EmptySD3LatentImage   ─→ latent_image
                         lora_name        = <your_lora>.safetensors
                         block_list       = D00,D01,...,D18,S00,...,S37  （或子集）
                         value_list       = 0,0.25,0.5,0.75,1.0
                         baseline_weight  = 1.0  （knock-out）  |  0.0 （solo）
                         seed/steps/cfg/sampler/scheduler/denoise = 与 KSampler 一致

images out ─→ LoRA Block Weight Save Grid   （标注网格 PNG）
           └→ SaveImage                    （也保留单独的格子）
```

故意没有 `CLIP` 输入 —— positive/negative 已经在上游编码完了，CLIP 侧的
LoRA patch 不会生效。如果你需要 CLIP 侧 LoRA，用常规 Block Weight 节点
+ Efficiency XY Plot。

### 第一轮推荐配方

为了更快地拿到首轮信号，只扫描 19 个 double block：

```
D00,D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18
```

共 19 × 5 = 95 张图。看完之后再决定是否扩到 38 个 single block 跑第二
轮。（README 里的 demo 就是这个配方。）

## 块组扫描（在大致定位之后）

用 **LoRA Block Weight Group (FLUX)** 把连续的块范围作为一个整体扫描。
组可以是范围（`D00-D06`）、单块（`S15`）或混合逗号列表
（`D00-D03,S20`）。范围不能跨 D 和 S。

默认的 8 组划分是**朴素均分** —— double 块的三个三分之一加 single 块
的三个三分之一，再加两个完整的半段作为锚点。**把它当作起点，等单块
扫描看清效果集中在哪里之后，再按 LoRA 单独调整。**

## 精细微调全部 57 块

用 **LoRA Block Weight Custom (FLUX)**。按以下顺序粘贴 57 个值的逗号列表：

    D00,D01,...,D18,S00,S01,...,S37

示例 —— 保持所有 double 块满强度，后段 single 块逐步衰减：

```
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0
```

## Qwen-Image

Qwen-Image 有 60 个 single-stream transformer 块（标签 `B00..B59`）。
所有 Qwen 节点和 FLUX 节点一一对应 —— 只是把节点名里的 `(FLUX)` 换成
`(Qwen-Image)`，块标签从 `D00-D18,S00-S37` 换成 `B00..B59`。

```
UNETLoader            ─→ model (qwen_image_fp8_e4m3fn.safetensors)
ModelSamplingAuraFlow ─→ model (shift = 4.0)
VAELoader             ─→ vae   (qwen_image_vae.safetensors)
CLIPLoader            ─→ clip  (qwen_2.5_vl_7b_fp8_scaled.safetensors, type=qwen_image)
CLIPTextEncode        ─→ positive / negative (空字符串)
EmptySD3LatentImage   ─→ latent_image
                         lora_name = <your_qwen_lora>.safetensors
                         block_list = B00,B01,...,B59  (或子集)
                         value_list = 0,0.25,0.5,0.75,1.0
                         baseline_weight = 1.0 (knock-out) | 0.0 (solo)
                         seed/steps/cfg = 跟你工作流一致
                         (Qwen-Image Lightning: 8 steps, cfg=1.0)

images out ─→ LoRA Block Weight Save Grid   （标注网格 PNG）
           └→ SaveImage                    （也保留单独的格子）
```

配 Efficiency XY Plot 用的话，X 轴粘贴 60 个 Qwen 块标签：

```
B00,B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,B33,B34,B35,B36,B37,B38,B39,B40,B41,B42,B43,B44,B45,B46,B47,B48,B49,B50,B51,B52,B53,B54,B55,B56,B57,B58,B59
```

### 第一轮推荐配方（Qwen-Image）

60 × 5 = 300 张图比较多。为了更快拿到首轮信号，每 5 块取一个：

```
B00,B05,B10,B15,B20,B25,B30,B35,B40,B45,B50,B55
```

12 × 5 = 60 张图。看出 action 集中在哪一段之后，用 Group 节点对那段
连续范围（如 `B20-B29`）做细化。

## SD3.5 Large

SD3.5 Large 有 38 个 MMDiT joint block（标签 `J00..J37`）。所有 SD3.5
节点和上面 FLUX/Qwen 版本一一对应 —— 节点名换成 `(SD3.5 Large)`，块
标签用 `J00..J37` 即可。

```
CheckpointLoaderSimple ─→ model + vae（Stable Diffusion 3.5 Large.safetensors）
TripleCLIPLoader       ─→ clip（clip_l + clip_g + t5xxl_fp16）
CLIPTextEncode         ─→ positive / negative
EmptySD3LatentImage    ─→ latent_image（推荐 1024×1024）
                          lora_name = <your_sd35_lora>.safetensors
                          block_list = J00,J01,...,J37（或子集）
                          value_list = 0,0.25,0.5,0.75,1.0
                          baseline_weight = 1.0 (knock-out) | 0.0 (solo)
                          cfg = 4.5, sampler = euler, scheduler = sgm_uniform, steps = 20

images out ─→ LoRA Block Weight Save Grid   （标注网格 PNG）
           └→ SaveImage                    （也保留单独的格子）
```

配 Efficiency XY Plot 用的话，X 轴粘贴 38 个 J 标签：

```
J00,J01,J02,J03,J04,J05,J06,J07,J08,J09,J10,J11,J12,J13,J14,J15,J16,J17,J18,J19,J20,J21,J22,J23,J24,J25,J26,J27,J28,J29,J30,J31,J32,J33,J34,J35,J36,J37
```

### 第一轮推荐配方（SD3.5）

38 × 5 = 190 张图在 H 卡 1024² 下大约 50 分钟。想更快拿到首轮信号，
每 4 块取一个：

```
J00,J04,J08,J12,J16,J20,J24,J28,J32,J36
```

10 × 5 = 50 张图。看出 action 在哪一段之后，用 Group 节点对那段
（如 `J20-J30`）做细化。

## 小贴士

- **固定 seed。** 如果唯一变量不是块权重，这张网格就没意义了。
- **低分辨率扫描**（768×768，约 20 步）以保证总时长在一小时内。找到
  最佳设置后再在完整分辨率下复测。
- **285 张图在 20–25 步下需要 1–3 小时**（单卡消费级 GPU）。务必先跑
  19 个 double 块（95 张、H800 上约 20 分钟）再决定要不要做完整扫描。
- 把 `info` STRING 输出接到 `ShowText` 节点，可以在 UI 里直接看每个
  格子的参数。
