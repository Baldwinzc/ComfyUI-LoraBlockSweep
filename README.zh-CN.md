# ComfyUI-LoraBlockSweep

[English](README.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-success)](https://github.com/comfyanonymous/ComfyUI)
[![Model: FLUX](https://img.shields.io/badge/model-FLUX-orange)](https://blackforestlabs.ai/)
[![Model: Qwen-Image](https://img.shields.io/badge/model-Qwen--Image-purple)](https://github.com/QwenLM/Qwen-Image)

为 DiT 系列模型提供按块（per-block）的 LoRA 权重控制 —— **FLUX**（19 个
double + 38 个 single = 57 块）和 **Qwen-Image**（60 个 transformer 块）。
独立设置每块强度，从标注网格里直接读出每块的影响。

![分组关掉对比：Full / Top-7 off / Bot-7 off / No LoRA](docs/hero_group.png)

> 相同的 LoRA、相同的 seed、相同的 prompt。关掉 **MSE 最高的 7 个块**
> （Top-7 off），LoRA 的插画风格几乎全部丢失 —— 接近无 LoRA 的基线。
> 关掉 **MSE 最低的 7 个块**（Bot-7 off），视觉上和 Full LoRA 没区别。
> 剩下 12 个块承担了大部分效果；那 7 个低 MSE 块是可以丢掉的"死重"，
> 去掉它们能换来更好的兼容性、更轻的叠加、更快的速度，且无视觉损失。
> **这个节点的作用就是帮你找到这个分界线。**

## 它做什么

大多数 LoRA 加载器只接受一个 `strength` 标量，对整个 adapter 生效。
SD1.5/SDXL 的按块加载器（LBW、Bobs Lora Loader）暴露的是 14 个左右的
概念块组。**本节点暴露的是 FLUX 真实的 57 个 transformer block**，
按块扫一遍就能拿到清晰的逐层信号 —— 哪些块重要、哪些是死重、哪些
可以调低而不丢风格。

可以配合 [Efficiency Nodes 的 XY Plot](https://github.com/jags111/efficiency-nodes-comfyui)
使用，或通过自带的 batch 节点独立运行,不需要外部编排工具。

## Demo:hero 图是怎么做出来的

Hero 图来自一个两阶段实验,LoRA 用的是
[`alvdansen/frosting_lane_flux`](https://huggingface.co/alvdansen/frosting_lane_flux),
一个风格化插画 LoRA。

**Stage 1 —— 按块扫描。** 把全部 19 个 double block 在权重
`{0, 0.25, 0.5, 0.75, 1.0}` 下扫一遍,其他块全部保持 `1.0`。共 95 张图。
对每个块计算 `weight=0`(关掉)和 `weight=1.0`(完整)两张结果之间的
MSE。MSE 越大,说明这个块对 LoRA 效果的贡献越大。

**Stage 2 —— 分组关掉。** 把 MSE 最高的 7 个块一起置零(Top-7 off);
把 MSE 最低的 7 个块一起置零(Bot-7 off)。和 Full LoRA、No LoRA 对比 ——
就是上面那张四联图。Stage 1 的 MSE 排序很干净地预测了 Stage 2 的视觉
效果:关掉顶部 7 个,风格丢失;关掉底部 7 个,毫无损失。

![按块影响力柱状图](docs/impact_chart_d.png)

|              | 块    | MSE       |
|--------------|-------|-----------|
| **关键**     | D00   | 0.00778   |
|              | D09   | 0.00694   |
|              | D15   | 0.00612   |
|              | D08   | 0.00592   |
|              | D02   | 0.00488   |
|              | D07   | 0.00470   |
|              | D03   | 0.00464   |
| **中等**     | D11   | 0.00456   |
|              | D06   | 0.00440   |
|              | D16   | 0.00430   |
|              | D05   | 0.00426   |
|              | D04   | 0.00417   |
|              | D12   | 0.00283   |
| **可忽略**   | D17   | 0.00221   |
|              | D18   | 0.00220   |
|              | D14   | 0.00218   |
|              | D13   | 0.00179   |
|              | D10   | 0.00091   |
|              | D01   | 0.00057   |

D00 → D01 之间 14× 的差距,正是 hero 的 Bot-7 off 看起来和 Full
没区别的原因:底部那些块几乎不贡献。完整的 19×5 标注网格见
[docs/grid_preview_d.png](docs/grid_preview_d.png)。

复现:
- [`_dev/full_sweep_D.json`](_dev/full_sweep_D.json) —— Stage 1 API workflow
- [`_dev/fetch_and_analyze.py`](_dev/fetch_and_analyze.py) —— 下载 + MSE 排名
- [`_dev/build_group_workflow.py`](_dev/build_group_workflow.py) —— 生成 Stage 2 workflow
- [`_dev/make_hero.py`](_dev/make_hero.py) —— 合成四联 hero 图

## 安装

**通过 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)(待收录后)**:
搜索 "LoraBlockSweep" → Install。

**手动安装:**

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/Baldwinzc/ComfyUI-LoraBlockSweep.git
```

重启 ComfyUI。

依赖(`numpy`、`Pillow`、`torch`)都已经被 ComfyUI 自带。

## 节点

| 节点 | 适用场景 |
|------|----------|
| **LoRA Block Sweep (FLUX)** / **(Qwen-Image)** | 可直接替换 `LoraLoader`,一次一个块 × 一个值。配 Efficiency XY Plot 跑网格扫描。 |
| **LoRA Block Sweep Batch (FLUX)** / **(Qwen-Image)** | 一体式:内部循环 `(block, value)`、逐个采样、返回 batched IMAGE。不需要 XY plot。上面 demo 用的就是 FLUX 版本。 |
| **LoRA Block Sweep Group (FLUX)** / **(Qwen-Image)** | 对**块组**扫描(比如 `D00-D06`、`B10-B19`),适合大致定位之后做细化。 |
| **LoRA Block Sweep Custom (FLUX)** / **(Qwen-Image)** | 终极调试:通过逗号分隔列表单独设置每一个块(FLUX 57 个,Qwen-Image 60 个)。 |
| **LoRA Block Sweep Save Grid** | 把 batched IMAGE 输出渲染成标注网格 PNG(Y 轴块名、X 轴权重)。模型无关。 |

块标签:
- **FLUX**:`D00..D18`(double-stream)+ `S00..S37`(single-stream)= 57 块
- **Qwen-Image**:`B00..B59` = 60 个 transformer 块

`baseline_weight` 决定实验模式:

- `1.0` → **Knock-out(剔除)**:其他块全部满强度,只动当前块。*"去掉这一块会损失什么?"*
- `0.0` → **Solo(独奏)**:其他块全部置零,只留当前块。*"这一块自己能贡献什么?"*

输入/输出层(`img_in` / `txt_in` / `time_in` / `vector_in` /
`guidance_in` / `final_layer`)始终跟随 `baseline_weight`,不会作为
扫描目标 —— 它们没有块索引。

详细 workflow 食谱见 [USAGE.md](USAGE.md)。

## 为什么要按模型出适配器

每个 DiT 模型的 transformer 块结构都不一样,为 SDXL U-Net 写的按块
LoRA 加载器映射不过来:

- **FLUX**:19 个 double-stream 块(`double_blocks.{N}`)+ 38 个
  single-stream 块(`single_blocks.{N}`)。标签 `D00..D18` 和 `S00..S37`。
- **Qwen-Image**:60 个 single-stream MMDiT 块(`transformer_blocks.{N}`),
  每块内部做图文联合注意力。标签 `B00..B59`。

本节点按各自模型的真实块索引,通过正则匹配 state_dict key 来分组 LoRA
权重,确保你设的强度对应模型实际跑的 transformer。新增一个 DiT 模型
只需写一个 `BlockSpec` 加 4 个薄子类 —— 模板见
`lora_block_sweep/_qwen.py`。

## 致谢 / 灵感来源

按块 LoRA 权重这项技术来自
[hako-mikan/sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)
(SD1.5/SDXL,A1111)。本节点把这个思路移植到 ComfyUI,并适配了 FLUX 的
transformer 结构。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。

## 贡献

欢迎通过 [Issues](https://github.com/Baldwinzc/ComfyUI-LoraBlockSweep/issues)
提 bug 报告和 LoRA 相关发现 —— 如果你扫了某个流行 LoRA 并找到有意思
的块排序,欢迎分享。
