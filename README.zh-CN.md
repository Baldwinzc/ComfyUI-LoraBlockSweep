# ComfyUI-LoraBlockWeight

[English](README.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-success)](https://github.com/comfyanonymous/ComfyUI)
[![Model: FLUX.1](https://img.shields.io/badge/model-FLUX.1-orange)](https://blackforestlabs.ai/)
[![Model: Qwen-Image](https://img.shields.io/badge/model-Qwen--Image-purple)](https://github.com/QwenLM/Qwen-Image)
[![Model: SD3.5](https://img.shields.io/badge/model-SD3.5%20Large-blue)](https://stability.ai/news/introducing-stable-diffusion-3-5)

**LoRA 太霸道？盖过 prompt、把风格糊到不该糊的地方、和别的 LoRA 打架？**
按块权重让你保留有效果的那几块、把作乱的那几块归零 —— 但你得先知道
*哪些块*是哪些。本节点就是干这个的。

为 DiT 系列模型提供按块（per-block）的 LoRA 权重控制 —— **FLUX.1**（19 个
double + 38 个 single = 57 块）、**Qwen-Image**（60 个 transformer 块）和
**SD3.5 Large**（38 个 joint block）。独立设置每块强度，从标注网格里直接
读出每块的影响。

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
概念块组。**本节点暴露的是 DiT 真实的 transformer block**（FLUX.1 57 个、
Qwen-Image 60 个、SD3.5 Large 38 个），按块扫一遍就能拿到清晰的逐层
信号 —— 哪些块重要、哪些近乎死重、哪些可以调低而不丢风格。

可以配合 [Efficiency Nodes 的 XY Plot](https://github.com/jags111/efficiency-nodes-comfyui)
使用，或通过自带的 batch 节点独立运行,不需要外部编排工具。

![ComfyUI 里的一体化 Batch 扫描连法](docs/workflow_batch_flux.png)

一体化连法:加载 model / CLIP / VAE,编码 prompt,然后全部接进 **Batch**
节点(它内部自动循环每个 `(block, value)`),再用 **Save Grid** 渲染出网格 ——
全程不需要 XY Plot。把
[`example_workflows/flux_batch_sweep.json`](example_workflows/flux_batch_sweep.json)
拖进 ComfyUI 画布即可加载这套现成连法。

## Demo 1 —— FLUX.1 + [Frosting Lane](https://huggingface.co/alvdansen/frosting_lane_flux)

最上面那张 hero 图来自一个两阶段实验。

**Stage 1 —— 按块扫描。** 把全部 19 个 double block 在权重
`{0, 0.25, 0.5, 0.75, 1.0}` 下扫一遍,其他块全部保持 `1.0`。共 95 张图。
对每个块计算 `weight=0`(关掉)和 `weight=1.0`(完整)两张结果之间的
MSE。MSE 越大,说明这个块对 LoRA 效果的贡献越大。

**Stage 2 —— 分组关掉。** 把 MSE 最高的 7 个块一起置零(Top-7 off);
把 MSE 最低的 7 个块一起置零(Bot-7 off)。和 Full LoRA、No LoRA 对比 ——
就是上面那张四联图。Stage 1 的 MSE 排序很干净地预测了 Stage 2 的视觉
效果:关掉顶部 7 个,风格丢失;关掉底部 7 个,毫无损失。

![FLUX.1 按块影响力柱状图](docs/impact_chart_d.png)

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
- `python _dev/fetch_and_analyze.py <prompt_id>` —— 下载 + MSE 排名
- [`_dev/build_group_workflow.py`](_dev/build_group_workflow.py) —— 生成 Stage 2 workflow
- `python _dev/make_hero.py <prompt_id>` —— 合成四联 hero 图

## Demo 2 —— Qwen-Image + [Modern Anime](https://huggingface.co/alfredplpl/qwen-image-modern-anime-lora)

同样的实验流程换到 Qwen-Image + 另一个 LoRA,证明这个方法不是 FLUX.1 专属。

![Qwen-Image 分组关掉对比:Full / Top-12 off / Bot-12 off / No LoRA](docs/hero_group_qwen.png)

> Qwen-Image 有 60 个 transformer 块。关掉 **MSE 最高的 12 个块**,
> 这个 modern-anime LoRA 的插画风格被剥离,结果往写实方向回落。关掉
> **MSE 最低的 12 个块**,视觉上和 Full LoRA 没区别。块数更多,LoRA
> 信号也分得更散:**Top/Bot MSE 比 = 24×**,而 FLUX.1 是 14×。

**Stage 1。** 把全部 60 个 transformer 块在权重
`{0, 0.25, 0.5, 0.75, 1.0}` 下扫一遍,其他块全部保持 `1.0`。
共 300 张图,768×768,fp8。对每块算 knockout vs full 的 MSE 并排序。

**Stage 2。** 把 MSE 最高的 12 个块一起置零、最低的 12 个一起置零,
和 Full LoRA、No LoRA 对比。相同 prompt、相同 seed。

![Qwen-Image 按块影响力柱状图](docs/impact_chart_b.png)

|                | 块    | MSE       |
|----------------|-------|-----------|
| **关键**       | B29   | 0.00586   |
|                | B28   | 0.00476   |
|                | B38   | 0.00430   |
|                | B31   | 0.00398   |
|                | B18   | 0.00395   |
|                | B30   | 0.00353   |
|                | B16   | 0.00301   |
|                | B37   | 0.00286   |
|                | B15   | 0.00229   |
|                | B19   | 0.00229   |
|                | B00   | 0.00223   |
|                | B34   | 0.00211   |
| **可忽略**     | B11   | 0.00066   |
|                | B25   | 0.00062   |
|                | B21   | 0.00061   |
|                | B09   | 0.00057   |
|                | B12   | 0.00056   |
|                | B10   | 0.00055   |
|                | B56   | 0.00045   |
|                | B24   | 0.00041   |
|                | B05   | 0.00031   |
|                | B06   | 0.00028   |
|                | B07   | 0.00025   |
|                | B50   | 0.00024   |

完整 60 行排名:[docs/impact_ranking_b.txt](docs/impact_ranking_b.txt)。
完整 60×5 标注网格(约 950 KB JPEG):[docs/grid_preview_b.jpg](docs/grid_preview_b.jpg)。

复现(Qwen 变体):
- [`_dev/full_sweep_B.json`](_dev/full_sweep_B.json) —— Stage 1 API workflow
- `python _dev/fetch_and_analyze.py <prompt_id> --model qwen` —— 下载 + MSE 排名
- [`_dev/build_group_workflow_qwen.py`](_dev/build_group_workflow_qwen.py) —— 生成 Stage 2 workflow
- `python _dev/make_hero.py <prompt_id> --model qwen` —— 合成四联 hero 图

## Demo 3 —— SD3.5 Large + [Anime LoRA](https://huggingface.co/prithivMLmods/SD3.5-Large-Anime-LoRA)

第三个 DiT、第三种架构 —— SD3.5 Large 的 38 个 MMDiT `joint_blocks`，每块
内部分成 `context_block`（文本）与 `x_block`（图像）两半，共享联合注意力。

![SD3.5 Large 分组关掉对比:Full / Top-12 off / Bot-12 off / No LoRA](docs/hero_group_sd35.png)

> 同样的实验。关掉 **MSE 最低的 12 个块**（Bot-12 off），视觉上和 Full
> LoRA 没区别 —— 12 个死重块确认。关掉 **MSE 最高的 12 个块**（Top-12
> off），结果明显变了 —— 构图坍缩（户外云景退化为纯背景）、风格变柔，
> 但**没有**完全回退到 No-LoRA 的样子。SD3.5 的 joint block 比 FLUX.1 的
> double-stream 更冗余地承载 LoRA 信号：只关掉头部一组还不足以剥离风格。
> **Top/Bot MSE 比 = 47×**，是三个 demo 里差距最大的。

**Stage 1。** 把全部 38 个 joint block 在权重 `{0, 0.25, 0.5, 0.75, 1.0}`
下扫一遍，其他块全部保持 `1.0`。共 190 张图，1024×1024。对每块算
knockout vs full 的 MSE 并排序。

**Stage 2。** 把 MSE 最高的 12 个块一起置零、最低的 12 个一起置零，和
Full LoRA、No LoRA 对比。相同 prompt、相同 seed。

![SD3.5 Large 按块影响力柱状图](docs/impact_chart_j.png)

|                | 块    | MSE      |
|----------------|-------|----------|
| **关键**       | J07   | 0.00952  |
|                | J24   | 0.00666  |
|                | J26   | 0.00634  |
|                | J21   | 0.00622  |
|                | J30   | 0.00615  |
|                | J22   | 0.00609  |
|                | J00   | 0.00568  |
|                | J20   | 0.00551  |
|                | J25   | 0.00546  |
|                | J09   | 0.00484  |
|                | J01   | 0.00479  |
|                | J19   | 0.00476  |
| **可忽略**     | J37   | 0.00242  |
|                | J02   | 0.00234  |
|                | J34   | 0.00233  |
|                | J12   | 0.00211  |
|                | J14   | 0.00185  |
|                | J06   | 0.00123  |
|                | J13   | 0.00115  |
|                | J36   | 0.00096  |
|                | J08   | 0.00094  |
|                | J05   | 0.00059  |
|                | J35   | 0.00040  |
|                | J10   | 0.00021  |

完整 38 行排名：[docs/impact_ranking_j.txt](docs/impact_ranking_j.txt)。
完整 38×5 标注网格（约 950 KB JPEG）：[docs/grid_preview_j.jpg](docs/grid_preview_j.jpg)。

复现（SD3.5 变体）：
- [`_dev/full_sweep_J.json`](_dev/full_sweep_J.json) —— Stage 1 API workflow
- `python _dev/fetch_and_analyze.py <prompt_id> --model sd35` —— 下载 + MSE 排名
- [`_dev/build_group_workflow_sd35.py`](_dev/build_group_workflow_sd35.py) —— 生成 Stage 2 workflow
- `python _dev/make_hero.py <prompt_id> --model sd35` —— 合成四联 hero 图

## 推荐工作流

三轮下来，一个 LoRA 就从"一个全局 strength 标量"变成"按块调过的配方"：

1. **全扫 + 排序。** 用 Batch 节点（或 XY plot）做一次完整扫描，
   `baseline_weight = 1.0`。把结果喂给 `_dev/fetch_and_analyze.py`
   按 MSE 排每一块。这一步你就拿到了上面 demo 里那种
   **关键 / 中等 / 可忽略** 的三档划分。

2. **去掉死重。** 打开 Custom 节点，把每个"可忽略"块设成 `0`。
   这些块本来就几乎不贡献什么 —— 归零它们能减少和其他 LoRA 叠加时的
   干扰、给 prompt 让出表达空间、轻微降低推理成本。风险最低、ROI 最高
   的一刀。

3. **调中间。** "关键"块保持 `1.0`（或你常用的全局 strength）。
   "中等"块就是可调旋钮 —— 在 Custom 节点里把它们设成 `0.5` 跑一遍和
   Full 对比。如果 LoRA 还在往不该出现的地方糊风格，继续把中等块往下
   压。如果你想要更强的 LoRA 味道，把中等块推过 `1.0`。

只想快速看一眼，可以跳过第 1 步的完整扫描，直接用 [USAGE.md](USAGE.md)
里每个模型的"第一轮推荐配方" —— 稀疏取 10–12 个块就能定位活跃段，
50–95 张图就够，不用跑满 190–300。

> **排名要当成"本次实验专属"来读。** 这里的 MSE 是**单个** prompt、单个
> seed、单个分辨率、单个采样器下的像素级差异 —— 是视觉影响的近似代理，
> 不是某块"普适价值"的定论。某块这一轮排进"可忽略"，不代表它在所有场景
> 都没用;把某块永久置零之前，建议换几个 prompt 复测一下。

## 什么时候别用它

- **只是想让 LoRA 在满强度下好看** —— 用普通 LoRA 加载器就行，这是诊断
  显微镜，不是一键增强。
- **时间 / 显存紧张** —— 完整扫描是 190–300 张图。先用 [USAGE.md](USAGE.md)
  里的稀疏首轮配方，或者干脆别扫。
- **你需要 LoRA 的 text-encoder（CLIP 侧）效果** —— Batch 节点只 patch
  UNet/transformer（prompt 在上游就编码好了）。要 CLIP 侧 LoRA，请用普通
  加载器 + 单块节点配 XY Plot。
- **LoRA 很轻、本来就乖** —— 按块手术对"盖过 prompt、糊错风格、叠加打架"
  的 LoRA 才划算，对本身就规矩的 LoRA 收益有限。

## 安装

**通过 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)(待收录后)**:
搜索 "LoraBlockWeight" → Install。

**手动安装:**

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/Baldwinzc/ComfyUI-LoraBlockWeight.git
```

重启 ComfyUI。

依赖(`numpy`、`Pillow`、`torch`)都已经被 ComfyUI 自带。

兼容当前版本的 ComfyUI —— 能加载 FLUX.1 / Qwen-Image / SD3.5 的那些版本
即可。它依赖 `comfy.lora` / `comfy.sample` 这些相对稳定的内部接口;万一以后
ComfyUI 重构导致加载失败，欢迎
[提 issue](https://github.com/Baldwinzc/ComfyUI-LoraBlockWeight/issues)。

## 节点

| 节点 | 适用场景 |
|------|----------|
| **LoRA Block Weight (FLUX.1)** / **(Qwen-Image)** / **(SD3.5 Large)** | 可直接替换 `LoraLoader`,一次一个块 × 一个值。配 Efficiency XY Plot 跑网格扫描。 |
| **LoRA Block Weight Batch (FLUX.1)** / **(Qwen-Image)** / **(SD3.5 Large)** | 一体式:内部循环 `(block, value)`、逐个采样、返回 batched IMAGE。不需要 XY plot。上面 demo 用的就是这一类节点。 |
| **LoRA Block Weight Group (FLUX.1)** / **(Qwen-Image)** / **(SD3.5 Large)** | 对**块组**扫描(比如 `D00-D06`、`B10-B19`、`J00-J09`),适合大致定位之后做细化。 |
| **LoRA Block Weight Custom (FLUX.1)** / **(Qwen-Image)** / **(SD3.5 Large)** | 终极调试:通过逗号分隔列表单独设置每一个块(FLUX.1 57 个,Qwen-Image 60 个,SD3.5 38 个)。 |
| **LoRA Block Weight Save Grid** | 把 batched IMAGE 输出渲染成标注网格 PNG(Y 轴块名、X 轴权重)。模型无关。 |

块标签:
- **FLUX.1**:`D00..D18`(double-stream)+ `S00..S37`(single-stream)= 57 块
- **Qwen-Image**:`B00..B59` = 60 个 transformer 块
- **SD3.5 Large**:`J00..J37` = 38 个 joint block(MMDiT 内含 context + image 两半)

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

- **FLUX.1**:19 个 double-stream 块(`double_blocks.{N}`)+ 38 个
  single-stream 块(`single_blocks.{N}`)。标签 `D00..D18` 和 `S00..S37`。
- **Qwen-Image**:60 个 single-stream MMDiT 块(`transformer_blocks.{N}`),
  每块内部做图文联合注意力。标签 `B00..B59`。
- **SD3.5 Large**:38 个 MMDiT joint block(`joint_blocks.{N}`),每块内部
  分成 `context_block`(文本)和 `x_block`(图像)两半,共享联合注意力。
  标签 `J00..J37`。

本节点按各自模型的真实块索引,通过正则匹配 state_dict key 来分组 LoRA
权重,确保你设的强度对应模型实际跑的 transformer。新增一个 DiT 模型
只需写一个 `BlockSpec` 加 4 个薄子类 —— 最简单的模板见
`lora_block_weight/_sd35.py`。

## 致谢 / 灵感来源

按块 LoRA 权重这项技术来自
[hako-mikan/sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)
(SD1.5/SDXL,A1111)。本节点把这个思路移植到 ComfyUI,并适配了 FLUX.1 的
transformer 结构。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。

## 贡献

欢迎通过 [Issues](https://github.com/Baldwinzc/ComfyUI-LoraBlockWeight/issues)
提 bug 报告和 LoRA 相关发现 —— 如果你扫了某个流行 LoRA 并找到有意思
的块排序,欢迎分享。
