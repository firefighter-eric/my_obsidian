# Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition

## 来源信息

- 类型：论文 / arXiv
- 来源链接：https://arxiv.org/abs/2512.15603
- 原始文件：../../raw/pdf/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.pdf
- 全文文本：../../raw/text/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.md
- 作者：Shengming Yin, Zekai Zhang, Zecheng Tang, Kaiyuan Gao, Xiao Xu, Kun Yan, Jiahao Li, Yilei Chen, Yuxiang Chen, Heung-Yeung Shum, Lionel M. Ni, Jingren Zhou, Junyang Lin, Chenfei Wu
- 年份：2025
- 状态：已整理

## 摘要

`Qwen-Image-Layered` 试图把图像编辑中的一致性问题，转化为图像表示问题而非单纯编辑算法问题。论文提出一个端到端 diffusion decomposer，把单张 RGB 图像直接分解成多个语义解耦的 `RGBA` 图层，使缩放、移动、改色、局部替换等操作可以在目标层上独立完成，从而减少对未编辑区域的语义漂移与几何错位。

## 关键事实

- 论文明确把目标定义为 `single RGB image -> multiple semantically disentangled RGBA layers`，强调 `inherent editability`，而不是传统 mask-guided local editing。
- 方法有三个核心部件：`RGBA-VAE`、支持可变图层数的 `VLD-MMDiT`、以及把预训练图像生成模型逐步改造成图层分解器的 `multi-stage training`。
- `RGBA-VAE` 的作用不是单独做透明图生成，而是为输入 RGB 图像和输出 RGBA 图层建立共享潜空间，避免图层分解时出现通道表示断裂。
- `VLD-MMDiT` 在 `Layer3D RoPE` 和多模态注意力下直接建模图层内与图层间关系，目标是一次性支持可变数量图层，而不是反复做前景/背景递归拆分。
- 训练流程分三阶段推进：`text-to-RGB -> text-to-RGBA -> text-to-multi-RGBA -> image-to-multi-RGBA`，说明作者把图层分解理解为在已有生成模型基础上逐步迁移的新任务，而不是完全从零训练。
- 数据方面，作者从真实 `PSD` 文件中抽取、过滤、合并并标注多图层样本，以补足高质量 multilayer image 数据稀缺问题。
- 论文在 `Crello` 基准上报告优于 `LayerD` 等方法的结果，特别强调 `Alpha soft IoU` 提升更明显，说明方法优势主要体现在 alpha 通道和图层边界质量上。
- 该路线与 `Qwen-Image` 的关系不是简单替代。`Qwen-Image` 更偏文本渲染与图像编辑基础模型，而 `Qwen-Image-Layered` 更偏把“可编辑图层表示”直接做成图像分解接口。

## 争议与不确定点

- 论文主要聚焦图层分解质量与可编辑性逻辑，尚不足以证明这种 layered representation 会成为所有图像编辑系统的统一主流接口。
- 多图层监督依赖真实 `PSD` 工作流数据；这种数据分布与自然照片、开放互联网图像之间的泛化边界仍需后续来源验证。
- 论文与仓库中现有 `Qwen-Image` 来源之间并非同一任务设定，不能直接把两者 benchmark 成绩做一一对应比较。

## 关联页面

- 概念：[Qwen-Image-Layered](../../wiki/concepts/Qwen-Image-Layered.md)
- 概念：[RGBA 图层图像](../../wiki/concepts/RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- 概念：[Qwen-Image](../../wiki/concepts/Qwen-Image.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md)
