# Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition

## 来源信息

- 类型：论文 / arXiv / Adobe Research
- 来源链接：https://arxiv.org/abs/2511.17454
- 原始文件：../../raw/pdf/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.pdf
- 全文文本：../../raw/text/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.md
- 作者：Nissim Maruani, Peiying Zhang, Siddhartha Chaudhuri, Matthew Fisher, Nanxuan Zhao, Vladimir G. Kim, Pierre Alliez, Mathieu Desbrun, Wang Yifan
- 年份：2026
- 状态：已整理

## 摘要

`Illustrator's Depth` 试图把“图像分层”重新定义为一种面向编辑的深度，而不是物理世界中的几何深度。论文提出对每个像素预测一个全局一致的 layer index，使平面插画、海报、阴影、轮廓线等内容能被分解成可重排、可编辑的有序层结构，从而服务 vectorization 与 depth-aware editing。

## 关键事实

- 作者明确区分 `illustrator's depth` 与 monocular depth：前者服务编辑与构图顺序，后者服务物理几何。
- 论文同时区分它与 panoptic / amodal segmentation：分层不只要分区域，还要给出全局可传递的层级顺序。
- 方法基于大规模分层 `SVG` 数据训练网络，从 raster input 直接预测每像素的 layer index / ordinal depth map。
- 数据来源依赖分层矢量图，并通过合并连续同色层、移除歧义案例、再光栅化为 RGB-depth 对的流程生成监督。
- 作者把这一路线定位为 vectorization 的核心前置能力：有了 layer index，传统 raster-to-vector pipeline 才能输出更可编辑的 SVG 层栈。
- 文中还展示文本到矢量图、2D 到 3D relief、depth-aware object insertion 等下游应用，说明它不是纯分析任务，而是编辑基础设施。
- 这条路线拓宽了 layered 主题的边界：分层不只属于 `RGBA` raster 编辑，也可以是矢量插画中的离散全局排序问题。

## 争议与不确定点

- 该方法的训练与评测高度依赖分层 `SVG` 分布，因此它对自然照片和复杂真实场景的适用边界仍不清楚。
- 论文强调“creative abstraction”而非 physical depth，这一设定对设计很有价值，但也意味着它不能直接替代所有通用深度或 segmentation 模型。
- 它更偏矢量化与插画编辑，而不是 `PSD` 式透明图层分解；将其与 `LayerDecomp` 或 `Qwen-Image-Layered` 直接视作同任务并不准确。

## 关联页面

- 主题：[图像分层 layered](../../wiki/topics/%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)
- 主题：[传统 CV](../../wiki/topics/%E4%BC%A0%E7%BB%9F%20CV.md)
- 主题：[Slide 理解与生成](../../wiki/topics/Slide%20%E7%90%86%E8%A7%A3%E4%B8%8E%E7%94%9F%E6%88%90.md)
