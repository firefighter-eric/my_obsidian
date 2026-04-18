# Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition.pdf
- 原始 HTML：../../raw/html/Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition.html
- 全文文本：../../raw/text/Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition.md
- 作者：Simonyan, Zisserman
- 年份：2014
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`VGG` 的关键贡献不是发明某个全新的卷积算子，而是把“更深的卷积网络是否能稳定提升大规模视觉识别效果”系统化为一个可复用的设计原则：用重复的 `3x3` 小卷积堆叠替代更大感受野卷积，并把网络深度一路推到 `16-19` 个带权重层。它的重要性在于把网络设计从早期较多启发式组件的拼装，推进到更规则、可扩展、易迁移的深层卷积 backbone 范式。

## 关键事实

- 论文围绕“在其他设计近似固定时，单独提升卷积网络深度会带来什么收益”展开，核心变量是深度而不是花哨模块。
- `VGG` 采用重复的 `3x3` 卷积堆叠，以较小卷积核逐步扩大有效感受野，并形成高度规则的网络结构。
- 最终代表配置把网络深度推进到 `16-19` 个带权重层，并在 `ILSVRC 2014` 分类与定位中取得领先结果。
- 论文强调其 learned representation 能迁移到其他视觉数据集，这也是 `VGG` 长期被当作通用视觉特征提取 backbone 的原因之一。
- 从后续架构史看，`VGG` 的地位主要在于确立“规则堆叠的小卷积深层网络”这一设计范式，为 `ResNet`、`DenseNet` 等更深结构提供了清晰基线。

## 争议与不确定点

- `VGG` 的历史影响力很强，但它并不直接解决深层网络优化困难与参数效率问题；这些问题是在后续 `ResNet / DenseNet / MobileNet` 中被进一步处理的。
- 论文中的优势主要建立在当时的 `ImageNet` 识别设置与硬件条件上，不应把其复杂度-精度比直接外推为当前最优。
- 当前 summary 聚焦其在经典 CNN 演化中的结构地位，尚未细拆不同配置 `A-E` 的全部实验细节。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[VGG](../../wiki/concepts/VGG.md)
