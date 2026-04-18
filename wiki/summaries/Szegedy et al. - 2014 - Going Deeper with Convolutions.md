# Szegedy et al. - 2014 - Going Deeper with Convolutions

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Szegedy et al. - 2014 - Going Deeper with Convolutions.pdf
- 原始 HTML：../../raw/html/Szegedy et al. - 2014 - Going Deeper with Convolutions.html
- 全文文本：../../raw/text/Szegedy et al. - 2014 - Going Deeper with Convolutions.md
- 作者：Szegedy et al.
- 年份：2014
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

这篇论文提出的 `Inception`/`GoogLeNet` 主线，核心不是单纯“把网络做得更深”，而是重新分配卷积计算：通过多分支、多尺度变换与 `1x1` 降维，在预算受控的前提下同时提升深度与宽度。它代表经典 CNN 从单一路径堆叠，转向显式多尺度分支设计的重要节点。

## 关键事实

- 论文提出 `Inception module`，把不同感受野的卷积与池化分支并行组织，再在输出端聚合。
- 为避免多分支导致计算爆炸，论文大量使用 `1x1` 卷积做降维与投影，这是其效率设计的关键。
- 文中给出的 `GoogLeNet` 是一个约 `22` 层的具体实现，并在 `ILSVRC 2014` 分类与检测任务中取得领先结果。
- 该路线体现的是“多尺度处理 + 受控复杂度”优先级，而不是 `VGG` 式统一重复模块。
- 从架构史看，`Inception` 把经典 CNN 的创新方向从“更深”扩展到“更复杂但计算受控的模块化多分支设计”。

## 争议与不确定点

- `Inception` 提升效率与精度的代价是模块设计更手工化、超参数更多，这也是后续 `ResNet / ResNeXt` 想重新收敛到更规则结构的重要背景。
- 当前 summary 讨论的是 `Inception v1 / GoogLeNet` 节点，不等于后续整个 `Inception` 家族的全部演化。
- 论文中的“保持计算预算常数”是相对同时代架构的设计目标，不代表它在今天仍是最简洁或最易部署的卷积 backbone。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[GoogLeNet](../../wiki/concepts/GoogLeNet.md)
