# He et al. - 2015 - Deep Residual Learning for Image Recognition

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/He et al. - 2015 - Deep Residual Learning for Image Recognition.pdf
- 原始 HTML：../../raw/html/He et al. - 2015 - Deep Residual Learning for Image Recognition.html
- 全文文本：../../raw/text/He et al. - 2015 - Deep Residual Learning for Image Recognition.md
- 作者：He et al.
- 年份：2015
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`ResNet` 的核心贡献，是把深层卷积网络的主要障碍从“表示能力不足”重新定位为“优化困难”，并用残差连接把层堆叠改写为“学习相对输入的增量修正”。它不是单纯更深的 `VGG`，而是经典 CNN 主线里最关键的优化接口重写：深度从此不再只是参数堆叠，而成为可以更稳定利用的能力来源。

## 关键事实

- 论文明确指出深层网络出现的 `degradation problem`：层数增加后训练误差反而变差，问题不只是过拟合。
- `ResNet` 通过 identity shortcut 把若干层重写为残差函数学习，并用逐元素相加把输入直接传到更深层。
- 论文展示了最高到 `152` 层的残差网络，并强调其比 `VGG` 更深但复杂度更低。
- 在论文叙述里，`ResNet` 不只提升 `ImageNet` 分类，也显著强化了检测与分割等下游视觉任务。
- 从知识库主线看，`ResNet` 是后续 `ResNeXt`、`ConvNeXt` 乃至许多检测/分割 backbone 的共同祖先接口。

## 争议与不确定点

- `ResNet` 解决的是深层优化问题，不等于同时解决参数效率、端侧部署效率或所有尺度建模问题。
- 论文中的经典残差块是后续变体的起点，不应机械等同于所有带 skip connection 的现代卷积架构。
- 当前 summary 聚焦其方法论地位，尚未细拆 basic block、bottleneck block 与不同 shortcut 选项的全部实验差异。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)
- 概念：[ResNeXt](../../wiki/concepts/ResNeXt.md)
- 概念：[ConvNeXt](../../wiki/concepts/ConvNeXt.md)
