# VGG

## 简介

`VGG` 是经典深层卷积网络的代表概念。在当前知识库中，它对应“小卷积规则堆叠 + 依靠深度提升视觉识别能力”的早期标准范式。

## 关键属性

- 类型：视觉 backbone / 经典 CNN
- 代表来源：[Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition](../../wiki/summaries/Simonyan,%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md)
- 当前角色：作为 `ResNet` 之前最典型的规则深层卷积基线

## 相关主张

- `VGG` 用重复的 `3x3` 卷积堆叠建立了规则化的深层 CNN 设计语言。
- 它的重要性在于证明更深的卷积网络能系统提升大规模视觉识别性能，而不是在于参数效率。
- 在当前知识库里，`VGG` 是理解 `ResNet / ConvNeXt` 何以重要的必要历史基线。

## 来源支持

- [Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition](../../wiki/summaries/Simonyan,%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ResNet](./ResNet.md)
