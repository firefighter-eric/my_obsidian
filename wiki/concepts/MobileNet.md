# MobileNet

## 简介

`MobileNet` 是轻量级卷积 backbone 的代表概念。在当前知识库中，它对应“围绕端侧约束重写 CNN 精度-延迟-模型大小折中”的主线。

## 关键属性

- 类型：视觉 backbone / 轻量 CNN
- 代表来源：[Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications](../../wiki/summaries/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md)
- 当前角色：承接移动端高效视觉模型路线

## 相关主张

- `MobileNet` 通过 depthwise separable convolution 显著降低计算量。
- 它把 width multiplier 与 resolution multiplier 做成全局调节手柄，强调工程可调性。
- 在当前知识库里，`MobileNet` 代表经典 CNN 中效率优先而非纯精度优先的独立分支。

## 来源支持

- [Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications](../../wiki/summaries/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ConvNeXt](./ConvNeXt.md)
