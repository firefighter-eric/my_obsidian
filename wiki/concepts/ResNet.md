# ResNet

## 简介

`ResNet` 是经典残差卷积网络的代表概念。在当前知识库中，它对应“用残差连接重写深层 CNN 优化问题”的核心节点。

## 关键属性

- 类型：视觉 backbone / 经典 CNN
- 代表来源：[He et al. - 2015 - Deep Residual Learning for Image Recognition](../../wiki/summaries/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md)
- 当前角色：连接 `VGG` 深层堆叠范式与后续现代 backbone 家族的中枢节点

## 相关主张

- `ResNet` 通过 identity shortcut 让深层网络更易优化，把深度真正转化为可用能力。
- 它成为后续 `ResNeXt`、`ConvNeXt` 乃至大量检测/分割 backbone 的共同基座。
- 在当前知识库里，`ResNet` 是经典 CNN topic 的主干概念，而不是其中一个普通变体。

## 来源支持

- [He et al. - 2015 - Deep Residual Learning for Image Recognition](../../wiki/summaries/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md)
- [Liu et al. - 2022 - A ConvNet for the 2020s](../../wiki/summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ResNeXt](./ResNeXt.md)
- [ConvNeXt](./ConvNeXt.md)
- [Faster R-CNN](./Faster%20R-CNN.md)
