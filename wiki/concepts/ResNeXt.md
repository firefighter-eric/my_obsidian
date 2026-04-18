# ResNeXt

## 简介

`ResNeXt` 是聚合残差变换网络的代表概念。在当前知识库中，它对应经典 CNN 中“把 cardinality 作为独立容量维度”的关键节点。

## 关键属性

- 类型：视觉 backbone / 经典 CNN
- 代表来源：[Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks](../../wiki/summaries/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md)
- 当前角色：连接 `ResNet` 的规则残差结构与 `Inception` 的多分支思想

## 相关主张

- `ResNeXt` 提出 `cardinality` 是与 depth、width 并列的重要模型维度。
- 它用 grouped convolution 和统一模块模板把多分支思想收敛为更规则的残差家族。
- 在当前知识库里，`ResNeXt` 也是许多检测 backbone 偏好采用的高性能残差分支之一。

## 来源支持

- [Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks](../../wiki/summaries/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ResNet](./ResNet.md)
- [GoogLeNet](./GoogLeNet.md)
