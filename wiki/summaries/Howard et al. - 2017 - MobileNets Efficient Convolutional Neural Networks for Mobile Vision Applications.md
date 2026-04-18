# Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.pdf
- 原始 HTML：../../raw/html/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.html
- 全文文本：../../raw/text/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.md
- 作者：Howard et al.
- 年份：2017
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`MobileNet` 把经典 CNN 主线中的核心问题从“如何继续提升精度”明确扩展为“如何在移动端和嵌入式场景下做可调节的精度-延迟-模型大小折中”。它的代表性不只是轻量，而是把 `depthwise separable convolution` 与全局缩放超参数一起做成一套工程化 backbone 设计语言。

## 关键事实

- 论文以 `depthwise separable convolution` 作为主体算子，把空间滤波与通道混合拆开，以显著降低计算量。
- `MobileNet` 引入 width multiplier 与 resolution multiplier 两个全局超参数，让模型尺寸和延迟能够按资源预算连续调节。
- 论文把目标场景明确设定为 mobile / embedded vision，而不是桌面 GPU 上的纯精度竞赛。
- 文中不仅评估 `ImageNet` 分类，也展示其在检测、细粒度识别、人脸属性和地理定位等任务中的迁移性。
- 从演化视角看，`MobileNet` 代表经典 CNN 主线从通用高性能 backbone 分化出“资源受限优先”的独立路线。

## 争议与不确定点

- `depthwise separable convolution` 的理论高效不总能等价转化为所有硬件上的实际吞吐优势，部署收益取决于内核实现与平台特性。
- `MobileNet` 的价值主要在效率折中，不宜直接拿其最高精度与高算力 backbone 做简单优劣判断。
- 当前 summary 讨论的是 `MobileNet v1`，并不自动覆盖后续 `v2 / v3` 与倒残差等进一步演化。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[MobileNet](../../wiki/concepts/MobileNet.md)
