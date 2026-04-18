# Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks.pdf
- 原始 HTML：../../raw/html/Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks.html
- 全文文本：../../raw/text/Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks.md
- 作者：Xie et al.
- 年份：2016
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`ResNeXt` 的关键贡献，是在 `ResNet` 已经证明残差连接有效之后，把网络容量的主调节维度从“只看深度和宽度”扩展到第三个维度 `cardinality`。它试图保留残差家族的规则结构，同时吸收 `Inception` 的多分支思想，但用更统一、更易扩展的模块化方式实现。

## 关键事实

- 论文提出 `cardinality` 作为与 depth、width 并列的重要模型维度，指同拓扑变换分支的数量。
- `ResNeXt` 采用聚合残差变换与 grouped convolution，在保持复杂度受控时提升表示能力。
- 作者明确把它定位为结合 `VGG/ResNet` 的规则重复模块与 `Inception` 的 split-transform-merge 思想。
- 论文报告在 `ImageNet-1K / ImageNet-5K / COCO` 上均优于对应 `ResNet` 基线，并作为 `ILSVRC 2016` 重要提交基础。
- 从知识库结构看，`ResNeXt` 是经典 CNN 从“更深”转向“更强模块内部并行度”的关键节点，也影响了后续检测 backbone。

## 争议与不确定点

- `ResNeXt` 虽然比 `Inception` 更规则，但 grouped convolution 的真实硬件效率依赖具体实现，不能只凭理论复杂度判断部署成本。
- `cardinality` 是强有力的结构维度，但并不意味着深度和宽度从此不重要；它更像是在残差家族中新增一个更有效的容量调节手柄。
- 当前 summary 主要覆盖 `ResNeXt` 相对 `ResNet / Inception` 的结构定位，尚未逐项展开 `32x4d` 等配置差异。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[ResNeXt](../../wiki/concepts/ResNeXt.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)
- 概念：[GoogLeNet](../../wiki/concepts/GoogLeNet.md)
