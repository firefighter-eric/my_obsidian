# Huang et al. - 2016 - Densely Connected Convolutional Networks

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Huang et al. - 2016 - Densely Connected Convolutional Networks.pdf
- 原始 HTML：../../raw/html/Huang et al. - 2016 - Densely Connected Convolutional Networks.html
- 全文文本：../../raw/text/Huang et al. - 2016 - Densely Connected Convolutional Networks.md
- 作者：Huang et al.
- 年份：2016
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`DenseNet` 将跨层连接从 `ResNet` 的逐块残差相加进一步推进为“每层接收此前所有层特征”的密集连接模式。它的重要性不只是又一个更深 backbone，而是在经典 CNN 中明确提出：跨层信息流可以通过特征复用而非重复学习来改善训练效率和参数效率。

## 关键事实

- 论文把每一层的输入定义为此前所有层特征图的串接，而不是只接前一层或通过残差相加接一个 shortcut。
- 这种 dense connectivity 被论文解释为缓解梯度消失、加强 feature propagation、鼓励 feature reuse，并减少冗余参数。
- `DenseNet` 在 `CIFAR-10 / CIFAR-100 / SVHN / ImageNet` 上报告了强结果，同时强调较高参数效率。
- 与 `ResNet` 相比，`DenseNet` 的关键差异不在“有没有跨层连接”，而在连接是 `summation` 还是 `concatenation`，以及由此带来的信息复用逻辑。
- 从演化视角看，`DenseNet` 代表经典 CNN 一条“通过更强连接结构提高效率”的分支，而不是继续单纯加深或加宽网络。

## 争议与不确定点

- 密集连接提升了特征复用，但也带来更重的特征拼接与内存访问负担；因此它不是所有部署场景下的默认优选。
- `DenseNet` 的参数效率很强，但其整体系统效率未必在所有硬件上都优于更规则的残差家族。
- 当前 summary 聚焦其连接思想与结构位置，尚未细拆 growth rate、transition layer 等实现超参数。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[DenseNet](../../wiki/concepts/DenseNet.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)
