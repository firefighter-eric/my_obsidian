# Jiang et al. - 2023 - Mistral 7B

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Jiang et al. - 2023 - Mistral 7B.pdf
- 全文文本：../../raw/text/Jiang et al. - 2023 - Mistral 7B.md
- 作者：Jiang et al.
- 年份：2023
- 状态：已基于现有全文整理

## 摘要

`Mistral 7B` 是开放模型从“尽量做大”转向“在小得多的参数规模上追求更强效率与更高质量”的关键节点。报告通过 `GQA` 与 `sliding window attention` 把 `7B` 级模型推到足以正面挑战更大开放模型的水平，也由此奠定了 Mistral 家族的工程声望。

## 关键事实

- `Mistral 7B` 是 `7B` 参数开放基础模型，并附带 `Instruct` 版本。
- 报告明确强调其使用 `Grouped-Query Attention (GQA)` 与 `Sliding Window Attention (SWA)` 来兼顾推理效率与长序列处理。
- 它的代表性意义在于：用更小尺寸打出对 `Llama 2 13B` 等更大开放模型的竞争力。
- 在开放模型主线中，`Mistral 7B` 是“小而强”的高效 dense 模型范式节点。

## 争议与不确定点

- `Mistral 7B` 很强，但其成功不等于“规模不重要”，更准确地说是显示了架构、数据和训练配方的重新平衡。
- 它是 Mistral 家族起点，不应把后续 `Mixtral` 的 MoE 特征反向投射到这一代 dense 模型上。

## 关联页面

- 概念：[Mistral 7B](../../wiki/concepts/Mistral%207B.md)
- 概念：[Mixtral](../../wiki/concepts/Mixtral.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)
