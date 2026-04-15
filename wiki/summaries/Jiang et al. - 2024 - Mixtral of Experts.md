# Jiang et al. - 2024 - Mixtral of Experts

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Jiang et al. - 2024 - Mixtral of Experts.pdf
- 全文文本：../../raw/text/Jiang et al. - 2024 - Mixtral of Experts.md
- 作者：Jiang et al.
- 年份：2024
- 状态：已基于现有全文整理

## 摘要

`Mixtral` 是 Mistral 家族把高效 dense 路线推进到 `MoE` 的关键节点。其核心不是简单堆总参数，而是通过稀疏激活维持较低每 token 计算成本，同时继续扩大模型总体容量，从而成为开放 `MoE` 家族的代表。

## 关键事实

- `Mixtral` 的代表架构是 `8x7B` 专家混合模型，并延伸到后续更大版本。
- 它把 Mistral 家族从高效 dense 模型推进到开放 `MoE` 主线。
- 该路线的重要工程意义在于：总参数显著增大，但每 token 激活量仍受控，从而兼顾能力与成本。
- 在当前知识库里，`Mixtral` 是开放模型中 `dense -> sparse/MoE` 演进的关键证据之一。

## 争议与不确定点

- `Mixtral` 的 benchmark 强势不应被简单等同于“MoE 一定优于 dense”，更准确地说，它表明 `MoE` 已成为开放模型竞争中的现实工程路线。
- 不同代际 `Mixtral` 版本在能力和部署成本上的差异，仍需更细 comparison 承接。

## 关联页面

- 概念：[Mixtral](../../wiki/concepts/Mixtral.md)
- 概念：[Mistral 7B](../../wiki/concepts/Mistral%207B.md)
- 概念：[MoE](../../wiki/concepts/MoE.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
