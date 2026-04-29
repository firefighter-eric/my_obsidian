# Muon

## 简介

`Muon` 是 `DeepSeek-V4` 中用于多数模块的训练优化器。DeepSeek-V4 报告将其作为提升收敛速度与训练稳定性的关键组件之一，并与 `AdamW` 分工使用。

## 关键属性

- 类型：LLM 训练优化器 / 大规模训练稳定性技术
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- 当前角色：`DeepSeek-V4` 在架构之外的训练优化组件

## 相关主张

- `DeepSeek-V4` 使用 `Muon` 更新多数模块，同时保留 `AdamW` 用于 embedding、prediction head、`mHC` 的静态 bias / gating factors 和 `RMSNorm` 权重。
- 官方报告把 `Muon` 的作用概括为更快收敛与更强训练稳定性，并使用 hybrid Newton-Schulz iterations 做近似正交化。
- `Muon` 在 DeepSeek-V4 中不是单独的模型能力来源，而是与 `MoE`、`CSA / HCA`、`mHC` 一起构成大规模训练和长上下文效率架构的工程底座。
- 由于 `Muon` 需要完整梯度矩阵，官方报告还专门设计了与 `ZeRO` 并行和 MoE 参数更新兼容的实现策略。

## 来源支持

- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)

## 关联页面

- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
