# Manifold-Constrained Hyper-Connections

## 简介

`Manifold-Constrained Hyper-Connections (mHC)` 是 `DeepSeek-V4` 用来强化 Transformer block 间 residual connection 的结构设计。它在 `Hyper-Connections (HC)` 的基础上，把 residual mapping 约束到特定流形上，以改善深层堆叠中的信号传播稳定性。

## 关键属性

- 类型：Transformer residual connection 改造 / 训练稳定性结构
- 缩写：`mHC`
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- 当前角色：`DeepSeek-V4` 在 attention 与 MoE 之外的稳定性结构

## 相关主张

- `mHC` 的动机是：普通 residual connection 或 naive `Hyper-Connections` 在深层堆叠中可能出现数值不稳定，限制模型继续扩展。
- `DeepSeek-V4` 把 residual mapping matrix 约束到 doubly stochastic matrices 所在的 Birkhoff polytope，使映射具备 non-expansive 性质，帮助前向传播和反向传播稳定。
- `mHC` 不是 attention 机制，也不是 MoE 路由；它更像是 DeepSeek-V4 为深层模型稳定训练增加的连接结构。
- 官方报告也指出 `mHC` 会增加 activation memory 与 pipeline communication，因此需要 fused kernels、recomputation 与 pipeline overlap 等工程优化。

## 来源支持

- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)

## 关联页面

- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [Transformer](./Transformer.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

