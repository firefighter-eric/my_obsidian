# DeepSeek-V4

## 简介

DeepSeek-V4 是 DeepSeek 系列中面向百万 token 上下文、长程 agent 工作流和高效 MoE 推理的预览版模型家族。在当前知识库中，它承接 `DeepSeek-V3` 的高效 MoE 基座路线，但重点从“高效训练与强基座能力”推进到“长上下文推理成本、KV cache 压缩和 agent 化使用”。

## 关键属性

- 类型：MoE 语言模型家族 / 长上下文与 agent-oriented 模型
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- 当前角色：DeepSeek 从 `V3 / R1` 走向百万上下文、混合 attention 与 agent 工作流的一代节点

## 相关主张

- `DeepSeek-V4` 的关键不只是参数规模，而是把 `CSA / HCA` 混合 attention、`mHC` 与 `Muon` optimizer 组合成面向超长上下文的效率架构。
- `CSA` 负责“压缩后再稀疏选择”，`HCA` 负责更激进的重度 KV 压缩；二者交替使用，构成 `DeepSeek-V4` 的 hybrid attention。
- `mHC` 通过改造 residual connection 支撑深层信号传播稳定性，`Muon` 则作为多数模块的优化器服务于收敛速度与训练稳定性。
- `DeepSeek-V4-Pro` 与 `DeepSeek-V4-Flash` 的差异体现了同一家族内“能力上限”和“推理成本”之间的分层：Pro 追求更强复杂任务能力，Flash 更强调较低激活参数与更轻推理负担。
- 相比 `DeepSeek-V3`，`DeepSeek-V4` 更适合被放在“长上下文 + agent + sparse efficiency”的交叉位置，而不是只作为另一个开放模型 benchmark 节点。
- 由于官方报告把预训练、后训练、推理模式和 agent infrastructure 写在一起，使用本概念页时应按主题拆分证据：预训练页只引用其 base model 与架构事实，LLM RL 或 agent 相关页面再引用其后训练与工具使用设计。

## 来源支持

- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)

## 关联页面

- [DeepSeek](./DeepSeek.md)
- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [MoE](./MoE.md)
- [Compressed Sparse Attention](./Compressed%20Sparse%20Attention.md)
- [Heavily Compressed Attention](./Heavily%20Compressed%20Attention.md)
- [Manifold-Constrained Hyper-Connections](./Manifold-Constrained%20Hyper-Connections.md)
- [Muon](./Muon.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [LLM RL](../topics/LLM%20RL.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
