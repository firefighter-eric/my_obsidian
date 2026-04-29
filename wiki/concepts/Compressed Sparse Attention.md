# Compressed Sparse Attention

## 简介

`Compressed Sparse Attention (CSA)` 是 `DeepSeek-V4` 中用于百万 token 长上下文的 attention 机制之一。它先把连续 token 的 `KV cache` 压缩成更少的 compressed KV entries，再用 `DeepSeek Sparse Attention (DSA)` 从压缩后的 KV 块中选择 top-k 参与核心 attention。

## 关键属性

- 类型：长上下文 attention / `KV cache` 压缩 / 稀疏 attention
- 缩写：`CSA`
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- 当前角色：`DeepSeek-V4` 混合 attention 中负责“压缩 + 稀疏选择”的分支

## 相关主张

- `CSA` 的核心不是只做固定窗口 attention，而是先压缩 `KV cache`，再通过 lightning indexer / sparse selection 选择最相关的 compressed KV entries。
- 相比完整 attention，`CSA` 的目标是减少长上下文下的 attention 计算和缓存访问；相比 `HCA`，它保留了更细的稀疏选择能力。
- 在 `DeepSeek-V4` 中，`CSA` 与 `HCA` 交替构成 hybrid attention，使百万 token context 在工程上更可行。
- `CSA` 仍属于 `DeepSeek-V4` 官方报告中的 preview 架构事实，其独立泛化价值仍需要更多外部复现。

## 来源支持

- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)

## 关联页面

- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [Heavily Compressed Attention](./Heavily%20Compressed%20Attention.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

