# Heavily Compressed Attention

## 简介

`Heavily Compressed Attention (HCA)` 是 `DeepSeek-V4` 中面向极长上下文的高压缩 attention 分支。它把更大跨度的 token `KV cache` 合并成单个 compressed KV entry，以更激进的压缩率降低长上下文推理中的缓存和 attention 计算负担。

## 关键属性

- 类型：长上下文 attention / 重度 `KV cache` 压缩
- 缩写：`HCA`
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- 当前角色：`DeepSeek-V4` 混合 attention 中负责“更高压缩率”的分支

## 相关主张

- `HCA` 与 `CSA` 都压缩 `KV cache`，但 `HCA` 使用更大的压缩率，并不执行 `CSA` 那种 top-k sparse attention selection。
- 为弥补重度压缩带来的局部细节损失，`DeepSeek-V4` 在 `CSA / HCA` 中还加入 sliding window attention 分支来保留近邻 token 依赖。
- 在 `DeepSeek-V4` 中，`HCA` 更偏“把超长上下文成本压到足够低”，`CSA` 更偏“压缩后仍做相关性选择”；二者组合才构成 V4 的 hybrid attention。
- `HCA` 的价值边界取决于压缩后信息是否仍足以支撑任务，不能只从 context length 数字判断。

## 来源支持

- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)

## 关联页面

- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [Compressed Sparse Attention](./Compressed%20Sparse%20Attention.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

