# Chinchilla

## 简介

Chinchilla 是 `Hoffmann et al. 2022` 中提出的 compute-optimal 训练代表模型。在当前知识库中，它主要表示“参数规模与训练 token 数量需要共同扩张”的修正路线。

## 关键属性

- 类型：compute-optimal 训练代表模型
- 代表来源：
  - [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../raw/summary/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- 当前角色：对早期“只扩模型、不扩数据”路线的关键修正点

## 相关主张

- `Hoffmann et al. 2022` 认为很多大模型在固定数据量下是欠训练的。
- Chinchilla 代表的不是“更大参数”本身，而是“在给定计算预算下重新平衡模型大小和训练 token”。
- 在知识库主线里，Chinchilla 是 GPT-3 之后 scaling 逻辑的重要纠偏节点。

## 来源支持

- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../raw/summary/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Scaling 与 compute-optimal training](../topics/Scaling%20与%20compute-optimal%20training.md)

## 关联页面

- [GPT-3](./GPT-3.md)
- [LLM 预训练](LLM%20预训练.md)
- [Scaling 与 compute-optimal training](../topics/Scaling%20与%20compute-optimal%20training.md)
