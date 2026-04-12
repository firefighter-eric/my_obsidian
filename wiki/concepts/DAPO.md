# DAPO

## 简介

DAPO 是 `Decoupled Clip and Dynamic sAmpling Policy Optimization` 的缩写。在当前知识库中，它表示面向长链路 reasoning RL 的大规模开源训练系统与算法 recipe，而不只是对 `GRPO` 的一个小修补。

## 关键属性

- 类型：reasoning-oriented RL / large-scale training system
- 代表来源：
  - [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../raw/summary/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- 当前角色：把 `GRPO`-style reasoning RL 推进到更可复现、系统化工程实现的代表节点

## 相关主张

- `DAPO` 从 naive `GRPO` 的失败案例出发，认为长 CoT RL 的主要难点在于熵塌缩、奖励噪声、长度偏置和梯度退化。
- 它通过 `Clip-Higher`、`Dynamic Sampling`、`Token-Level Policy Gradient Loss`、`Overlong Reward Shaping` 等机制修正基础 `GRPO`。
- 在当前知识库里，`DAPO` 的意义不只是“又一个 RL 算法”，而是公开了大规模 reasoning RL 的工程 recipe。

## 来源支持

- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../raw/summary/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- [LLM RL](../topics/LLM%20RL.md)

## 关联页面

- [GRPO](./GRPO.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)
