# DPO

## 简介

DPO 是 `Rafailov et al. 2023` 提出的偏好优化方法。在当前知识库中，它表示“无需显式奖励模型与 PPO，也能做偏好对齐”的方法分支。

## 关键属性

- 类型：偏好优化 / post-training 方法
- 代表来源：
  - [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../raw/summary/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- 当前角色：RLHF 之后更轻量的对齐方法代表

## 相关主张

- `Rafailov et al. 2023` 认为可把标准 RLHF 问题改写成更直接的优化形式，从而避免复杂的 reward model + RL 管线。
- 在知识库里，DPO 代表“偏好对齐可以更稳定、更简单地实现”的方法论变化。
- 它适合作为对比 InstructGPT / PPO 式 RLHF 的方法概念页。

## 来源支持

- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../raw/summary/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [LLM RL](../topics/LLM%20RL.md)

## 关联页面

- [InstructGPT](./InstructGPT.md)
- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)
