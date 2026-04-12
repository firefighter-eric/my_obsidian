# GRPO

## 简介

GRPO 是 `Group Relative Policy Optimization` 的缩写。在当前知识库中，它表示 reasoning-oriented LLM RL 中最有代表性的 `PPO` 变体之一：通过组内相对分数估计 baseline，省去 critic model，以更低资源成本完成策略优化。

## 关键属性

- 类型：在线 RL / policy optimization 方法
- 代表来源：
  - [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../raw/summary/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
  - [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
  - [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../raw/summary/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- 当前角色：连接 `PPO` 式 RLHF 与 reasoning-oriented RL 的关键算法节点

## 相关主张

- `DeepSeekMath` 将 `GRPO` 定义为 `PPO` 的高效变体：不训练 critic，而用同组采样结果的相对分数估计 advantage。
- `DeepSeek-R1` 把 `GRPO` 用作核心 RL 框架，说明它已从数学专门场景进入通用 reasoning model 训练主线。
- `DAPO` 进一步表明，朴素 `GRPO` 在长链路推理 RL 中会暴露 entropy collapse、奖励噪声与稳定性问题，因此 `GRPO` 更像基础骨架，而不是完整工程 recipe。

## 来源支持

- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../raw/summary/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../raw/summary/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- [LLM RL](../topics/LLM%20RL.md)

## 关联页面

- [DAPO](./DAPO.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)

