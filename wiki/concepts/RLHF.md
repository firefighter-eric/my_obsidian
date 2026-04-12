# RLHF

## 简介

RLHF 是 Reinforcement Learning from Human Feedback 的缩写。在当前知识库中，它表示“利用人类偏好信号对语言模型进行后训练”的经典对齐管线。

## 关键属性

- 类型：对齐 / post-training 方法框架
- 代表来源：
  - [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
  - [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../raw/summary/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- 当前角色：连接 InstructGPT 与 DPO 的上位方法概念

## 相关主张

- `Ouyang et al. 2022` 将 RLHF 管线拆成 supervised fine-tuning、reward modeling 与 reinforcement learning 三个主要阶段。
- 在当前知识库里，RLHF 不是单一算法，而是一整套“用偏好数据塑造模型行为”的工程框架。
- `Rafailov et al. 2023` 对 DPO 的论述以 RLHF 为对照，因此 RLHF 也是理解后续偏好优化简化路线的参照基线。
- `ORPO`、`KTO` 等方法说明 RLHF 后续分化出的很多路线，本质上都仍在回答“如何用反馈塑造策略”这一同类问题，只是数据接口和优化形式不同。
- 在 reasoning-oriented RL 中，`GRPO`、`DAPO` 等方法把 RLHF 的“RL”部分推进到了更直接的能力激励场景，但这不应与早期“偏好对齐型 RLHF”混为同一目标。

## 来源支持

- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../raw/summary/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [LLM RL](../topics/LLM%20RL.md)

## 关联页面

- [InstructGPT](./InstructGPT.md)
- [DPO](./DPO.md)
- [ORPO](./ORPO.md)
- [KTO](./KTO.md)
- [GRPO](./GRPO.md)
- [DAPO](./DAPO.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)
