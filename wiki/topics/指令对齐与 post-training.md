# 指令对齐与 post-training

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页讨论预训练之后，模型如何从“会续写文本”转向“更符合用户意图地完成任务”。这里的重点是 instruction tuning、监督示范、偏好数据与 RLHF 管线如何共同塑造模型行为。与 `LLM RL` 相比，本页更强调从 InstructGPT 出发的总方法框架；与 `LLM预训练` 相比，本页把能力底座视为前提，而把行为塑形视为独立阶段。

## 核心问题

- 为什么强预训练模型仍然可能不 helpful、不 truthful、不 harmless。
- 指令微调、偏好学习和 RLHF 在管线中分别解决什么问题。
- 行为改善为何能在小于基础模型的参数规模上体现出来。
- 后训练方法的收益应如何与基础能力提升区分。

## 主线脉络 / 方法分层

- 指令可遵循性阶段：`Wei et al. 2021` 展示 instruction tuning 可以显著提升 zero-shot 泛化，是从“语言建模”到“任务指令接口”的第一步。
- RLHF 管线阶段：`Ouyang et al. 2022` 把 demonstrations、偏好排序与 reinforcement learning 组织成一条完整对齐管线，明确把对齐目标从 next-token prediction 转向用户意图。
- 偏好优化简化阶段：`Rafailov et al. 2023` 以 DPO 说明，偏好对齐不一定需要显式 reward model + PPO 才能成立，从而把 post-training 分化成更轻量的方法族。
- reasoning-oriented 后训练阶段：在当前知识库里，`DeepSeek-R1` 表明 RL 已不只用于“更听话”，也被直接用于激励推理行为与 reasoning style 的形成。

## 关键争论与分歧

- instruction tuning 与 RLHF 的关系：当前证据更支持把 instruction tuning 视为前置层或相邻层，而非 RLHF 的完整替代。
- 对齐收益来自哪一层：`Ouyang 2022` 的结果说明行为质量可以独立于参数规模改善，但这不等于基础能力不再重要。
- DPO 是否会取代 RLHF：从当前证据看，DPO 是方法简化而非对所有后训练情形的完全替代。
- reasoning RL 是否仍属于传统 alignment：`DeepSeek-R1` 说明该边界正在模糊，后训练目标开始从“安全与可用”延伸到“推理轨迹与问题求解能力”。

## 证据基础

- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../raw/summary/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../raw/summary/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)

## 代表页面

- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)

## 未解决问题

- 当前知识库尚未把 reward model、PPO、RLAIF、constitutional alignment 等后续分支单独沉淀成对照页。
- helpful、truthful、harmless 三类目标之间的权衡，仍缺乏系统比较而不只是目标口号层的整理。
- reasoning-oriented RL 与传统 alignment 共享多少评测与数据假设，仍需更多 summary 支撑。

## 关联页面

- [LLM RL](./LLM%20RL.md)
- [LLM 基础脉络](./LLM%20基础脉络.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
