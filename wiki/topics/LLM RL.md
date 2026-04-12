# LLM RL

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页聚焦 LLM 后训练中与强化学习、偏好优化和 reasoning-oriented RL 直接相关的部分。它比 `指令对齐与 post-training` 更窄，不再讨论一般性的 instruction following，而是专门讨论 reward shaping、偏好学习、RL 优化与其在推理模型中的新角色。本页默认把 `RLHF / preference optimization / online RL / reasoning RL` 放在同一主题下讨论，但会明确区分它们的数据接口、优化方式与目标函数。

## 核心问题

- RL 在 LLM 后训练中到底扮演什么角色：行为校正、偏好拟合，还是能力激励。
- RLHF、DPO、ORPO、KTO 与 reasoning RL 的关系是替代、并列，还是分阶段演化。
- 复杂 RL 管线是否必要，哪些情况可以用更轻量的偏好优化替代，哪些情况又必须回到在线 RL。
- `GRPO`、`DAPO` 这类 reasoning RL 方法到底是在改优化器、改奖励设计，还是在补系统工程细节。
- 当 RL 目标从“更符合用户意图”转向“更强推理能力”时，评估与风险会发生什么变化。

## 主线脉络 / 方法分层

在当前知识库里，`LLM RL` 至少应拆成四层，而不是只写成 `RLHF -> DPO -> R1` 的线性演进。

- 经典 RLHF 管线：`Ouyang et al. 2022` 建立 demonstrations + preference rankings + reward model + RL 的完整对齐范式。这里的核心是“先学人类偏好，再用 RL 优化行为”，典型优化器背景是 `PPO`，重点是让模型更 helpful、truthful、harmless，而不是直接追求推理能力。
- reference-based 偏好优化：`Rafailov et al. 2023` 的 `DPO` 说明，在特定假设下可以不再显式训练 reward model + PPO，而把 RLHF 改写成更直接的分类式目标。它不是把 RL 从概念上彻底移除，而是把 RLHF 的最优策略形式折叠进闭式损失。
- reference-free / 弱化数据接口的偏好优化：`ORPO` 进一步把 `SFT + preference alignment` 合并为单阶段、无 reference model 的目标；`KTO` 则改写数据接口，只要求 unary desirable / undesirable 信号，而不要求 pairwise preference。它们共同表明，post-training 方法族正从“完整 RLHF 管线”分化到多种更轻量的 preference optimization。
- reasoning-oriented 在线 RL：`DeepSeekMath` 提出 `GRPO`，把 critic-free、group-relative advantage 的思路引入 LLM reasoning 训练；`DeepSeek-R1` 则将 RL 从“对齐手段”推进为“推理行为激励机制”，并让 `DeepSeek-R1-Zero` 这种无 SFT 起步的 RL 训练进入主流讨论。
- 大规模 reasoning RL 工程化：`DAPO` 表明，`GRPO` 只是 reasoning RL 的起点。真正把长 CoT RL 跑稳，还需要处理 entropy collapse、reward noise、长度偏置、sample efficiency 与 token-level loss 等系统问题，因此方法创新与训练工程在这一层已高度耦合。

从工程分层看，可粗分为“指令微调层”“偏好优化层”“在线 RL 行为校正层”“reasoning RL 层”。这些层彼此相连，但不能混写成单一技术名词。

## 关键争论与分歧

- RLHF 是否只是历史阶段：`DPO / ORPO / KTO` 证明传统 `reward model + PPO` 管线并非唯一做法，但这更像方法分化，而不是 RLHF 已完全过时。需要在线探索、显式奖励塑形或复杂 rollout 的场景，仍可能回到 RL 管线。
- 偏好优化是否等于“不要 RL”：从 `DPO` 的推导到 `DeepSeekMath` 的统一视角都表明，很多所谓“非 RL”方法仍可被理解为对 RLHF 的简化、重参数化或离线化，而不是与 RL 完全断裂。
- 对齐与推理强化是否应归为同一主题：当前知识组织上将其放在同一页，是因为两者共享奖励驱动与策略优化语法；但“更符合人类偏好”与“更会解题 / 更会长链推理”并非同一目标函数。
- SFT 是否必须存在：`ORPO` 默认把偏好对齐并入单阶段训练，`KTO` 讨论弱化偏好接口，`DeepSeek-R1-Zero` 则让“无 SFT 起步”成为可讨论路线；但其可读性、语言混杂与稳定性问题说明 SFT 仍具有强工程价值。
- reasoning RL 的关键难点是算法还是系统：`GRPO` 证明 critic-free online RL 可行，但 `DAPO` 说明真正决定成败的往往是 clip 设计、采样策略、token-level loss 与 overlong shaping 等工程细节。
- RL 收益如何评估：当前证据更强调 benchmark 提升、偏好胜率和数学 / 代码成绩，但对 reward hacking、monitorability、可解释性与长期稳定性的证据仍明显不足。

## 证据基础

- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../raw/summary/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../../raw/summary/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../raw/summary/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../raw/summary/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../raw/summary/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)

## 代表页面

- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)
- [GRPO](../concepts/GRPO.md)
- [DAPO](../concepts/DAPO.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)

## 未解决问题

- 当前知识库虽然已纳入 `ORPO / KTO / GRPO / DAPO`，但尚未做成 `RLHF vs DPO vs ORPO vs KTO` 的独立比较页，因此方法边界仍主要停留在 topic 级解释。
- reward model、`PPO`、RLAIF、process supervision、online iterative RLHF 等关键子概念还未完整进入 concept 层。
- `GRPO -> DAPO` 之间哪些提升来自算法，哪些来自工程 recipe，当前仍缺少跨论文的稳定比较页。
- reasoning RL 的评测指标、可读性约束、monitorability 与行为安全性仍需要更多来源支撑，而不能只用 `DeepSeek-R1` 与 `DAPO` 两个技术报告概括。

## 关联页面

- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM 预训练](LLM%20预训练.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)
- [GRPO](../concepts/GRPO.md)
- [DAPO](../concepts/DAPO.md)
