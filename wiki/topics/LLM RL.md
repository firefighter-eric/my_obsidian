# LLM RL

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 **LLM 后训练中以奖励、偏好与策略优化为核心接口** 的方法族。它覆盖 `RLHF`、`DPO`、`ORPO`、`KTO`、`GRPO`、`DAPO`、`OPD` 以及 `DeepSeek-R1 / DeepSeek-V3.2` 一类 reasoning-oriented 与 thinking tool-use 路线，但 **不覆盖一般性的 instruction tuning 细节**，也不把所有 post-training 方法都笼统写成 RL。本页的边界是：只有当方法明确围绕 **奖励信号、偏好信号、策略更新、在线 rollout、行为激励、密集 token 级监督或其等价重写** 展开时，才进入 `LLM RL`。

因此，本页处理的不是“模型如何学会遵循指令”这一宽泛问题，而是更窄也更关键的问题：**当预训练能力已经存在后，是否需要通过奖励驱动的后训练机制来改变模型行为，乃至直接激励推理能力。** 这也是本页与 [指令对齐与 post-training](./指令对齐与%20post-training.md) 的区别。后者讨论行为塑形这一总框架；本页则讨论其中最具争议、最容易分化、也最接近策略优化语言的一支。

当前知识库中的稳定证据支持一个较强判断：**LLM RL 已经不是单一技术名词，而是从经典 RLHF 管线分化出的一个方法族。** 其中有些路线试图显式学习奖励并在线优化策略，有些路线把 RLHF 折叠为更直接的偏好目标，有些路线则把 RL 从“让模型更符合人类偏好”推进到“直接塑造推理行为与求解策略”。这些路线共享的是奖励驱动语法，而不是统一的训练配方。

## 核心问题

- **RL 在 LLM 后训练中到底解决什么问题**：是行为校正、偏好拟合、在线探索，还是直接提升推理能力。
- `RLHF`、`DPO`、`ORPO`、`KTO` 与 reasoning RL 的关系究竟是阶段演化、方法分叉，还是不同约束下的并行接口。
- 哪些场景可以把复杂的 `reward model + online RL` 简化为离线偏好优化，哪些场景又必须保留在线 rollout 与策略更新。
- `GRPO`、`DAPO` 这类 reasoning RL 方法的增益主要来自 **优化目标**、**奖励设计**，还是 **大规模训练工程**。
- 当 RL 的目标从“更 helpful / truthful / harmless”转向“更会推理 / 更会解题”时，评估、风险与可监控性会如何变化。

## 主线脉络 / 方法分层

本页不采用“`RLHF -> DPO -> R1`”的线性叙述，而按 **训练接口与目标函数的变化** 来分层。这样做的原因是：很多后续方法并不是简单替代前一代，而是在改变数据接口、参考模型依赖、在线性要求与奖励对象。

- **经典 RLHF 管线**：`Ouyang et al. 2022` 给出的不是一个局部 trick，而是一个完整范式：先用 demonstrations 做监督微调，再用 preference rankings 训练 reward model，最后用 RL 优化策略。其成立前提是，**人类偏好可以被近似建模，并作为比 next-token likelihood 更贴近产品目标的训练信号。** 这一层的关键贡献不在于 PPO 本身，而在于把“帮助性、真实性、无害性”转写成一个可迭代优化的后训练流程。
- **reference-based 偏好优化层**：`Rafailov et al. 2023` 的 `DPO` 之所以重要，不只是因为它“更简单”，而是因为它指出在一定假设下，RLHF 的最优策略可以被改写成更直接的 preference objective。这里的方法分层依据是：**奖励模型与在线 RL 是否必须显式存在。** `DPO` 保留了“偏好决定策略”的核心思想，但把优化形式从显式 RL 管线折叠为闭式损失。
- **reference-free 或弱化监督接口的偏好优化层**：`ORPO` 与 `KTO` 的价值，不是单纯再造一个对齐 loss，而是继续削弱 RLHF 管线中对外部部件与标注形式的依赖。`ORPO` 把 `SFT + preference alignment` 合并为单阶段目标；`KTO` 则把二元成对偏好改写为 unary desirable / undesirable 信号。这一层反映出 post-training 的一个稳定趋势：**方法正在从“完整 RL 管线”向“更轻量、数据接口更便宜的偏好优化族”扩散。**
- **reasoning-oriented 在线 RL 层**：`Shao et al. 2024` 的 `DeepSeekMath` 以及 `DeepSeek-R1` 表明，RL 在 LLM 中的角色已经发生变化。这里不再只是通过奖励让回答“更像人偏好的答案”，而是让模型在数学、代码或长链推理任务中 **形成更有效的中间行为模式**。`GRPO` 的意义在于，它把 critic-free、group-relative advantage 的在线优化方案带入 reasoning 训练，使“推理行为激励”成为一个可规模化讨论的对象。
- **大规模 reasoning RL 工程化层**：`DAPO` 说明 reasoning RL 的瓶颈并不止于“有没有一个好优化器”。当训练目标转向长链推理，长度偏置、reward noise、entropy collapse、sample efficiency、token-level loss、rollout 管理等问题会快速上升为一等公民。也就是说，**算法层与系统层在 reasoning RL 中已经高度耦合**，单独讨论某个 loss 往往不足以解释最终效果。
- **on-policy distillation 层**：`OPD` 把 teacher-student distillation 拉回到 student 自己的 rollout 分布上：student 先生成轨迹，再在这些轨迹上接受 teacher 的 token 级分布监督。它与 `GRPO / RLVR` 共享 on-policy 语法，但用 dense distillation signal 缓解 outcome reward 稀疏的问题；与 `SFT` 式 off-policy distillation 相比，它又更强调训练分布与推理分布的一致性。`G-OPD / ExOPD` 进一步把 OPD 解释为 dense KL-constrained RL 的特殊情形，说明蒸馏与 RL 在 LLM 后训练中并不是完全分离的两条线。
- **thinking tool-use 层**：`DeepSeek-R1-0528` 与 `DeepSeek-V3.2` 说明 reasoning model 正在从“会推理”走向“能用工具持续执行”。`R1-0528` 增加 JSON output 与 function calling，更多是可用性接口；`V3.2` 则把 thinking 直接集成进 tool-use，并引入覆盖 `1,800+` environments 与 `85k+` complex instructions 的 agent 训练数据合成。这个层次不是一般预训练能力，也不是纯 API 功能，而是 reasoning 后训练与 agent 系统接口开始合流的证据。

如果从知识组织角度再压缩一次，可以把本页方法族粗分为六类：**偏好建模型 RLHF**、**离线化偏好优化**、**online reasoning RL**、**reasoning RL 工程系统**、**on-policy dense distillation**、**thinking tool-use**。这样切分比按论文时间顺序更稳定，因为它对应的是不同的训练接口与目标边界。

## 关键争论与分歧

- **`RLHF` 是否只是过渡技术**：`DPO / ORPO / KTO` 的出现说明，传统 `reward model + PPO` 并非唯一实现路径；但这并不足以推出“RLHF 已经过时”。这一争论只有在区分“概念范式”和“具体工程配方”后才成立。更稳妥的结论是：**经典 RLHF 作为完整工程配方的中心性在下降，但奖励驱动的对齐范式并未消失。**
- **偏好优化是否等于“不要 RL”**：从 `DPO` 的推导，到 `DeepSeekMath` 对不同优化形式的统一理解，现有证据更支持把很多“非 RL”方法理解为 **对 RLHF 的离线化、闭式化或重参数化**，而不是与 RL 完全断裂。只有在把“是否显式在线 rollout”误写成“是否仍属于奖励驱动策略优化”时，这个争论才会被过度简化。
- **对齐 RL 与推理 RL 是否应放在同一主题**：本页把两者放在一起，不是因为目标相同，而是因为它们共享奖励与策略优化语法。争论真正成立的前提是：必须承认 **“更符合用户偏好”** 与 **“更会求解复杂问题”** 不是同一个目标函数。也因此，`DeepSeek-R1` 不应被直接当作 `InstructGPT` 的自然后续，而应视为 RL 在 LLM 中功能重心的一次转移。
- **`SFT` 是否仍然必要**：`ORPO`、`KTO` 和 `DeepSeek-R1-Zero` 都削弱了“必须先有强 SFT 起点”的直觉；但当前可追溯证据同样显示，完全绕开 SFT 往往会带来可读性、语言稳定性与训练可控性问题。因此更稳妥的判断不是“有无 SFT 的二选一”，而是：**SFT 仍是强工程先验，但其必要性已从理论前提退化为稳健性工具。**
- **reasoning RL 的主要瓶颈是算法还是系统**：`GRPO` 给出了 reasoning RL 的代表性算法接口，但 `DAPO` 更强调系统工程细节的决定性作用。只要当前证据仍主要来自技术报告而不是统一对照实验，就不能草率地把收益归因给单一算法创新。
- **OPD 是蒸馏还是 RL**：`OPD` 表面上是 teacher-student distillation，但 `Yang et al. 2026` 把它解释为 dense KL-constrained RL 的特殊情形。这个争论的关键不是命名，而是训练信号来源：如果只看优化形式，它更像带 teacher implicit reward 的密集 RL；如果看监督接口，它仍依赖 teacher logits。当前更稳妥的写法是把它放在 `LLM RL` 的相邻层，而不是把它硬塞进 `DPO / ORPO / KTO` 偏好优化分支。
- **tool-use 是否仍属于 RL / post-training**：`DeepSeek-V3.2` 使边界变得更复杂。工具调用格式本身不是 RL，但如果模型通过大规模环境、复杂指令和 thinking-in-tool-use 训练获得持续执行能力，就不能只把它视为产品 API。当前更稳妥的组织方式是把它放在 `LLM RL` 与后续 agent topic 的交叉位置，而不是写入 `LLM 预训练` 的基础规律。
- **RL 收益应如何评估**：现有 summary 多集中于 benchmark、偏好胜率、数学与代码成绩；但对 reward hacking、过程可监控性、链式思维可读性与长期行为稳定性的证据仍不足。因此当前 topic 可以较稳地讨论“性能收益”，却还不能对“安全收益”或“长期可控性收益”下过强结论。

## 证据基础

- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../../wiki/summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../wiki/summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- [DeepSeek AI - 2025 - DeepSeek-V3.2 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- [Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation](../../wiki/summaries/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)

## 代表页面

- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)
- [GRPO](../concepts/GRPO.md)
- [DAPO](../concepts/DAPO.md)
- [OPD](../concepts/OPD.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [RLHF vs DPO vs ORPO vs KTO](../comparisons/RLHF%20vs%20DPO%20vs%20ORPO%20vs%20KTO.md)

## 未解决问题

- 当前知识库已经能稳定区分 `RLHF / DPO / ORPO / KTO / GRPO / DAPO / OPD` 的任务接口，但 **`reward model`、`PPO`、RLAIF、process supervision、iterative online RLHF** 仍未形成完整 concept 与 comparison 支撑，因此本页对经典 RLHF 内部机制的展开仍然偏粗。
- `GRPO -> DAPO` 之间哪些提升来自 **算法改写**，哪些来自 **系统 recipe**，目前缺少独立 comparison 页与统一实验框架支撑；因此本页只能给出“高度耦合”的稳健判断，而不能精确归因。
- `OPD / G-OPD / ExOPD` 目前只有单篇 summary 进入知识库；其独立复现、跨模型族稳定性、长度偏置与 reward extrapolation 风险仍需要后续来源补强。
- `DeepSeek-V3.2` 已经把 thinking tool-use 写成 reasoning model 的重要接口，但当前知识库还没有正式 agent topic，因此工具环境、复杂任务执行与 agent benchmark 只能暂时在本页和 [DeepSeek 系列](./DeepSeek%20系列.md) 中承接。
- reasoning RL 的 **评测指标、可读性约束、monitorability、reward hacking 风险** 目前仍缺少更多 `wiki/summaries/` 支撑，现有结论主要依赖 `DeepSeek-R1` 与 `DAPO` 两个节点，证据覆盖面仍偏窄。
- 当前还不能凭现有 summary 断言“未来 post-training 将被 RL 统一”，也不能断言“偏好优化必然彻底替代在线 RL”；这两种说法都超出了当前证据基础。

## 关联页面

- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [FLAN](../concepts/FLAN.md)
- [LoRA](../concepts/LoRA.md)
- [OPT-IML](../concepts/OPT-IML.md)
- [Prompt Tuning](../concepts/Prompt%20Tuning.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)
- [GRPO](../concepts/GRPO.md)
- [DAPO](../concepts/DAPO.md)
- [OPD](../concepts/OPD.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [Toolformer](../concepts/Toolformer.md)
- [Llama Guard](../concepts/Llama%20Guard.md)
- [RLHF vs DPO vs ORPO vs KTO](../comparisons/RLHF%20vs%20DPO%20vs%20ORPO%20vs%20KTO.md)
