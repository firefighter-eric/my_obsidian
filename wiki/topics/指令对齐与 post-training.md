# 指令对齐与 post-training

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 **预训练之后，模型如何被重写为更可用的交互系统**。它的中心不是“模型还会不会继续学知识”，而是“已有能力如何被组织成更能遵循指令、响应偏好、保持安全边界并适应产品接口的行为形态”。这里的 `post-training` 主要覆盖 instruction tuning、监督示范、偏好数据、偏好优化与 `RLHF` 总框架，但 **不把 reasoning-oriented RL 的细节作为本页主角**；那部分应主要回收到 [LLM RL](./LLM%20RL.md)。

本页与 `LLM 预训练` 的边界在于：预训练解释 **能力底座的形成**，而 post-training 解释 **行为接口的塑形**。与 `LLM RL` 的边界在于：本页讨论从 instruction following 到 preference alignment 的总流程与方法结构；`LLM RL` 则专门处理奖励驱动、策略优化与 reasoning RL 的细部问题。

当前证据支持一个已经相当稳定的判断：**post-training 不是预训练的附属补丁，而是把 base model 转换为可交互系统的独立阶段。** 这也是为什么 `InstructGPT` 在知识史上的地位，不只是又一个微调技巧，而是明确建立了“预训练能力”与“面向用户的行为质量”之间的阶段性分工。

## 核心问题

- 为什么强预训练模型仍可能 **不 helpful、不 truthful、不 harmless**，以及这些缺陷为何不能仅靠扩大预训练规模自动消失。
- instruction tuning、监督示范、偏好学习与 `RLHF` 在后训练管线中分别解决什么问题。
- 行为改善为何能够在 **参数规模不占优** 的情况下显著提升产品体验，这种收益与基础能力提升应如何区分。
- `DPO` 一类方法到底是在替代 `RLHF`，还是在改写其实现方式。
- 当 post-training 从“让模型更会回答”延伸到“让模型更会推理、更会使用工具”时，主题边界应如何收束。

## 主线脉络 / 方法分层

本页按 **后训练解决的问题类型** 分层，而不是按“哪篇论文先发表”来写。因为 instruction tuning、偏好对齐与 RLHF 真正的差异，在于它们改变的是不同层级的行为接口。

- **指令接口建立层**：`Wei et al. 2021` 的 `Finetuned Language Models Are Zero-Shot Learners` 之所以关键，不只是因为它提升了 zero-shot 表现，而是因为它说明：**把任务表达统一成自然语言指令，本身就是一种可泛化的接口设计。** 在这一层，模型开始从“会续写文本”过渡到“能把指令当作任务约束来执行”。
- **监督示范塑形层**：在后训练管线中，监督示范的作用并不是提供全部知识，而是把模型拉入更接近用户预期的输出分布。它通常解决的是语气、结构、任务完成格式与初始行为稳定性问题。即便许多来源没有单独把这一层展开为独立论文，它仍然是理解 `InstructGPT` 管线不可省略的中间层。
- **偏好建模与 `RLHF` 层**：`Ouyang et al. 2022` 给出的关键不是“又做了一次微调”，而是证明了 demonstrations、preference rankings、reward model 与 RL 可以被组织成一个统一的对齐框架。在这里，后训练的目标从“预测下一个 token”转向 **优化更接近用户价值判断的行为分布**。也正是在这一层，post-training 被明确写成一个独立于预训练的产品化阶段。
- **偏好优化简化层**：`Rafailov et al. 2023` 的 `DPO` 指出，偏好对齐不一定必须经过显式 reward model + PPO 这一完整管线。它把 post-training 方法族进一步分化为“完整 RLHF 管线”与“更直接的 preference optimization”。因此，本页更适合把 `DPO` 理解为 **post-training 的方法内部分化**，而不是将其简单记作“RLHF 的替代者”。
- **reasoning-oriented 后训练外溢层**：`DeepSeek-R1` 说明后训练目标已经开始从经典 alignment 外溢到推理行为塑形。但在本页中，这一层只作为边界说明出现：它表明 post-training 不再只关心“更听话”，也开始关心“更会做题、更会长链推理”；其更细的优化与奖励问题仍应下沉到 `LLM RL`。

从知识组织上看，本页最稳定的分层是：**instruction interface 建立**、**监督行为塑形**、**偏好建模与 RLHF**、**偏好优化分化**。这样写能保持 post-training 的主题边界，而不会让页面被 reasoning RL 全面接管。

## 关键争论与分歧

- **instruction tuning 与 `RLHF` 的关系是什么**：当前证据更支持把 instruction tuning 视为后训练的前置层或相邻层，而不是 `RLHF` 的完整替代。只有在区分“让模型理解指令接口”与“让模型按人类偏好优化行为”之后，这一争论才有意义。
- **对齐收益来自哪一层**：`Ouyang 2022` 的经典结果常被概括为“小模型经过后训练可优于更大但未对齐的模型”。这一结论成立，但其适用边界是 **行为质量与交互可用性**，而不是“基础知识和推理能力已完全可被后训练替代”。因此不能把该结论过度外推为“预训练规模不再重要”。
- **`DPO` 是否会取代 `RLHF`**：现有证据更支持“`DPO` 是重要分化方向”而不是“全面替代”。这一争论只有在区分 **工程复杂度** 与 **概念目标** 后才站得住。`DPO` 改变了实现路径，但并没有让“偏好决定后训练目标”这个中心前提消失。
- **reasoning RL 是否仍属于 alignment**：`DeepSeek-R1` 使这一边界开始模糊。当前更稳妥的做法不是强行划一，而是承认：后训练正在从“helpfulness / harmlessness”扩展到“问题求解行为塑形”，但这并不意味着传统 alignment 议题已经失效。
- **post-training 是否只是产品层技巧**：当前证据并不支持这种降格理解。无论是 `InstructGPT` 的阶段性影响，还是后续偏好优化方法族的扩展，都说明 post-training 已经是现代 LLM 系统设计中的核心组成部分，而非上线前的小修补。

## 证据基础

- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)

## 代表页面

- [FLAN](../concepts/FLAN.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [OPT-IML](../concepts/OPT-IML.md)
- [LoRA](../concepts/LoRA.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)

## 未解决问题

- 当前知识库还未把 **reward model、PPO、RLAIF、constitutional alignment、process supervision** 系统沉淀为 concept 或 comparison 层，因此本页对 post-training 内部谱系的展开仍不够细。
- `helpful / truthful / harmless` 三类目标之间的张力，目前仍缺少稳定 comparison 页；现阶段只能在 topic 里指出其并非自然一致，而不能给出细粒度结论。
- reasoning-oriented RL 与经典 alignment 共享多少数据、评测与优化假设，仍需要更多 `wiki/summaries/` 支撑；因此本页只能把它写成 **边界正在扩张**，而不能写成已完成统一。
- 当前 evidence base 仍不足以判断不同家族在 post-training 上的长期最优路线，例如“`DPO` 型方法是否会全面取代完整 RLHF 管线”，这一问题仍应保持不确定。

## 关联页面

- [LLM 预训练](./LLM%20预训练.md)
- [LLM RL](./LLM%20RL.md)
- [GPT-3](../concepts/GPT-3.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [Prompt Tuning](../concepts/Prompt%20Tuning.md)
