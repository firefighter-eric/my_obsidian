# LLM 基础脉络

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页是当前知识库中关于大语言模型主线的总入口，目标不是覆盖所有模型家族，而是建立一个最稳定、最可追溯的骨干叙事：大规模预训练如何带来通用能力、为何单纯扩参数不足以解释后续进展、以及为什么 post-training 最终成为独立主线。它与 `LLM预训练`、`Scaling 与 compute-optimal training`、`指令对齐与 post-training` 的区别在于：这些页面分别展开某一段路线，而本页负责把几段路线接成一条连续脉络。

## 核心问题

- 大模型能力的第一性来源是什么：参数规模、训练数据规模，还是训练目标的通用性。
- 预训练阶段获得的 few-shot 能力，是否足以支撑真实交互场景中的“可用性”。
- 在固定计算预算下，模型规模与训练 token 的最优配比应如何理解。
- 当模型能力已经显著提升后，为什么仍需要单独的 post-training / alignment 阶段。

## 主线脉络 / 方法分层

- 预训练规模化阶段：`Brown et al. 2020` 以 GPT-3 展示了“大规模自回归预训练 + 纯文本 prompting”可以显著提升 few-shot 迁移能力，使“无需任务专属梯度更新也能完成多类任务”成为 LLM 主线的起点。
- compute-optimal 修正阶段：`Hoffmann et al. 2022` 指出早期大模型往往是 under-trained，问题不只是参数要不要更大，而是参数与训练 token 是否共同扩张。自此，主线从“更大模型”转向“给定计算预算下的更优训练配置”。
- 对齐与行为塑形阶段：`Ouyang et al. 2022` 表明强预训练能力并不自动等于更 helpful、truthful、harmless 的交互行为，因而 supervised fine-tuning、偏好数据与 RLHF 被提升为独立阶段。
- 后续分化阶段：在当前知识库中，这条骨干线进一步延伸出开放模型家族、MoE 高效训练、reasoning-oriented RL 和多模态扩张等支线，但这些支线不改变三段式骨架本身。

## 关键争论与分歧

- 能力与行为是否应视为同一主线：当前证据更支持把“预训练能力获得”和“后训练行为塑形”视为两个相关但不可混同的阶段。
- scaling 的核心变量是什么：`Brown 2020` 更突出模型规模与 few-shot 泛化，`Hoffmann 2022` 则把数据量和 FLOPs 配置拉回到同等重要的位置。
- 代表论文是否足以概括主线：当前页面以三篇代表来源构建骨架，适合做稳定入口，但并不等于完整历史；MoE、tool use、多模态和 reasoning RL 仍需在子主题里展开。
- 开放模型时代是否改变主干逻辑：从目前知识库证据看，开放模型家族主要改变的是实现生态与工程可达性，而不是“预训练规模化 → 训练配置修正 → post-training 对齐”的基本顺序。

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../raw/summary/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)：支撑“规模化预训练带来 few-shot 能力跃迁”的第一阶段。
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../raw/summary/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)：支撑“参数量与 token 数需共同优化”的第二阶段。
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)：支撑“alignment / post-training 为独立问题”的第三阶段。

## 代表页面

- [LLM 预训练](LLM%20预训练.md)
- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [GPT-3](../concepts/GPT-3.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [InstructGPT](../concepts/InstructGPT.md)

## 未解决问题

- 当前骨干页尚未把开放模型家族如何继承这三段式逻辑写成更细的比较结构。
- reasoning-oriented RL 与传统 RLHF 的关系，目前仍停留在后续支线层，而非主干结论层。
- 多模态与 tool use 是否属于 LLM 主线的自然延展，还是应被视作与文本 LLM 并行的基础模型路线，仍需更多 summary 支撑。

## 关联页面

- [LLM 预训练](LLM%20预训练.md)
- [LLM RL](./LLM%20RL.md)
- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
