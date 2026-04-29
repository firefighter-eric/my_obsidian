# LLM 预训练

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 **大语言模型在 post-training 之前，如何通过大规模自监督训练获得通用能力底座**。这里的重点是预训练目标、规模化规律、数据与计算预算配置、开放模型家族的训练取向，以及 dense 与 sparse 路线的结构性分化。它 **不讨论** RLHF、DPO、GRPO 这类后训练行为塑形方法，也不把工具调用、多模态或 agent 行为当作预训练页的直接中心，除非它们能被明确回收到“能力底座如何形成”这一问题。

本页的边界必须严格，因为当前很多模型报告会把预训练、SFT、RL、部署工程与产品能力写在同一份技术文档里。知识组织上，如果不把 **“能力底座”** 与 **“行为改写”** 分开，`LLM 预训练` 就会退化成一个总目录页。当前更稳妥的理解是：**预训练决定模型大致会不会、能不能、会到什么程度；post-training 决定这些能力怎样被组织成可交互、可约束、可产品化的行为接口。**

从现有 summary 来看，本页的主线不是“某一家模型赢了什么 benchmark”，而是三层连续变化：第一，`GPT-3 / PaLM` 所代表的 **dense scaling** 如何证明大规模自回归预训练能产生通用 few-shot 能力；第二，`Chinchilla` 如何把讨论从“继续变大”修正为“在固定算力下合理配置参数量与 token 数”；第三，开放模型家族如何在这个框架下分化出多语言、代码、MoE、本地部署与 fully open 等不同竞争方向。

## 核心问题

- **通用语言能力主要如何从预训练中形成**，以及这种能力与后训练行为改善应如何切分。
- dense scaling、compute-optimal 修正与 sparse/MoE 路线之间的关系是什么，哪些是补充，哪些是路线分化。
- 开放模型家族之间真正可比的维度是什么：参数规模、数据规模、训练效率、语言覆盖、透明度，还是部署友好性。
- “更大的模型”与“更合理的数据和计算配置”之间，哪个更应被视为预训练阶段的核心驱动力。
- 当前知识库对 LLM 主线的叙述，应该以闭源标杆为骨架，还是以开放家族竞争格局为骨架。

## 主线脉络 / 方法分层

本页按 **能力形成逻辑** 分层，而不是按模型发布时间罗列。因为对预训练的理解，关键不在于记住家族名单，而在于把“为何能力出现”“如何更有效训练”“为何开放家族分叉”放到同一结构里。

- **dense scaling 证明期**：`Brown et al. 2020` 与 `Chowdhery et al. 2022` 共同支撑了预训练时代的第一个核心判断：**在自回归语言建模框架下，随着参数、数据与训练系统规模扩大，模型会出现更强的 few-shot 与跨任务泛化能力。** `GPT-3` 的意义在于让 prompt 成为任务接口；`PaLM` 的意义在于说明这条路线在更大训练系统、更多语言与代码场景下仍然成立。
- **compute-optimal 修正期**：`Hoffmann et al. 2022` 并没有推翻 dense scaling，而是修正其粗糙版本。它指出许多早期大模型不是“参数不够大”，而是 **在既定计算预算下 token 训练不足**。因此，本页理解 `Chinchilla` 的正确方式，不是“从大模型转向小模型”，而是从“只扩参数”转向 **参数量与数据量的联合最优配置**。这是预训练叙事里最重要的纠偏节点。
- **能力底座与行为塑形分层期**：`Ouyang et al. 2022` 之所以应在本页中被提及，不是因为它属于预训练，而是因为它为“预训练页的边界”提供了反证。即使 base model 已很强，它仍不会自动变成 helpful、truthful、harmless 的交互系统。这一事实支持一个重要结构判断：**预训练负责通用能力底座，后训练负责行为接口重写。**
- **开放模型家族并行竞争期**：`LLaMA / Llama 2 / Llama 3 / Mistral / Mixtral / Gemma / OLMo 2 / DBRX / OpenELM / Falcon 3 / BLOOM / StarCoder 2 / GLM-130B / Qwen / DeepSeek-V3 / DeepSeek-V4` 等来源共同表明，预训练主线已从闭源演示阶段进入多家族并行推进阶段。这里真正的分化并不只是“是否开源”，而是 **多语言覆盖、代码能力、上下文长度、训练效率、MoE 采用、研究透明度与部署形态** 的组合差异。
- **sparse scaling 与效率导向期**：`Mixtral`、`DBRX`、`DeepSeek-V3 / DeepSeek-V4` 等节点说明，预训练不再只沿 dense Transformer 一条线扩张。MoE 的引入使“总参数规模”与“单 token 激活成本”发生脱钩，预训练讨论因此从“模型有多大”转向“**每单位计算预算能激活多强的有效容量**”。`DeepSeek-V4` 又把这个问题进一步推进到百万 token 上下文下的 attention FLOPs、`KV cache` 成本和深层训练稳定性：`CSA / HCA` 处理混合压缩 attention，`mHC` 处理 residual signal stability，`Muon` 处理大规模训练优化。因此 sparse scaling 已经不只是训练容量问题，也变成了长上下文推理工程与训练稳定性问题。这并没有废除 dense 叙事，但确实改变了后续竞争的工程重点。
- **中国重要家族与全球开放主线交叉期**：`GLM-130B`、`Qwen`、`DeepSeek-V3` 以及 `Kimi` 相关来源说明，中国模型竞争不应被简化为 “Qwen 对其他一切”。其中有些家族以 open-weight 方式参与开放主线，有些则以高影响但非 open-weight 的形式构成重要对照节点。知识组织上，**开放家族主线** 与 **重要非开源对照** 必须区分，否则本页会把“开放性”“影响力”“研究价值”混成一个维度。

如果进一步压缩，本页方法分层可概括为：**dense 能力形成**、**compute-optimal 预算修正**、**开放家族分叉**、**sparse 效率扩张**。这四层共同构成当前 LLM 预训练叙事的稳定骨架。

## 关键争论与分歧

- **更大是否仍是最主要驱动力**：现有证据支持“规模仍然关键”，但不再支持“只扩参数即可”的朴素版本。这个争论真正成立的前提是：区分 **规模本身有效** 与 **规模配置是否合理**。`Chinchilla` 修正的是后者，而不是前者。
- **dense 与 sparse 哪条更代表未来主线**：当前 summary 仍以 dense scaling 为能力讨论的共同语言，但 `Mixtral`、`DBRX`、`DeepSeek-V3` 说明 sparse/MoE 已经成为现实工程路线。现阶段更稳妥的结论不是“dense 被 sparse 替代”，而是：**dense 仍提供主干理论语言，sparse 则在工程竞争中不断扩大实际权重。**
- **预训练与后训练应如何分界**：许多技术报告会把预训练、SFT、RL 一并叙述，尤其是 `DeepSeek-V3 / DeepSeek-V4` 一类综合性报告更容易模糊边界。但只要当前知识库仍把“能力底座”与“行为塑形”视为两阶段结构，就不应把强 post-training 或 agent benchmark 效果反写成预训练规律本身。
- **开放模型是否主要只是分发策略差异**：当前证据不支持这种过窄理解。`BLOOM`、`OLMo 2` 强调研究透明度；`Gemma` 强调 practical size；`OpenELM` 与 `Phi-3` 强调端侧与效率；`Qwen`、`Llama`、`DeepSeek` 强调家族化延展。也就是说，开放模型之间存在真实技术分化，而不仅是 license 分化。
- **应否把 `Kimi` 这类非 open-weight 但高影响家族写入本页**：更稳妥的做法是保留，但明确标注其角色是 **重要对照节点**，而不是“开放模型家族成员”。否则会在主题层面混淆“开放主线”与“行业重要节点”。
- **预训练是否已经足以解释当前模型差异**：随着 agent、多模态与 tool use 路线扩张，单靠预训练已难解释全部产品能力差异。当前证据仍支持本页把预训练当作能力骨干，但也支持一个限制性判断：**预训练已不再独自解释最终系统表现。**

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](../../wiki/summaries/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)
- [Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md)
- [Roziere et al. - 2023 - Code Llama Open Foundation Models for Code](../../wiki/summaries/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md)
- [Scao et al. - 2022 - BLOOM A 176B-Parameter Open-Access Multilingual Language Model](../../wiki/summaries/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md)
- [MosaicML - 2023 - MPT-7B](../../wiki/summaries/MosaicML%20-%202023%20-%20MPT-7B.md)
- [Jiang et al. - 2023 - Mistral 7B](../../wiki/summaries/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md)
- [Jiang et al. - 2024 - Mixtral of Experts](../../wiki/summaries/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md)
- [Team, Google - 2024 - Gemma Open Models Based on Gemini Research and Technology](../../wiki/summaries/Team,%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md)
- [Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size](../../wiki/summaries/Team,%20Google%20DeepMind%20-%202024%20-%20Gemma%202%20Improving%20Open%20Language%20Models%20at%20a%20Practical%20Size.md)
- [Lozhkov et al. - 2024 - StarCoder 2 and The Stack v2 The Next Generation](../../wiki/summaries/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md)
- [Databricks - 2024 - DBRX A Highly Efficient Open LLM](../../wiki/summaries/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md)
- [Mehta et al. - 2024 - OpenELM An Efficient Language Model Family with Open Training and Inference Framework](../../wiki/summaries/Mehta%20et%20al.%20-%202024%20-%20OpenELM%20An%20Efficient%20Language%20Model%20Family%20with%20Open%20Training%20and%20Inference%20Framework.md)
- [Abdin et al. - 2024 - Phi-3 Technical Report A Highly Capable Language Model Locally on Your Phone](../../wiki/summaries/Abdin%20et%20al.%20-%202024%20-%20Phi-3%20Technical%20Report%20A%20Highly%20Capable%20Language%20Model%20Locally%20on%20Your%20Phone.md)
- [Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date](../../wiki/summaries/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md)
- [TII - 2024 - Falcon 3](../../wiki/summaries/TII%20-%202024%20-%20Falcon%203.md)
- [Zeng et al. - 2022 - GLM-130B An Open Bilingual Pre-trained Model](../../wiki/summaries/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md)
- [Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs](../../wiki/summaries/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md)
- [Bai et al. - 2023 - Qwen Technical Report](../../wiki/summaries/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
- [Dubey et al. - 2024 - The Llama 3 Herd of Models](../../wiki/summaries/Dubey%20et%20al.%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)

## 代表页面

- [GPT-3](../concepts/GPT-3.md)
- [PaLM](../concepts/PaLM.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [T5](../concepts/T5.md)
- [Switch Transformer](../concepts/Switch%20Transformer.md)
- [OPT](../concepts/OPT.md)
- [mT5](../concepts/mT5.md)
- [Qwen](../concepts/Qwen.md)
- [Llama](../concepts/Llama.md)
- [LLaMA](../concepts/LLaMA.md)
- [Llama 2](../concepts/Llama%202.md)
- [Code Llama](../concepts/Code%20Llama.md)
- [Llama 3](../concepts/Llama%203.md)
- [BLOOM](../concepts/BLOOM.md)
- [MPT](../concepts/MPT.md)
- [Mistral 7B](../concepts/Mistral%207B.md)
- [Mixtral](../concepts/Mixtral.md)
- [Gemma](../concepts/Gemma.md)
- [Gemma 2](../concepts/Gemma%202.md)
- [Gemma 3](../concepts/Gemma%203.md)
- [StarCoder2](../concepts/StarCoder2.md)
- [DBRX](../concepts/DBRX.md)
- [OpenELM](../concepts/OpenELM.md)
- [Phi-3](../concepts/Phi-3.md)
- [OLMo 2](../concepts/OLMo%202.md)
- [Falcon 3](../concepts/Falcon%203.md)
- [MiniCPM](../concepts/MiniCPM.md)
- [GLM](../concepts/GLM.md)
- [Kimi](../concepts/Kimi.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [DeepSeek-V3](../concepts/DeepSeek-V3.md)
- [DeepSeek-V4](../concepts/DeepSeek-V4.md)
- [Compressed Sparse Attention](../concepts/Compressed%20Sparse%20Attention.md)
- [Heavily Compressed Attention](../concepts/Heavily%20Compressed%20Attention.md)
- [Manifold-Constrained Hyper-Connections](../concepts/Manifold-Constrained%20Hyper-Connections.md)
- [Muon](../concepts/Muon.md)
- [MoE](../concepts/MoE.md)
- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)

## 未解决问题

- 当前知识库已经具备 `dense -> compute-optimal -> 开放家族 -> MoE` 的稳定主线，但 **数据质量、去重策略、长上下文训练代价、数据混配** 仍未形成细粒度 summary 支撑，因此本页对“为什么某些家族更强”的解释仍偏宏观。
- `dense vs MoE` 尚未形成独立 comparison 页；因此 sparse scaling 在本页中仍主要以结构性判断出现，而不是以系统对照结论出现。
- 当前 evidence base 仍不足以对 `Qwen / Llama 3 / DeepSeek-V3 / DeepSeek-V4 / Mistral / Gemma / GLM` 做高置信统一排序，本页只能稳定讨论 **路线差异**，不能稳定讨论 **家族优劣总排名**。
- `tool use、多模态、agent` 是否应被视为预训练主干的自然外推，还是更应归于后训练与系统整合，目前仍需要更多 `wiki/summaries/` 支撑；因此本页暂不把这些能力写成预训练本身的必然结果。
- `ChatGLM / GLM-4 / Gemma 3 / Phi` 后续代际，以及 `Kimi` 全家族节点，目前仍未形成更闭合的时间线与 comparison 层支撑，这限制了本页对“开放家族长期演进格局”的判断强度。

## 关联页面

- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)
