# LLM 预训练

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论语言模型在 post-training 之前如何通过大规模预训练获得通用能力。这里的重点是预训练目标、规模化规律、模型家族与训练工程，而不是 RLHF、DPO 等行为对齐方法。当前知识库中，LLM 的骨干叙事也统一收敛到本页：大规模预训练如何带来通用能力、compute-optimal 视角如何修正“只扩参数”的早期叙述，以及为什么后训练应被视为独立于能力底座的下一阶段。

## 核心问题

- 通用语言能力主要如何从大规模预训练中获得。
- dense scaling、compute-optimal 修正与 sparse scaling 之间的关系是什么。
- 开放模型家族的技术差异主要体现在哪些训练与架构选择上。
- 预训练阶段的能力边界与后训练阶段的行为改写边界应如何切分。

## 主线脉络 / 方法分层

- dense scaling 主线：`Brown et al. 2020` 与 `Chowdhery et al. 2022` 支撑了“继续扩大模型与训练系统能持续提升 few-shot 能力”这一方向，GPT-3 与 PaLM 是其中最典型的 dense scaling 代表。
- compute-optimal 修正：`Hoffmann et al. 2022` 把讨论从“能否继续变大”转向“在固定 FLOPs 下如何平衡参数量与 token 数”，Chinchilla 因此成为 dense scaling 的关键纠偏点。
- 能力形成与行为塑形的阶段切分：当前证据更支持把 LLM 主线拆成“预训练获得通用能力”与“post-training 改写交互行为”两个阶段。`Ouyang et al. 2022` 之所以重要，不是因为它改变了预训练规律，而是因为它说明 strong base model 仍不足以自动变成更 helpful、truthful、harmless 的交互系统。
- 开放模型家族阶段：`Touvron et al. 2023`、`Scao et al. 2022`、`Jiang et al. 2023/2024`、`Team, Google 2024`、`Databricks 2024`、`Ai2 2024`、`Bai et al. 2023`、`Dubey et al. 2024` 和 `Unknown 2024 DeepSeek-V3` 说明预训练主线已从闭源标杆转向多个开放家族并行推进。这里的分化不只体现在“谁开源”，还体现在多语言、代码、`MoE`、小模型、本地部署、fully open 研究透明度与平台工程化等不同方向。
- 中国重要家族对照节点：`GLM-130B` 说明中文语境中的重要家族并不只有 `Qwen`；而 `Kimi k1.5` 则提醒我们，中国模型竞争中也存在高影响但非 `open-weight` 的重要家族。知识组织上应把“开放模型主线”与“重要非开源对照节点”区分开来，而不是统称为开源模型。
- sparse scaling 与高效训练：`DeepSeek-V3` 与 `MoE` 路线表明预训练不再只沿着 dense Transformer 扩张，而是转向“总参数更大、单 token 激活更少”的效率设计。

## 关键争论与分歧

- 更大是否仍是最主要驱动力：当前证据表明规模仍重要，但 `Hoffmann 2022` 已经否定了“只扩参数即可”的简单叙述。
- dense 与 sparse 哪条更代表未来主线：从现有 summary 看，dense scaling 仍是能力讨论的基础语言，但 sparse/MoE 已经成为工程与开源竞争中的现实路线。
- 预训练与后训练如何分界：`DeepSeek-V3` 这类报告同时覆盖预训练、SFT 与 RL，说明现实系统往往跨阶段联合报告，但知识组织上仍需把“能力底座”与“行为对齐”分开。
- LLM 主线是否应只写预训练：当前页承担的是“能力骨干”的总入口，因此需要保留从 GPT-3 到 Chinchilla、再到“为何 post-training 成为独立阶段”的连续叙事；但具体对齐与 RL 方法仍应下沉到 `指令对齐与 post-training` 与 `LLM RL`。
- 开放模型是否主要是发布策略差异：当前证据不足以把开放模型仅视为分发差异，它们在多语言、工具支持、上下文长度与训练效率上也形成了真实技术分化。
- 应否把 `Kimi` 这类重要 API 家族写入本页：当前更稳妥的做法是写，但必须明确其不是 `open-weight` 家族，否则会把“开放模型主线”和“中国重要家族对照”混成一类。

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
- [DeepSeek-V3](../concepts/DeepSeek-V3.md)
- [MoE](../concepts/MoE.md)
- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)

## 未解决问题

- 当前知识库尚未建立 `dense vs MoE` 的独立比较页，因此 sparse scaling 的优劣仍只停留在主题级概述。
- 预训练数据质量、token 去重、长上下文训练成本等问题，在现有 summary 中仍未形成细粒度证据链。
- Qwen、Llama 3、DeepSeek-V3、Mistral、Gemma、GLM 之间更系统的横向比较仍有待 comparison 层继续细化；当前只新增了一页开放性与家族角色对照页。
- 预训练骨干与 post-training 分界目前已有稳定主线，但“tool use、多模态与 agent 化”是否应被视为同一主干的自然延展，仍需要更多 summary 支撑。
- `ChatGLM / GLM-4`、`Gemma 3`、`Phi` 后续代际与 `Kimi` 全家族节点，目前仍未形成更完整的时间线与家族级 topic。

## 关联页面

- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)
