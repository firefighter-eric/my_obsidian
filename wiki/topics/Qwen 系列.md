# Qwen 系列

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 **Qwen 家族从 Qwen1 到 Qwen3.5 的技术主线、模态分叉与系统角色演进**。它不是单篇论文摘要，也不是所有 `Qwen-*` 型号的罗列页，而是要回答一个更高阶的问题：**一个开放模型家族如何从中文/多语言 LLM，演进为覆盖 VL、Omni、图像生成与 native multimodal agent 的系统谱系。**

与 [LLM 预训练](./LLM%20预训练.md) 相比，本页聚焦单一家族内部的连续演化；与视觉或多模态 topic 相比，本页只在 **Qwen 家族框架** 内讨论视觉、音频、图像生成与 agent 化，而不把这些能力的一般方法史全部搬入。也就是说，本页的中心不是“某项能力本身”，而是“这些能力如何被组织进同一个家族工程学”。

当前 evidence base 支持一个相当清晰的判断：**Qwen 的主线不是简单的“参数越来越大”，而是模型在系统中的角色不断升级。** Qwen1 到 Qwen2.5 的主轴仍是“以 LLM 为骨架扩展代码、数学、多语言与长上下文能力”；Qwen2-VL、Qwen2.5-VL、Qwen2.5-Omni 和 `Qwen-Image` 则把感知、生成与交互能力并入家族；到 Qwen3 与 Qwen3.5，家族叙事开始围绕 **thinking、RL、MCP、环境交互与 native multimodal agent** 重新组织。

## 核心问题

- Qwen 家族从 Qwen1 到 Qwen3.5 的代际升级，核心究竟来自 **参数规模、数据规模、训练范式**，还是 **模型在系统中的角色重写**。
- 纯文本 LLM 主线、VL 分支、Omni 分支与图像生成分支分别解决什么问题，它们在家族里是附属分支还是已经构成主干。
- Qwen3 与 Qwen3.5 的 `thinking / RL / agent / native multimodal` 转向，与 Qwen2.x 时代的逻辑差异究竟体现在哪里。
- 家族内部的复用关系如何影响代际演进，例如前一代模型是否开始为后一代制造训练数据、感知能力与行为脚手架。
- 哪些判断来自来源直接支持，哪些是基于多篇 summary 的结构性归纳，哪些仍应保持不确定。

## 主线脉络 / 方法分层

本页按 **家族角色与能力接口** 分层，而不是仅按版本号罗列。因为对 Qwen 的理解，关键不是记住“出过哪些模型”，而是把这些模型如何共同构成一个持续扩张的开放家族写清楚。

- **LLM 底座建立期**：`Qwen Technical Report` 所对应的 Qwen1 已经不是单一 base model，而是 **基础模型 + chat 对齐 + 代码 / 数学分支 + tool use / agent 评测** 的完整起点。其真正意义在于建立 Qwen 的家族工程学：多语言 tokenizer、`3T` 级预训练、对齐流程、长上下文外推、工具调用与评测体系被组织为同一套可持续迭代的框架。
- **家族化与部署友好期**：`Qwen1.5` 把这一框架从“研究起点”推进为“可广泛分发的开放家族”。这一代强调 `0.5B` 到 `110B` 的尺寸覆盖、`32K` 上下文、Hugging Face 原生支持与部署兼容性，说明重点不只是性能提升，而是 **让 Qwen 成为一条稳定可用的开放模型产品线**。
- **多语言与后训练体系成熟期**：`Qwen2` 的变化不只是参数和基准分数，而是把多语言覆盖、`128K` 上下文、全尺寸 `GQA`、系统化 post-training 与正式 `MoE` 节点整合进主线。从这一代开始，更稳妥的说法是：Qwen 已从“中文强模型”升级为 **全球开放家族竞争者**，而不应继续只以中文优势来概括。
- **通用 LLM 强化与任务外溢期**：`Qwen2.5-LLM` 把预训练扩到 `18T` token，并明显强调代码、数学、长文本、结构化输出与 JSON 友好性。这里的关键不是“一模全能”，而是 **先把通用 LLM 底座做到足够强，使 coder、math、agent 等路线可以从同一底座外溢**。这也是 Qwen 家族能持续扩张而不彻底碎片化的重要前提。
- **VL 感知层系统化期**：`Qwen2-VL` 与 `Qwen2.5-VL` 把视觉路线从图文问答推进为结构化多模态感知系统。`Qwen2-VL` 通过动态分辨率与 `M-ROPE` 解决多模态 token 化与位置编码；`Qwen2.5-VL` 则进一步引入 native dynamic-resolution ViT、绝对坐标 grounding、文档 omni-parsing、动态 FPS、绝对时间建模与 GUI agent 数据。基于现有 summary，更稳妥的判断是：**VL 在 Qwen 中不是附属展示分支，而是后续 agent 化的感知前置层。**
- **Omni 输入输出统一期**：`Qwen2.5-Omni` 把文本、图像、音频、视频与语音输出统一进一个端到端框架，`Thinker-Talker` 与 `TMRoPE` 表明其目标已从“多模态理解”扩展到 **实时交互与多模态生成统一**。到 `Qwen3.5-Omni`，这条路线进一步变成系列化、`256K` 上下文、长音频处理与 `Hybrid-Attention MoE` 架构，说明 omni 已从示范模型升级为正式产品线。
- **图像生成与可编辑性支线**：`Qwen-Image` 说明 Qwen 的多模态扩张不只停留在理解型模型，而是进入 image generation 与创作工作流。其重点在 `20B MMDiT`、复杂文本渲染、双语排版与精确编辑；进一步到 `Qwen-Image-Layered`，又把图像生成推进到 **多层 `RGBA` 表示与可编辑图层工作流**。因此，Qwen 的图像支线并不是简单补齐“会画图”，而是在尝试把生成能力与编辑接口合并。
- **Reasoning 与 agent 主轴重写期**：`Qwen3` 是家族叙事真正的转折点。它不再只讲“更强 LLM”，而把 `hybrid thinking`、reasoning RL、agentic capabilities、MCP 能力与 `36T` 级预训练放在代际中心。更关键的是，Qwen3 明确复用 `Qwen2.5-VL`、`Qwen2.5`、`Qwen2.5-Math`、`Qwen2.5-Coder` 生成高质量合成数据，这意味着家族内部开始形成 **前代模型为后代模型提供训练资产** 的闭环。
- **Native multimodal agent 期**：`Qwen3.5` 把这种转向进一步写成 “Towards Native Multimodal Agents”。截至 `2026-04-18` 语境下知识库现有 summary 所覆盖的官方材料，它已不再被简单表述为“文本模型外挂视觉”，而是 `native vision-language model`，并在 RL 环境扩展、异步强化学习框架、混合注意力 `MoE`、多模态训练基础设施等层面做系统性重写。基于现有来源，更稳妥的判断是：**Qwen3.5 标志着家族重心从“模型能力集合”转向“可在真实环境中行动的多模态 agent 基座”。**

## 关键争论与分歧

- **Qwen 的核心身份是否仍是 LLM 家族**：从 Qwen1 到 Qwen3，LLM 仍是家族骨架；但到 Qwen3.5，官方叙事已明显转向 `native multimodal agent`。因此当前最稳妥的说法不是“Qwen 已不再是 LLM”，而是：**Qwen 以 LLM 起家，但其稳定身份正在向 multimodal agent family 演化。**
- **VL 与 Omni 是否只是附属分支**：现有证据更支持“不是”。尤其 `Qwen2.5-VL` 的文档理解、grounding、GUI 感知与视频时序能力，已经承担 agent 感知层角色，而非仅是可选插件。
- **图像生成是否已经构成 Qwen 主干**：`Qwen-Image` 与 `Qwen-Image-Layered` 足以说明图像生成已成为重要分支；但当前关于 `Qwen-Image` 的证据仍更接近官方博客与技术摘要，而非完整技术报告。因此还不能把它与 LLM / VL / Omni 完全等强地写成“已成熟主轴”。
- **Qwen3 的关键是否只是“thinking 更强”**：当前 evidence 更支持把 Qwen3 的核心变化理解为 **后训练目标从回答优化转向行为优化**。`hybrid thinking` 只是表层表现，更深层的是 RL、工具使用与 agent workflow 成为代际中心。
- **Qwen3.5 是否应继续放在传统 LLM 叙事中**：基于现有 summary，Qwen3.5 仍属于大模型家族，但已不宜再简单概括为“Qwen 的下一代纯文本 LLM”。如果继续沿传统 LLM 页面的写法处理，会遮蔽其 agent 化与 native multimodal 的结构变化。
- **开放权重主线与 API 节点如何组织**：Qwen2.5 之后，家族同时存在开放权重与 `Qwen-Plus / Turbo / Max / 3.5-Plus` 等 API 节点。当前 topic 只将有稳定 summary 支撑的主干节点纳入主线，这是必要约束；否则页面会退化成产品 SKU 列表。
- **官方指标与真实长期能力边界如何区分**：Qwen3.5、Qwen2.5-VL 与 `Qwen-Image` 都展示了强指标或强产品叙事，但 topic 层仍必须区分 **来源直接陈述** 与 **编者基于多篇 summary 的结构归纳**。当前可以稳健讨论路线重心变化，但不能把所有愿景叙事直接写成已证实终局。

## 证据基础

- [Bai et al. - 2023 - Qwen Technical Report](../../wiki/summaries/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
- [Qwen Team - 2024 - Introducing Qwen1.5](../../wiki/summaries/Qwen%20Team%20-%202024%20-%20Introducing%20Qwen1.5.md)
- [Qwen Team - 2024 - Hello Qwen2](../../wiki/summaries/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)
- [Qwen Team - 2024 - Qwen2.5-LLM Extending the boundary of LLMs](../../wiki/summaries/Qwen%20Team%20-%202024%20-%20Qwen2.5-LLM%20Extending%20the%20boundary%20of%20LLMs.md)
- [Qwen Team - 2025 - Qwen3 Think Deeper Act Faster](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
- [Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents](../../wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)
- [Qwen Team - 2024 - Qwen2-VL](../../wiki/summaries/Qwen%20Team%20-%202024%20-%20Qwen2-VL.md)
- [Bai et al. - 2025 - Qwen2.5-VL Technical Report](../../wiki/summaries/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md)
- [Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md)
- [Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](../../wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)
- [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)
- [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)

## 代表页面

- [Qwen](../concepts/Qwen.md)
- [Qwen1.5](../concepts/Qwen1.5.md)
- [Qwen2](../concepts/Qwen2.md)
- [Qwen2-VL](../concepts/Qwen2-VL.md)
- [Qwen2.5](../concepts/Qwen2.5.md)
- [Qwen2.5-VL](../concepts/Qwen2.5-VL.md)
- [Qwen2.5-Omni](../concepts/Qwen2.5-Omni.md)
- [Qwen-Image](../concepts/Qwen-Image.md)
- [Qwen-Image-Layered](../concepts/Qwen-Image-Layered.md)
- [Qwen3](../concepts/Qwen3.md)
- [Qwen3.5](../concepts/Qwen3.5.md)
- [Qwen3.5-Omni](../concepts/Qwen3.5-Omni.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)

## 未解决问题

- 当前知识库虽已覆盖 `Qwen3` 与 `Qwen3.5` 的官方材料，但两者仍缺更完整的 technical report 级 summary；因此 **hybrid thinking、general RL、native multimodal agent** 的训练细节还未达到完全论文级可追溯度。
- `Qwen-VL` 早期节点、`Qwen2-Audio` 及其与 `Qwen2.5-Omni` 的衔接尚未系统纳入，这使 **VL -> Omni** 的中间演进链仍不够闭合。
- `Qwen-Image` 当前证据仍偏向官方博客与摘要，尚不足以完整解释其训练路线、长期开源策略以及它与理解型多模态主线的长期关系；因此图像生成支线目前仍应保留一定不确定性。
- `Qwen-Image-Layered` 已把图像分解与多层 `RGBA` 工作流纳入 Qwen 图像支线，但当前仍缺 `Qwen-Image` 与 `Qwen-Image-Layered` 的家族内 comparison 页，二者关系尚未被系统固定。
- API 型号如 `Qwen-Plus / Turbo / Max / 3.5-Plus / 3.5-Max` 与开放权重主线之间的分工，目前仍缺少独立 comparison 页承接；因此本页暂不对商业 SKU 层做更强结论。
- 当前 topic 已能较稳地给出“Qwen 正从 LLM 家族走向 multimodal agent family”的结构判断，但这仍是基于现有来源的归纳，而非被完整技术报告完全坐实的终局结论。

## 关联页面

- [LLM预训练](./LLM%E9%A2%84%E8%AE%AD.md)
- [Qwen](../concepts/Qwen.md)
- [Qwen-Image](../concepts/Qwen-Image.md)
- [Qwen-Image-Layered](../concepts/Qwen-Image-Layered.md)
- [Qwen2.5-VL](../concepts/Qwen2.5-VL.md)
- [扩散模型与文生图](./%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [图像分层 layered](./%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)
