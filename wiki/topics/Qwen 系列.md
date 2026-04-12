# Qwen 系列

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页讨论 Qwen 家族从 Qwen1 到 Qwen3.5 的技术主线及其模态分叉。这里关心的不是单篇论文摘要，而是一个开放模型家族如何从中文/多语言 LLM，演进为同时覆盖 VL、Omni、图像生成与 native multimodal agent 的系统族谱。与 `LLM预训练` 相比，本页聚焦单一家族内部脉络；与 `传统CV` 相比，本页只把视觉/音频能力放在 Qwen 家族演进框架内讨论。

Qwen 的主线并不是简单的“参数越来越大”。更准确的理解是：它先把开放 LLM 的基础设施做成稳定家族，再逐步把文档理解、视觉 grounding、音视频输入、语音输出、工具调用和 agent 训练并入同一个产品与研究体系。编者基于当前来源做出的总体判断是，Qwen1 到 Qwen2.5 的主轴仍是“以 LLM 为骨架的能力外延”，而 Qwen3 到 Qwen3.5 则开始转向“以 reasoning / RL / agent workflow 为中心重新组织整个家族”。

进一步说，Qwen 的独特性不只在于它同时做语言、多模态和 agent，而在于这些路线彼此复用。Qwen3 的合成数据直接复用 Qwen2.5-VL、Qwen2.5、Qwen2.5-Math 与 Qwen2.5-Coder；Qwen2.5-VL 则把文档、OCR、GUI grounding 和视频理解变成后续 native multimodal agent 的感知基础。这使 Qwen 看起来不像若干独立模型的拼盘，而更像一个围绕数据生产、后训练和部署形态持续重构的开放模型家族。

## 核心问题

- Qwen 家族从 Qwen1 到 Qwen3.5 的升级，核心究竟是参数规模、数据规模、训练范式，还是“系统角色”的变化。
- 纯文本 LLM 主线、VL 分支、Omni 分支与 image generation 分支分别解决什么问题，它们在家族中是附属能力还是主干路线。
- Qwen3 与 Qwen3.5 的“thinking / RL / agent / native multimodal”转向，与 Qwen2.x 时代的底层逻辑有何本质差异。
- 哪些判断来自来源直接支持，哪些是编者基于多篇 summary 归纳出的结构性结论，哪些仍应保留不确定。

## 主线脉络 / 方法分层

- LLM 底座建立期：`Qwen Technical Report` 对应的 Qwen1 已经不是单点 base model，而是“基础模型 + SFT/RLHF chat + 代码 / 数学专门模型 + tool use/agent 评测”的完整框架。其关键意义在于建立了 Qwen 的家族工程学：多语言 tokenizer、3T 级预训练、对齐流程、长上下文外推、代码与工具调用都被视为同一产品体系的一部分。
- 家族化与部署友好期：`Qwen1.5` 把这条路线从“研究起点”推进到“可用家族”。官方强调从 `0.5B` 到 `110B` 的尺寸覆盖、`32K` 上下文、Hugging Face 原生支持和部署兼容性，说明这一代的重点是把 Qwen 从一篇技术报告升级为可大规模分发的开放模型线。
- 多语言与后训练体系成熟期：`Qwen2` 的本质变化不只是更大，而是把多语言覆盖、`128K` 上下文、全尺寸 `GQA`、可扩展 post-training 与 `57B-A14B` 级别的正式 MoE 节点整合进主线。编者判断：从这一代开始，Qwen 已从“中文强模型”转为“面向全球、多尺寸、可持续后训练”的开放家族竞争者。
- 通用 LLM 强化与任务外溢期：`Qwen2.5` 通过 `0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B` 补齐部署区间，把预训练扩到 `18T` token，并显著强调代码、数学、长文本、结构化输出与 JSON 友好性。这里的关键不是“所有能力都在一个模型里解决”，而是先把通用 LLM 底座做得足够强，使 coder、math、agent 等路线都能从同一底座外溢。
- VL 路线系统化期：`Qwen2-VL` 与 `Qwen2.5-VL` 把 Qwen 的视觉能力从图文问答推进为结构化多模态感知系统。`Qwen2-VL` 通过动态分辨率和 `M-ROPE` 解决多模态 token 化与位置编码问题；`Qwen2.5-VL` 则进一步把 native dynamic-resolution ViT、绝对坐标 grounding、文档 omni-parsing、动态 FPS、绝对时间建模与 GUI agent 数据纳入统一训练。编者归纳：VL 路线在 Qwen 家族中的真实角色不是“附加视觉能力”，而是后续 agent 化感知层。
- Omni 路线统一输入输出期：`Qwen2.5-Omni` 把文本、图像、音频、视频与语音输出统一进端到端模型，`Thinker-Talker` 与 `TMRoPE` 说明其目标已不是仅做多模态理解，而是做可实时交互的统一感知与生成系统。到 `Qwen3.5-Omni`，这条路线又被扩展为系列化、`256K` 上下文、长音频处理和 `Hybrid-Attention MoE` 架构，说明 omni 不再是单个示范模型，而是正式产品线。
- 图像生成支线：`Qwen-Image` 说明 Qwen 的多模态战略并不只限于“理解型多模态”。这条线聚焦 `20B MMDiT`、复杂文本渲染、双语排版和精确编辑，更接近生产设计与视觉创作基础模型。它目前仍是支线而非家族主轴，但它反映出 Qwen 正在把“模态覆盖”扩展到生成型视觉。
- Reasoning 与 agent 主轴重写期：`Qwen3` 是真正的转折点。它不再只讲更强 LLM，而是把 `hybrid thinking`、reasoning RL、agentic capabilities、MCP 使用能力和 `36T` 级预训练放到代际中心。尤其值得注意的是，Qwen3 已显式复用 Qwen2.5-VL 与 Qwen2.5 系列生成高质量合成数据，这说明家族各分支开始反向服务主干模型。
- Native multimodal agent 期：`Qwen3.5` 把这种转向明确写成“Towards Native Multimodal Agents”。截至 `2026-04-12` 的官方正文与摘要，Qwen3.5 已不再被表述为“文本模型外挂视觉”，而是 `native vision-language model`，并在 RL 环境扩展、异步强化学习框架、混合注意力 MoE、高效多模态训练基础设施上做系统级重写。编者判断：Qwen3.5 标志着家族重心从“模型能力集合”转向“可在真实环境中长期行动的多模态 agent 基座”。

## 深入分析

### 1. Qwen 的真实主轴是“系统角色”升级，而不是单纯代际升级

如果只按代际看，Qwen 像是在重复行业常见模式：更多数据、更长上下文、更大模型。但把多个来源并在一起看，更清晰的结论是：Qwen 每一阶段都在重写模型在系统中的角色。Qwen1 是“可对齐、可工具化的开放 LLM”；Qwen2.5-VL 是“能读文档、看界面、做 grounding 的感知系统”；Qwen2.5-Omni 是“统一输入输出接口”；Qwen3/3.5 则是“以 RL 和环境交互为中心的智能体底座”。因此，Qwen 家族最重要的连续性，不是参数表，而是它不断把更多工作流原生化进模型。

### 2. VL 在 Qwen 中不是边缘分支，而是 agent 化的感知前置层

这一点在浅层 topic 中常被忽略。若只看命名，容易以为 `Qwen2-VL` 和 `Qwen2.5-VL` 只是语言主干旁边的视觉支线；但从 `Qwen2.5-VL Technical Report` 看，它们承担的是文档解析、OCR、绝对坐标 grounding、长视频时间理解、GUI 交互感知等任务，而这些能力恰好构成 agent 在真实环境中行动前必须具备的感知与定位基础。换言之，Qwen3.5 所谓 native multimodal agent，并不是凭空冒出来的，它很大程度上是把 VL 路线中已经成熟的感知机制，与 Qwen3 的 RL/agent 主线重新整合。

### 3. Qwen3 的关键不只是“thinking”，而是把后训练从回答优化改成行为优化

Qwen1 和 Qwen2 的后训练重点，仍主要是 instruction following、helpfulness 与 alignment。到 Qwen3，官方开始把四阶段 post-training、reasoning RL、thinking/non-thinking 融合、general RL、MCP 能力放在核心卖点位置。这里真正重要的变化不是模型会不会输出更长 CoT，而是后训练目标从“如何更好回答”转向“如何根据任务难度、工具和环境约束选择行为方式”。这也是 Qwen3 与 Qwen2.5 的代际差异所在。

### 4. Qwen3.5 的核心竞争点是基础设施与环境扩展，而不只是模型指标

Qwen3.5 正文中大量篇幅讨论 RL environment scaling、异步强化学习框架、训推分离、FP8、多模态训练吞吐和百万级 agent 脚手架扩展。这些内容说明 Qwen3.5 的目标不是单纯发布一个更强的 benchmark model，而是建立能持续扩展 agent 行为的训练与部署基础设施。编者归纳：如果 Qwen2.5 是“把多模态能力做全”，Qwen3.5 更像是“把多模态 agent 的生产体系做出来”。

### 5. Qwen 家族的一个隐性优势是内部数据与模型复用

`Qwen3` 明确使用 `Qwen2.5-VL` 处理 PDF、用 `Qwen2.5 / Math / Coder` 生成高质量合成数据；`Qwen2.5-VL` 又把文档、视频和 GUI 交互数据沉淀为更高价值的多模态训练资产。这种“前一代模型为下一代制造更高质量训练数据”的闭环，使 Qwen 更像一个不断自增强的数据生产系统，而不是彼此割裂的产品发布。若这一机制持续存在，Qwen 的代际跃迁可能会越来越依赖内部流水线质量，而不只是公开参数规模。

## 关键争论与分歧

- 家族核心是否仍是 LLM：从 Qwen1 到 Qwen3，LLM 仍是骨架；但到 Qwen3.5，官方已把首个 open-weight 代表模型直接写成 `native vision-language model`。因此更稳妥的说法是，Qwen 的起点是 LLM 家族，但其 3.5 代后的稳定身份可能更接近 multimodal agent family。
- VL 与 Omni 是否只是附属分支：现有证据更支持“不是”。尤其 `Qwen2.5-VL` 的文档、grounding、GUI agent 数据路线，已经在功能上承担了 agent 感知层角色，而非纯展示分支。
- 图像生成是否属于 Qwen 主干：`Qwen-Image` 明确证明 Qwen 正进入 image generation 赛道，但当前只有官方博客摘要，缺完整技术报告。把它写成“已与 VL / Omni 并列的家族主轴”还过早。
- Qwen3.5 是否应继续归入传统 LLM 叙事：基于 `2026-02-16` 官方正文与 `2026-03-29` 官方摘要，当前更稳妥的判断是：Qwen3.5 仍属于大模型家族，但不宜再简单写成“Qwen 的下一代纯文本 LLM”。
- 开放权重与 API 节点如何组织：Qwen2.5 之后，家族同时存在开源权重模型与 `Qwen-Plus / Turbo / Max / 3.5-Plus` 等 API 节点。当前 topic 只把有稳定 summary 支撑的主干节点写入主线，不把所有 API SKU 都视为独立研究代际。
- 官方指标与真实能力边界如何区分：Qwen3.5 和 Qwen2.5-VL 都展示了很强 benchmark 结果，但 topic 写作中仍应区分“官方直接宣称的性能”与“编者基于训练路线做出的结构性判断”，避免把愿景叙事直接写成已证实结论。

## 证据基础

- LLM 主线：
  - [Bai et al. - 2023 - Qwen Technical Report](../../raw/summary/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
  - [Qwen Team - 2024 - Introducing Qwen1.5](../../raw/summary/Qwen%20Team%20-%202024%20-%20Introducing%20Qwen1.5.md)
  - [Qwen Team - 2024 - Hello Qwen2](../../raw/summary/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)
  - [Qwen Team - 2024 - Qwen2.5-LLM Extending the boundary of LLMs](../../raw/summary/Qwen%20Team%20-%202024%20-%20Qwen2.5-LLM%20Extending%20the%20boundary%20of%20LLMs.md)
  - [Qwen Team - 2025 - Qwen3 Think Deeper Act Faster](../../raw/summary/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
  - [Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents](../../raw/summary/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)
- VL / 感知路线：
  - [Qwen Team - 2024 - Qwen2-VL](../../raw/summary/Qwen%20Team%20-%202024%20-%20Qwen2-VL.md)
  - [Bai et al. - 2025 - Qwen2.5-VL Technical Report](../../raw/summary/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md)
- Omni / 原生多模态路线：
  - [Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All](../../raw/summary/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md)
  - [Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](../../raw/summary/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)
- 生成型视觉支线：
  - [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](../../raw/summary/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)

## 代表页面

- [Qwen](../concepts/Qwen.md)
- [Qwen1.5](../concepts/Qwen1.5.md)
- [Qwen2](../concepts/Qwen2.md)
- [Qwen2-VL](../concepts/Qwen2-VL.md)
- [Qwen2.5](../concepts/Qwen2.5.md)
- [Qwen2.5-VL](../concepts/Qwen2.5-VL.md)
- [Qwen2.5-Omni](../concepts/Qwen2.5-Omni.md)
- [Qwen-Image](../concepts/Qwen-Image.md)
- [Qwen3](../concepts/Qwen3.md)
- [Qwen3.5](../concepts/Qwen3.5.md)
- [Qwen3.5-Omni](../concepts/Qwen3.5-Omni.md)

## 未解决问题

- 当前知识库虽已补到 `Qwen3.5` 正文，但 `Qwen3` 与 `Qwen3.5` 仍缺独立 technical report summary，因此 hybrid thinking、general RL、native multimodal agent 的训练细节还未达到论文级可追溯度。
- `Qwen-VL` 早期节点、`Qwen2-Audio` 及其与 `Qwen2.5-Omni` 的衔接尚未系统纳入，这使 VL 到 Omni 的中间演进仍不够闭合。
- `Qwen-Image` 当前只有官方博客摘要，尚不足以完整说明其训练路线、开源策略以及它与家族理解型多模态主线的长期关系。
- API 型号如 `Qwen-Plus / Turbo / Max / 3.5-Plus / 3.5-Max` 与开放权重主线之间的分工，目前仍缺少独立 comparison 页承接。
- 当前 topic 已能给出结构性判断，但“Qwen 是否正在从 LLM 家族转为 agent family”仍属于基于现有来源的归纳，而非已被完整技术报告彻底坐实的终局结论。

## 关联页面

- [LLM预训练](./LLM%E9%A2%84%E8%AE%AD.md)
- [LLM 基础脉络](./LLM%20%E5%9F%BA%E7%A1%80%E8%84%89%E7%BB%9C.md)
- [Qwen](../concepts/Qwen.md)
- [Qwen-Image](../concepts/Qwen-Image.md)
- [Qwen2.5-VL](../concepts/Qwen2.5-VL.md)
- [扩散模型与文生图](./%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
