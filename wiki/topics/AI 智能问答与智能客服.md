# AI 智能问答与智能客服

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 **AI 驱动的智能问答与智能客服系统**：用户围绕产品、政策、订单、故障、售后和流程发起咨询，系统需要在多轮交互中完成意图理解、知识召回、回复生成、风险控制，以及在必要时触发工具调用或人工转接。这个主题的核心不是“模型能否把一句话答得通顺”，而是 **企业服务场景中的问答系统如何成为可控、可追溯、可运营的服务接口**。

它与 [传统 NLP](./传统%20NLP.md) 的边界在于：后者讨论检索、编码器、句向量等基础方法谱系，而本页讨论这些能力如何被组织成客服链路。它与 [搜索排序](./搜索排序.md) 的边界在于：排序是客服系统中的一个中间层，本页关心的是从召回到回答再到执行的完整服务回路。它与 [指令对齐与 post-training](./指令对齐与%20post-training.md) 及 [LLM RL](./LLM%20RL.md) 的边界在于：对齐与偏好优化在这里被视为客服系统的行为约束层，而不是独立算法终点。

因此，这个 topic 不应被理解为“客服版 LLM 应用合集”。更准确地说，它是一个 **由 FAQ 检索、对话状态建模、知识增强生成、工具调用、偏好对齐与安全护栏共同构成的系统问题域**。只讨论其中任一子模块，都不足以支撑“智能客服已经成立”的结论。

## 核心问题

- **高频标准问答、长尾知识问答与事务型请求的边界如何划分**：哪些问题适合 FAQ 检索，哪些需要 RAG，哪些本质上已经是工具执行而不是问答。
- **多轮对话中的有效上下文如何选择**：客服历史轮次并非越长越好，系统需要判断哪些历史信息构成任务状态，哪些只是噪声或礼貌性过渡。
- **企业知识如何进入模型**：通用 LLM 的语言能力并不自动等于企业知识能力，知识接入方式决定了正确性、可解释性与更新成本。
- **客服行为如何被约束**：礼貌、拒答、澄清、风险规避、工具调用和人工升级并非同一目标，需要不同层次的行为控制。
- **客服系统到底如何评测**：离线准确率、人工偏好和线上业务指标之间并不天然一致，当前很多“效果提升”只覆盖了局部能力。

## 主线脉络 / 方法分层

- **FAQ 检索底座层**：`Sakata et al. 2019` 把客服首先建模为 FAQ pair retrieval，并同时考虑 `query-question similarity` 与 `query-answer relevance`。这说明客服最早、也最稳定的一层并不是自由生成，而是 **把用户问题映射到已有标准答案**。这一路线对高频、规范、答案边界清晰的问题最有效，因为它天然可审计、可回放、可复核。它的局限同样明确：一旦问题跨多个文档、涉及条件组合、账户状态或异常情形，FAQ 匹配就会迅速退化。
- **对话状态建模层**：`Vlasov et al. 2019` 的 Dialogue Transformers 代表“客服不是单轮检索”的证据。客服对话里经常出现补充说明、上下文修正、用户情绪波动和主题切换，简单把所有历史文本拼接起来并不能稳定表示真正的任务状态。这个层次解决的是 **历史轮次的结构化编码问题**，即模型如何知道“哪一句历史还有效”。但这层只解决对话状态，不自动提供事实依据，也不能替代企业知识接入。
- **语义检索与领域匹配层**：`Karpukhin et al. 2020` 的 DPR 和 `Oğuz et al. 2021` 的 domain-matched retrieval 共同把客服系统推进到 **“检索不再依赖词项重合，而依赖语义表征与领域数据匹配”** 的阶段。它们的重要性不只在于性能提升，而在于改变了客服知识入口的基本假设：用户会使用口语、错词、泛称和跨文档问题描述，因此企业知识召回必须先解决“说法不一样但需求相同”的问题。与此同时，`Oğuz et al. 2021` 又提醒，客服检索不是拿一个通用 embedding 就能解决，领域语料和任务匹配仍然决定上限。
- **知识增强问答层**：`Wang - PIKE-RAG` 代表客服从“找到答案”走向“组织答案”的转折。复杂客服场景中的难点，常常不是缺某一条知识，而是缺 **对多条知识进行拆解、组合、解释和约束的能力**。例如退款规则、异常处理或多条件政策判断，往往要求系统先抽取专门知识，再按任务步骤组织 rationale。也正因此，RAG 在客服里真正解决的不是“让回答更像有依据”，而是 **让复杂服务问题从单文档回忆转成可解释的知识编排问题**。
- **指令泛化层**：`Wei et al. 2021` 和 `Iyer et al. 2022` 说明，客服系统不能只依赖固定 intent schema。真实客服涉及解释、安抚、澄清、重述、总结、转接、礼貌拒答和流程引导，这些行为接口更接近 instruction following，而不是传统 FAQ 分类。这里的关键判断是：**instruction tuning 解决的是任务接口统一问题，而不是事实正确性问题**。它让模型更像一个通用服务代理，但并不保证其掌握企业知识或遵守企业策略。
- **偏好对齐与服务风格层**：`Ouyang et al. 2022` 的 InstructGPT 与 `Rafailov et al. 2023` 的 DPO 共同支撑一个更系统的判断：客服中的“好回答”不是纯语义正确，而是 **正确、礼貌、可信、可执行、不过度承诺** 的复合目标。RLHF 提供了更完整的 demonstration + ranking + policy optimization 路线，DPO 则说明很多服务风格约束可以直接通过偏好对进行较轻量优化。两者并非简单替代，更像是在不同工程复杂度下实现行为塑形的两种路径。
- **工具执行层**：`Schick et al. 2023` 的 Toolformer 指出，很多客服请求本质上不是知识问答，而是 **订单查询、流程触发、状态核验、工单创建、赔付计算** 之类的外部操作。没有工具层，生成式客服常常只能充当解释器；引入工具层后，系统才有可能从“回答型客服”进入“处理型客服”。这一区分非常关键，因为它决定了客服系统最终是停留在文本交互，还是能够闭环完成用户任务。
- **安全门控与系统治理层**：`Inan et al. 2023` 的 Llama Guard 和 `Liang et al. 2022` 的 HELM 共同提醒，客服系统不能把安全和评测完全内化为主模型的一部分。客服风险包括违规承诺、危险建议、越权访问、误导性解释和不当执行。`Llama Guard` 代表的是 **独立 safeguard 层**，而 `HELM` 代表的是 **多维系统评测框架**。这两层共同说明，企业客服的可靠性来自系统分层治理，而不是单一模型“更聪明”。

## 关键争论与分歧

- **客服系统应以检索为中心，还是以生成为中心**：现有证据并不支持二选一。`Sakata et al. 2019` 证明高频标准问题非常适合 FAQ 检索；`PIKE-RAG` 则说明复杂问题需要知识重组和解释。更稳健的结论是：**检索与生成对应的是不同复杂度层级的问题**。只有当问题跨文档、跨条件、需要整合说明时，生成层才真正成立；否则，检索式标准答案往往更可靠。
- **多轮对话能力是否等于客服能力**：`Vlasov et al. 2019` 支撑“上下文选择很重要”，但这并不意味着把对话历史编码得更好，就足以构成客服系统优势。很多客服失败发生在事实依据错误、工具不可用、权限边界不清或政策执行失误上。因而“多轮能力很强”只能在 **知识层和执行层已成立** 的前提下，才转化为真正的客服收益。
- **instruction tuning 是否足够支撑企业客服**：`Wei et al. 2021` 和 `Iyer et al. 2022` 支撑其在任务泛化上的价值，但 `Ouyang et al. 2022` 与 `Rafailov et al. 2023` 同时表明，若系统需要稳定符合企业服务风格、拒答边界和风险偏好，仅靠 instruction tuning 往往不够。这个争论成立的条件在于：**当客服目标从“会回答”转向“按组织要求回答”时，偏好优化的重要性显著上升**。
- **安全应主要靠主模型内化，还是靠外部护栏系统**：`InstructGPT` 代表“通过对齐改善模型本身”，`Llama Guard` 代表“通过外部 safeguard 单独拦截风险”。当前更稳健的系统判断是：在客服这类高约束场景中，**外部护栏通常不是可选增强，而是必要分层**。只有在风险成本极低的场景下，才可能更多依赖主模型内化。
- **客服评测应看离线基准、人工偏好还是线上业务指标**：`HELM` 已经说明单指标不足，但当前知识库中的 summary 对业务指标支撑仍明显不足。因此本页能较确定地说“需要多维评测”，却还不能较确定地说“哪些线上指标构成最优主指标”。这不是写作保守，而是 **证据基础尚不够覆盖企业运营层评价**。

## 证据基础

- [Sakata et al. - 2019 - FAQ retrieval using query-question similarity and BERT-based query-answer relevance](../../wiki/summaries/Sakata%20et%20al.%20-%202019%20-%20FAQ%20retrieval%20using%20query-question%20similarity%20and%20BERT-based%20query-answer%20relevance.md)：支撑客服以 FAQ 检索为稳定起点，以及问答相似度与答案相关性需要分开建模。
- [Vlasov, Mosig, Nichol - 2019 - Dialogue Transformers](../../wiki/summaries/Vlasov,%20Mosig,%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md)：支撑多轮客服对话需要选择性建模历史上下文，而不是简单拼接全部轮次。
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)：支撑客服知识入口从词项匹配走向语义检索。
- [Oğuz et al. - 2021 - Domain-matched Pre-training Tasks for Dense Retrieval](../../wiki/summaries/O%C4%9Fuz%20et%20al.%20-%202021%20-%20Domain-matched%20Pre-training%20Tasks%20for%20Dense%20Retrieval.md)：支撑客服检索上限受领域语料和领域匹配预训练显著影响。
- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)：支撑 instruction tuning 能把通用模型转成统一任务接口。
- [Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization](../../wiki/summaries/Iyer%20et%20al.%20-%202022%20-%20OPT-IML%20Scaling%20Language%20Model%20Instruction%20Meta%20Learning%20through%20the%20Lens%20of%20Generalization.md)：支撑客服型指令能力与任务分布、采样方式和 specialized dialogue 数据设计相关。
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)：支撑客服助手需要 helpful、truthful、harmless 的服务型行为对齐。
- [Liang et al. - 2022 - Holistic Evaluation of Language Models](../../wiki/summaries/Liang%20et%20al.%20-%202022%20-%20Holistic%20Evaluation%20of%20Language%20Models.md)：支撑客服评测不能只看单一正确率，而要考虑多维治理指标。
- [Rafailov, Mitchell, Jul - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov,%20Mitchell,%20Jul%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)：支撑客服风格偏好可通过较轻量的偏好对直接优化。
- [Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools](../../wiki/summaries/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md)：支撑客服从回答型代理走向处理型代理必须引入工具调用层。
- [Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations](../../wiki/summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)：支撑客服系统中独立输入输出安全门控的必要性。
- [Wang - Unknown - PIKE-RAG sPecIalized KnowledgE and Rationale Augmented Generation](../../wiki/summaries/Wang%20-%20Unknown%20-%20PIKE-RAG%20sPecIalized%20KnowledgE%20and%20Rationale%20Augmented%20Generation.md)：支撑复杂企业问答需要 specialized knowledge extraction、task decomposition 与 rationale construction。

## 代表页面

- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [Llama Guard](../concepts/Llama%20Guard.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)

## 未解决问题

- **客服闭环执行仍缺关键证据**：当前 summary 已足以支撑“检索、生成、对齐、护栏、工具”这些模块为何重要，但还不足以支撑“企业客服如何稳定完成端到端事务闭环”的成熟结论，因为工单分流、CRM 集成、权限控制、人工接管策略等环节在知识库中仍缺更直接 summary。
- **线上业务指标与模型指标之间的映射尚未建立**：本页可以较稳地说离线准确率不足，但还不能较稳地说首次解决率、升级率、投诉率、误拒率与模型子能力之间如何对应，因为当前证据基础没有覆盖完整的业务评测链。
- **多轮记忆、个性化与合规之间的联动仍不清楚**：现有 summary 能分别说明对话建模、知识检索和安全门控的重要性，但尚不足以证明三者如何在真实企业环境中形成兼容机制。
- **澄清、拒答、追问与转人工的最优策略仍属开放问题**：现有证据能说明这些动作必要，却不足以支撑它们的决策边界已经清晰。因此这里必须保留为未解决问题，而不是泛化成“未来可继续优化”的空泛表述。

## 关联页面

- [传统 NLP](./传统%20NLP.md)
- [LLM RL](./LLM%20RL.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [RLHF](../concepts/RLHF.md)
- [Llama Guard](../concepts/Llama%20Guard.md)
