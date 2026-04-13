# 传统 NLP

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页讨论 LLM 时代之前或与 LLM 并行存在的经典 NLP 方法脉络，重点是编码器预训练、句向量、检索、摘要与结构化任务，而不是大规模自回归生成模型本身。它与 `LLM预训练` 的边界在于：这里优先组织“具体任务与表征方法如何演进”，而不是“通用生成能力如何通过超大规模训练获得”。

## 核心问题

- 传统 NLP 如何从静态特征与任务专用模型过渡到统一预训练表征。
- 编码器预训练为何在分类、问答、抽取和摘要等任务上形成通用底座。
- 句向量与检索为什么发展成相对独立的方法支线。
- 任务特化改造与统一基础模型之间的关系应如何理解。

## 主线脉络 / 方法分层

- 预训练编码器主线：`BERT` 建立了双向预训练编码器范式，`RoBERTa` 则说明很多收益来自更充分的训练配方而非彻底改架构。
- 任务结构感知改造：`SpanBERT` 显示预训练目标可以围绕 span 级任务结构定制，从而在抽取、问答和共指等任务上获得额外收益。
- 句向量与语义匹配：`SimCSE` 代表对比学习句向量路线，说明编码器预训练之后，表征空间本身也需要针对相似度任务再整理。
- 稠密检索支线：`Dense Passage Retrieval` 把问答中的检索前端从稀疏词项匹配推进到双塔向量检索，是现代 RAG 思想的重要前史。
- 预训练到生成任务迁移：`Text summarization with pretrained encoders` 表明预训练编码器路线也能向摘要等生成/半生成任务延展，而不只局限于分类。

## 关键争论与分歧

- 传统 NLP 的边界应划在哪里：本页把 BERT 系列、句向量、检索和预训练编码器摘要方法视为传统 NLP 延展，而不把 GPT-3 之后的超大自回归模型纳入主干。
- 统一基础模型是否会消解任务差异：`SpanBERT`、DPR、摘要模型都表明，统一表征很重要，但任务结构仍会推动特化设计。
- 稠密检索是否应算入 LLM 主题：在知识史上它也是后续 RAG 的前置条件，但在当前页面中仍先作为传统 NLP 检索分支处理。
- 句向量是独立主题还是编码器副产品：`SimCSE` 等工作说明句向量质量不能被视为预训练编码器的自动副产物。

## 证据基础

- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../raw/summary/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)
- [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../raw/summary/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)
- [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../raw/summary/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)
- [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../raw/summary/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../raw/summary/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)
- [Liu, Lapata - 2020 - Text summarization with pretrained encoders](../../raw/summary/Liu,%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)

## 代表页面

- [BERT](../concepts/BERT.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)

## 未解决问题

- 当前页面尚未把信息抽取、依存分析、问答与摘要分别展开成 comparison 或子 topic。
- 传统 NLP 与 LLM 时代检索增强方法之间的连接还未系统梳理。
- 编码器路线与生成式路线在知识库中的长期边界，后续仍可能随着更多 summary 接入而调整。

## 关联页面

- [LLM 预训练](LLM%20预训练.md)
- [BERT类双向Transformer语言模型](./BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [搜索排序](./搜索排序.md)
- [BERT](../concepts/BERT.md)
- [Transformer](../concepts/Transformer.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
