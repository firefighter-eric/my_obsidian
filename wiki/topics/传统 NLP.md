# 传统 NLP

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 **LLM 时代之前形成、且在 LLM 时代仍持续影响方法结构的 NLP 主线**。这里的“传统”不是指纯手工规则或统计时代的全部历史，而是指 **以双向编码器、句向量、稠密检索、抽取式/编码器式摘要等为核心的现代 NLP 中间层**。它们通常不以开放式生成能力为目标，而以理解、匹配、召回、排序和结构化预测为主。

因此，本页并不试图覆盖从 `n-gram` 到 CRF 的完整史前谱系，也不把 GPT、PaLM、Llama 一类 decoder-only foundation model 纳入主干。更合适的理解是：这里整理的是 **通向 LLM 时代之前后的“表示学习 NLP”主线**，也就是后来很多 RAG、检索增强问答和 encoder-decoder 任务系统的技术前史。

它与 [BERT类双向Transformer语言模型](./BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md) 的区别在于：后者聚焦 BERT 家族本身；本页则把 BERT 视为传统 NLP 后期的中心节点之一，并同时纳入句向量、稠密检索与编码器式摘要等相邻路线。它与 [LLM 预训练](../topics/LLM%20预训练.md) 的边界在于：本页关心 **任务接口与表征结构如何演进**，而不是通用生成模型如何通过规模化训练获得能力。

## 核心问题

- **传统 NLP 是如何从任务专用模型转向统一表征底座的**：这决定了后续为何能出现“一个编码器服务多任务”的工作模式。
- **哪些问题促使传统 NLP 从 token 表征走向句向量和检索向量**：这背后是任务接口变化，而不是简单的模型更换。
- **为什么检索、排序、摘要、问答虽相邻却不能混成一个主题**：它们共享编码器底座，但解决的结构问题并不相同。
- **传统 NLP 与 LLM 时代的分界应划在哪里**：若边界划得过宽，会把一切都写成“LLM 前史”；划得过窄，又会丢失现代检索和 RAG 的技术根系。

## 主线脉络 / 方法分层

- **统一预训练编码器层**：`Devlin et al. 2019` 的 `BERT` 与 `Liu et al. 2019` 的 `RoBERTa` 共同标志传统 NLP 后期最重要的转折，即从大量任务专用建模转向 **统一预训练编码器底座**。这里的关键不是“Transformer 更强”这么简单，而是大量分类、抽取、问答、摘要任务开始共享同一套表示学习基础设施。这使传统 NLP 从“每个任务一套特征工程”过渡到“一个底座适配多任务”。
- **任务结构感知层**：`Joshi et al. 2020` 的 `SpanBERT` 表明，即使进入统一编码器时代，任务结构仍不会消失。抽取、问答、共指等任务高度依赖 span 单元，因此预训练目标也会围绕 span 重写。这个层次说明，**统一底座并没有消解任务差异，而是把任务差异转移到预训练目标与任务接口上**。
- **句向量与语义匹配层**：`SimCSE` 及相关句表示工作说明，分类型编码器并不天然等于高质量相似度空间。传统 NLP 在这一阶段分化出一条独立的句向量路线，其核心不是让模型“更懂句子”，而是让语义空间 **足够适合检索、聚类、匹配和迁移**。这条线后来直接影响 dense retrieval、reranking 与 RAG 中的表征接口。
- **稠密检索层**：`Karpukhin et al. 2020` 的 DPR 把问答系统的前端从稀疏召回推进到双塔语义检索。这条线在知识史上的价值非常高，因为它说明传统 NLP 后期已经开始把表征学习直接用于 **可索引的大规模召回问题**，为后来的检索增强生成提供了清晰前史。DPR 的出现也意味着“理解”不再只是做分类和抽取，而是成为信息访问系统的一部分。
- **编码器式摘要与任务外延层**：`Liu, Lapata 2020` 以及 `Liu 2019` 的抽取式摘要工作共同说明，传统 NLP 后期并不只停留在理解类任务。编码器底座已经被用来支撑摘要这类更复杂的任务结构，但它采取的仍是 **编码器主导的生成或半生成接口**，而不是 today LLM 式的通用开放生成。它们证明了传统 NLP 在生成任务上的延展能力，同时也暴露出其与 decoder-only 路线的边界。
- **不依赖大规模预训练的反思层**：`Yao et al. 2021` 的 “NLP From Scratch Without Large-Scale Pretraining” 代表一个重要提醒，即传统 NLP 的后期并非所有问题都自动收敛到“大规模预训练越大越好”。这条线提示我们，**传统 NLP 并不是单向走向更大模型**，而是始终存在对效率、任务结构和训练成本的反思。

## 关键争论与分歧

- **传统 NLP 的边界到底应划在哪里**：若把它等同于“LLM 之前的一切 NLP”，页面会失去结构；若只把它理解为 CRF、HMM 和词袋时代，它又无法解释为何 BERT、DPR、句向量仍应被视为传统 NLP 的延展。当前更合理的边界是：**把非开放式 foundation model、但已形成统一表示学习底座的一整段现代 NLP 主线纳入本页**。
- **统一编码器是否已经消解任务差异**：现有证据支持否定判断。`SpanBERT`、DPR、摘要模型都表明，统一底座只解决“共享表示”的问题，不解决“任务结构相同”的问题。只要任务的最优单元、评价方式和执行接口不同，特化设计就仍然成立。
- **稠密检索应算传统 NLP，还是应直接纳入 LLM 主题**：从今天的应用语境看，它常被写进 RAG 叙事；但从方法史看，它首先是编码器表征学习与信息检索结合的结果。因此在当前知识库中，把它保留为传统 NLP 的后期分支更能保留历史连续性。
- **生成式任务是否已经把传统 NLP 推向终点**：`Liu, Lapata 2020` 已经表明，传统编码器路线可以外延到摘要等任务；但它的生成能力仍然 strongly 受限于任务结构与模型接口。因此更稳妥的判断是：**传统 NLP 并未被直接终结，而是在开放生成问题上逐步把主导权让给了 decoder-only 路线**。

## 证据基础

- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../wiki/summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)：支撑统一预训练编码器底座的建立。
- [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../wiki/summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)：支撑训练配方重估在传统 NLP 后期的重要性。
- [Liu - 2019 - Fine-tune BERT for Extractive Summarization](../../wiki/summaries/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.md)：支撑编码器底座向抽取式摘要外延的代表节点。
- [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../wiki/summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)：支撑任务结构感知预训练目标仍然必要。
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)：支撑稠密检索作为传统 NLP 后期的重要分支。
- [Liu, Lapata - 2020 - Text summarization with pretrained encoders](../../wiki/summaries/Liu,%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)：支撑预训练编码器向更复杂摘要系统的迁移。
- [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../wiki/summaries/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)：支撑句向量支线从通用编码器中独立出来。
- [Yao et al. - 2021 - NLP From Scratch Without Large-Scale Pretraining A Simple and Efficient Framework](../../wiki/summaries/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.md)：支撑对“大规模预训练必然主导一切”这一叙事的反思。

## 代表页面

- [BERT](../concepts/BERT.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)

## 未解决问题

- **传统 NLP 的更早历史仍未纳入当前证据面**：本页目前能够稳固覆盖的是“预训练编码器时代及其相邻支线”，但还不能系统讨论 CRF、统计机器翻译、经典 learning-to-rank、信息抽取旧主线，因为缺少对应 summary。
- **句向量、检索与排序之间的中层连接仍偏松散**：当前证据已能说明它们共享编码器表征前史，但尚不足以形成一页更成熟的 comparison，去明确它们的任务边界、评测差异和部署分工。
- **传统 NLP 与 LLM 时代的长期分工仍需更多综述证据**：本页可以较稳地说很多结构化理解任务仍保留编码器优势，但关于这种分工会持续多久、会被哪些新型小型生成模型侵蚀，当前证据仍不足。

## 关联页面

- [LLM 预训练](../topics/LLM%20预训练.md)
- [BERT类双向Transformer语言模型](./BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [搜索排序](./搜索排序.md)
- [BERT](../concepts/BERT.md)
- [Transformer](../concepts/Transformer.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
