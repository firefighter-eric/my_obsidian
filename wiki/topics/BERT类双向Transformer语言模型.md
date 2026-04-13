# BERT类双向Transformer语言模型

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页讨论以 `BERT` 为起点的一类双向 Transformer 编码器语言模型，包括 `RoBERTa`、`SpanBERT`、`Sentence-BERT`、`SimCSE`、`XLM-R` 等代表节点。它们的共同点是：以 encoder-only 或双塔 encoder 为中心，主要服务理解、匹配、检索、抽取与多语言表征，而不是把自回归生成作为主目标。

这个主题不试图把“几十种 BERT 变体”逐个列成目录，而是把当前知识库已有证据收敛成几条稳定主线：原始 masked language modeling 编码器、训练配方优化、结构感知预训练、句向量化与检索化、多语言扩展。与 `LLM预训练` 的边界在于，本页不讨论 GPT/PaLM/Llama 一类 decoder-only foundation model；与 `传统NLP` 的边界在于，本页专门收紧到 BERT 类双向编码器家族，而不是覆盖整个传统 NLP 谱系。

## 核心问题

- 双向 Transformer 编码器为什么在分类、问答、抽取和检索中形成长期底座。
- BERT 家族内部的主要分化，是来自架构变化、训练配方变化，还是任务目标变化。
- 句向量、稠密检索与多语言建模为什么会从通用 BERT 主线中分化成相对独立的支线。
- 在 GPT 类生成模型兴起之后，BERT 类模型还在哪些问题上保持优势或必要性。

## 主线脉络 / 方法分层

- 原始范式建立：`Devlin et al. 2019` 的 `BERT` 把 masked language modeling 与深度双向上下文编码结合起来，奠定了“预训练编码器 + 轻量任务头微调”的标准范式。当前知识库中的很多后续路线，都是围绕这个基座做训练、目标或用途改造。
- 训练配方优化主线：`Liu et al. 2019` 的 `RoBERTa` 说明，BERT 的早期性能并不完全受架构限制，更充分的数据、训练步数、batch 规模与 masking 策略本身就能显著抬高上限。这条线的重点不是改成另一种模型，而是重新评估“BERT 到底有没有被训够”。
- 结构感知预训练主线：`Joshi et al. 2020` 的 `SpanBERT` 说明，若下游任务的核心单位不是单 token 而是 span，那么预训练目标也可以围绕 span 级表示来设计。它对应的是“BERT 家族继续演化，但演化方向不是更大，而是更贴近任务结构”。
- 句向量与检索主线：`Sentence-BERT`、`SimCSE`、`DeCLUTR`、`ConSERT`、`Trans-Encoder` 等 summary 共同表明，原始 BERT 适合理解但不天然等于“好用的句向量”。因此出现了一条把双向编码器显式改造成语义匹配、聚类、检索与双塔召回底座的路线。`Dense Retrieval` 与 `DPR` 可以视作这条路线在检索场景中的系统化落地。
- 多语言扩展主线：`XLM-R` 与 `Larger-Scale Transformers for Multilingual Masked Language Modeling` 表明，BERT 类方法并不只局限于英语任务，而是可以扩展成统一多语言编码器。这里的核心问题从“如何学好英文上下文表示”转成“如何在共享参数下平衡高资源与低资源语言”。
- 向任务系统迁移：`Text summarization with pretrained encoders` 等工作说明，BERT 类编码器虽然不是生成式 foundation model，但可以作为摘要、问答和结构化预测系统中的强底座。编者归纳上，这意味着 BERT 家族的价值不只在预训练论文本身，而在于它提供了一个可复用的通用理解骨架。

## 关键争论与分歧

- BERT 家族的提升到底来自“模型思想”还是“训练资源”：`RoBERTa` 强烈支持后者至少被早期低估，但 `SpanBERT` 又说明目标设计仍然重要，因此不能把所有改进都还原成“多训一点”。
- 双向编码器是否已经被 decoder-only LLM 淘汰：从通用聊天与生成上看，主导权已转向 decoder-only；但从句嵌入、reranking、分类、抽取与很多多语言理解任务看，encoder 仍有明显效率与结构优势。
- 句向量是否是预训练编码器的自动副产物：`Sentence-BERT`、`SimCSE` 及相关 summary 都指向否定答案，说明“会做 token-level 理解”不等于“句空间几何适合相似度检索”。
- BERT 家族应按架构还是按用途组织：当前知识库更适合按用途与问题结构组织，因为很多后续节点并没有根本改变 backbone，而是改变训练目标、池化方式或任务接口。

## 证据基础

- 原始范式与训练配方：
  - [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../raw/summary/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)
  - [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../raw/summary/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)
- 结构感知与任务特化：
  - [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../raw/summary/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)
  - [Liu, Lapata - 2020 - Text summarization with pretrained encoders](../../raw/summary/Liu,%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)
- 句向量与检索：
  - [Devlin, Liu - 2014 - Sentence-BERT Sentence Embeddings using Siamese BERT-Networks](../../raw/summary/Devlin,%20Liu%20-%202014%20-%20Sentence-BERT%20Sentence%20Embeddings%20using%20Siamese%20BERT-Networks.md)
  - [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../raw/summary/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)
  - [Giorgi et al. - 2021 - DeCLUTR Deep contrastive learning for unsupervised textual representations](../../raw/summary/Giorgi%20et%20al.%20-%202021%20-%20DeCLUTR%20Deep%20contrastive%20learning%20for%20unsupervised%20textual%20representations.md)
  - [Yan et al. - 2021 - ConSERT A contrastive framework for self-supervised sentence representation transfer](../../raw/summary/Yan%20et%20al.%20-%202021%20-%20ConSERT%20A%20contrastive%20framework%20for%20self-supervised%20sentence%20representation%20transfer.md)
  - [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../raw/summary/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)
- 多语言扩展：
  - [Conneau et al. - 2020 - Unsupervised cross-lingual representation learning at scale](../../raw/summary/Conneau%20et%20al.%20-%202020%20-%20Unsupervised%20cross-lingual%20representation%20learning%20at%20scale.md)
  - [Conneau - 2021 - Larger-Scale Transformers for Multilingual Masked Language Modeling](../../raw/summary/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md)

## 代表页面

- [BERT](../concepts/BERT.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [XLM-R](../concepts/XLM-R.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)

## 未解决问题

- 当前知识库还没有把 `ALBERT`、`ELECTRA`、`DeBERTa`、`MPNet`、`DistilBERT` 等常见 BERT 变体补成 summary 或 concept，因此本页只能先给出代表性分层，而不能形成完整家谱。
- 句向量路线目前已有多篇 summary，但还缺一个独立 comparison 页来比较 `Sentence-BERT`、`SimCSE`、`ConSERT`、`DeCLUTR` 与检索双塔方法。
- 多语言编码器与多语言生成器的边界，目前只在 `XLM-R` 与 `mT5` 的概念页层面被粗略区分，后续仍可细化为专门比较页。

## 关联页面

- [传统 NLP](传统%20NLP.md)
- [LLM 预训练](LLM%20预训练.md)
- [搜索排序](./搜索排序.md)
- [BERT](../concepts/BERT.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [XLM-R](../concepts/XLM-R.md)
