# Devlin, Liu - 2014 - Sentence-BERT Sentence Embeddings using Siamese BERT-Networks

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Devlin, Liu - 2014 - Sentence-BERT Sentence Embeddings using Siamese BERT-Networks.pdf
- 全文文本：../../raw/text/Devlin, Liu - 2014 - Sentence-BERT Sentence Embeddings using Siamese BERT-Networks.md
- 作者：Devlin, Liu
- 年份：2014
- 状态：已抽取全文，待精读

## 自动抽取摘要

BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS). How- ever, it requires that both sentences are fed into the network, which causes a massive com- putational overhead: Finding the most sim- ilar pair in a collection of 10,000 sentences requires about 50 million inference computa- tions (~65 hours) with BERT. The construction of BERT makes it unsuitable for semantic sim- ilarity search as well as for unsupervised tasks like clustering. In this publication, we present Sentence-BERT (SBERT), a modiﬁcation of the pretrained BERT network that use siamese and triplet net- work structures to derive semantically mean- ingful sentence embeddings that can be com- pared using cosine-similarity. This reduces the effort for ﬁnding the most similar pair from 65 hours with BERT / RoBERTa to about 5 sec- onds with SBERT, while maintaining the ac- curacy from BERT. We evaluate SBERT and SRoBERTa on com- mon STS tasks and transfer learning tasks, where it outperforms other state-of-the-art sentence embeddings methods.1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Devlin, Liu - 2014 - Sentence-BERT Sentence Embeddings using Siamese BERT-Networks.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
