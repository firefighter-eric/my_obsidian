# Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings.pdf
- 全文文本：../../raw/text/Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings.md
- 作者：Gao, Yao, Chen
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

This paper presents SimCSE, a simple con- trastive learning framework that greatly ad- vances the state-of-the-art sentence embed- dings. We ﬁrst describe an unsupervised ap- proach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, per- forming on par with previous supervised coun- terparts. We ﬁnd that dropout acts as mini- mal data augmentation and removing it leads to a representation collapse. Then, we pro- pose a supervised approach, which incorpo- rates annotated pairs from natural language inference datasets into our contrastive learn- ing framework, by using “entailment” pairs as positives and “contradiction” pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERTbase achieve an average of 76.3% and 81.6% Spearman’s correlation respectively, a 4.2% and 2.2% improvement compared to previous best results. We also show—both theoretically and empirically—that contrastive learning objective regularizes pre-trained em- beddings’ anisotropic space to be more uni- form, and it better aligns positive pairs when supervised signals are available.1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
