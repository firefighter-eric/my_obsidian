# Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021.pdf
- 全文文本：../../raw/text/Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021.md
- 作者：Liang et al.
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

Dropout is a powerful and widely used technique to regularize the training of deep neural networks. Though effective and performing well, the randomness introduced by dropout causes unnegligible inconsistency between training and inference. In this paper, we introduce a simple consistency training strategy to regularize dropout, namely R-Drop, which forces the output distributions of different sub models gen- erated by dropout to be consistent with each other. Speciﬁcally, for each training sample, R-Drop minimizes the bidirectional KL-divergence between the output dis- tributions of two sub models sampled by dropout. Theoretical analysis reveals that R-Drop reduces the above inconsistency. Experiments on 5 widely used deep learn- ing tasks (18 datasets in total), including neural machine translation, abstractive summarization, language understanding, language modeling, and image classiﬁca- tion, show that R-Drop is universally effective. In particular, it yields substantial improvements when applied to ﬁne-tune large-scale pre-trained models, e.g., ViT, RoBERTa-large, and BART, and achieves state-of-the-art (SOTA) performances with the vanilla Transformer model on WMT14 English→German translation (30.91 BLEU) and WMT14 English→French translation (43.95 BLEU), even surpassing models trained with extra large-scale data and expert-designed advanced variants of Transformer models. Our code is available at GitHub2.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
