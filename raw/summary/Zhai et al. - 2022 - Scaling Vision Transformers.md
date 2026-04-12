# Zhai et al. - 2022 - Scaling Vision Transformers

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Zhai et al. - 2022 - Scaling Vision Transformers.pdf
- 全文文本：../../raw/text/Zhai et al. - 2022 - Scaling Vision Transformers.md
- 作者：Zhai et al.
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

Attention-based neural networks such as the Vision Trans- former (ViT) have recently attained state-of-the-art results on many computer vision benchmarks. Scale is a primary ingredient in attaining excellent results, therefore, under- standing a model’s scaling properties is a key to designing future generations effectively. While the laws for scaling Transformer language models have been studied, it is un- known how Vision Transformers scale. To address this, we scale ViT models and data, both up and down, and character- ize the relationships between error rate, data, and compute. Along the way, we reﬁne the architecture and training of ViT, reducing memory consumption and increasing accuracy of the resulting models. As a result, we successfully train a ViT model with two billion parameters, which attains a new state-of-the-art on ImageNet of 90.45% top-1 accuracy. The model also performs well for few-shot transfer, for example, reaching 84.86% top-1 accuracy on ImageNet with only 10 examples per class. 1. Introduction Attention-based Transformer architectures [45] have taken computer vision domain by storm [8,16] and are be- coming an increasingly popular choice in research and prac- tice. Previously, Transformers have been widely adopted in the natural language processing (NLP) domain [7,15]. Opti- mal scaling of Transformers in NLP was carefully studied in [22], with the main conclusion that large models not only perform better, but do use large computational budgets more efﬁciently. However, it remains unclear to what extent these ﬁndings transfer to the vision domain, which has several important differences. For example, the most successful pre-training schemes in vision are supervised, as opposed to unsupervised pre-training in the NLP domain. In this paper we concentrate on scaling laws for transfer performance of ViT models pre-trained on image classiﬁca- ⋆equal contribution

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Zhai et al. - 2022 - Scaling Vision Transformers.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
