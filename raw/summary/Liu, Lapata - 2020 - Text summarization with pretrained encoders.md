# Liu, Lapata - 2020 - Text summarization with pretrained encoders

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Liu, Lapata - 2020 - Text summarization with pretrained encoders.pdf
- 全文文本：../../raw/text/Liu, Lapata - 2020 - Text summarization with pretrained encoders.md
- 作者：Liu, Lapata
- 年份：2020
- 状态：已抽取全文，待精读

## 自动抽取摘要

Bidirectional Encoder Representations from Transformers (BERT; Devlin et al. 2019) rep- resents the latest incarnation of pretrained lan- guage models which have recently advanced a wide range of natural language processing tasks. In this paper, we showcase how BERT can be usefully applied in text summariza- tion and propose a general framework for both extractive and abstractive models. We intro- duce a novel document-level encoder based on BERT which is able to express the semantics of a document and obtain representations for its sentences. Our extractive model is built on top of this encoder by stacking several inter- sentence Transformer layers. For abstractive summarization, we propose a new ﬁne-tuning schedule which adopts different optimizers for the encoder and the decoder as a means of al- leviating the mismatch between the two (the former is pretrained while the latter is not). We also demonstrate that a two-staged ﬁne-tuning approach can further boost the quality of the generated summaries. Experiments on three datasets show that our model achieves state- of-the-art results across the board in both ex- tractive and abstractive settings.1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Liu, Lapata - 2020 - Text summarization with pretrained encoders.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
