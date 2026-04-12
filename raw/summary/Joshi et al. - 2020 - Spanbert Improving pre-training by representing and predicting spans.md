# Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans.pdf
- 全文文本：../../raw/text/Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans.md
- 作者：Joshi et al.
- 年份：2020
- 状态：已抽取全文，待精读

## 自动抽取摘要

We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary represen- tations to predict the entire content of the masked span, without relying on the indi- vidual token representations within it. Span- BERT consistently outperforms BERT and our better-tuned baselines, with substantial gains on span selection tasks such as ques- tion answering and coreference resolution. In particular, with the same training data and model size as BERTlarge, our single model obtains 94.6% and 88.7% F1 on SQuAD 1.1 and 2.0 respectively. We also achieve a new state of the art on the OntoNotes coreference resolution task (79.6% F1), strong perfor- mance on the TACRED relation extraction benchmark, and even gains on GLUE.1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
