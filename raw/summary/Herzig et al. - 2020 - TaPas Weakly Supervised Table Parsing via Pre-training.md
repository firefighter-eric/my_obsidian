# Herzig et al. - 2020 - TaPas Weakly Supervised Table Parsing via Pre-training

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Herzig et al. - 2020 - TaPas Weakly Supervised Table Parsing via Pre-training.pdf
- 全文文本：../../raw/text/Herzig et al. - 2020 - TaPas Weakly Supervised Table Parsing via Pre-training.md
- 作者：Herzig et al.
- 年份：2020
- 状态：已抽取全文，待精读

## 自动抽取摘要

Answering natural language questions over ta- bles is usually seen as a semantic parsing task. To alleviate the collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of logical forms. However, training seman- tic parsers from weak supervision poses difﬁ- culties, and in addition, the generated logical forms are only used as an intermediate step prior to retrieving the denotation. In this pa- per, we present TAPAS, an approach to ques- tion answering over tables without generating logical forms. TAPAS trains from weak super- vision, and predicts the denotation by select- ing table cells and optionally applying a cor- responding aggregation operator to such selec- tion. TAPAS extends BERT’s architecture to encode tables as input, initializes from an ef- fective joint pre-training of text segments and tables crawled from Wikipedia, and is trained end-to-end. We experiment with three differ- ent semantic parsing datasets, and ﬁnd that TAPAS outperforms or rivals semantic parsing models by improving state-of-the-art accuracy on SQA from 55.1 to 67.2 and performing on par with the state-of-the-art on WIKISQL and WIKITQ, but with a simpler model architec- ture. We additionally ﬁnd that transfer learn- ing, which is trivial in our setting, from WIK- ISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Herzig et al. - 2020 - TaPas Weakly Supervised Table Parsing via Pre-training.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
