# Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction.pdf
- 全文文本：../../raw/text/Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction.md
- 作者：Wu et al.
- 年份：2020
- 状态：已抽取全文，待精读

## 自动抽取摘要

In this paper, we present CorefQA, an accu- rate and extensible approach for the corefer- ence resolution task. We formulate the prob- lem as a span prediction task, like in ques- tion answering: A query is generated for each candidate mention using its surrounding con- text, and a span prediction module is em- ployed to extract the text spans of the corefer- ences within the document using the generated query. This formulation comes with the fol- lowing key advantages: (1) The span predic- tion strategy provides the ﬂexibility of retriev- ing mentions left out at the mention proposal stage; (2) In the question answering frame- work, encoding the mention and its context ex- plicitly in a query makes it possible to have a deep and thorough examination of cues em- bedded in the context of coreferent mentions; and (3) A plethora of existing question an- swering datasets can be used for data augmen- tation to improve the model’s generalization capability. Experiments demonstrate signiﬁ- cant performance boost over previous models, with 83.1 (+3.5) F1 score on the CoNLL-2012 benchmark and 87.5 (+2.5) F1 score on the GAP benchmark. 1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
