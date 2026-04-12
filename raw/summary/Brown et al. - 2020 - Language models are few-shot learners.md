# Brown et al. - 2020 - Language models are few-shot learners

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Brown et al. - 2020 - Language models are few-shot learners.pdf
- 全文文本：../../raw/text/Brown et al. - 2020 - Language models are few-shot learners.md
- 作者：Brown et al.
- 年份：2020
- 状态：已抽取全文，待精读

## 自动抽取摘要

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by ﬁne-tuning on a speciﬁc task. While typically task-agnostic in architecture, this method still requires task-speciﬁc ﬁne-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art ﬁne- tuning approaches. Speciﬁcally, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or ﬁne-tuning, with tasks and few-shot demonstrations speciﬁed purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-ﬂy reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we ﬁnd that GPT-3 can generate samples of news articles which human evaluators have difﬁculty distinguishing from articles written by humans. We discuss broader societal impacts of this ﬁnding and of GPT-3 in general. ∗Equal contribution †Johns Hopkins University, OpenAI Author contributions listed at end of paper. arXiv:2005.14165v4 [cs.CL] 22 Jul 2020 Contents

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Brown et al. - 2020 - Language models are few-shot learners.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM预训练](../topics/LLM预训练.md)
- 综合：暂无
