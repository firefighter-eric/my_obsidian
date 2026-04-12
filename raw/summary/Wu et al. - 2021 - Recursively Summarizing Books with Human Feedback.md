# Wu et al. - 2021 - Recursively Summarizing Books with Human Feedback

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wu et al. - 2021 - Recursively Summarizing Books with Human Feedback.pdf
- 全文文本：../../raw/text/Wu et al. - 2021 - Recursively Summarizing Books with Human Feedback.md
- 作者：Wu et al.
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

A major challenge for scaling machine learning is training models to perform tasks that are very difﬁcult or time-consuming for humans to evaluate. We present progress on this problem on the task of abstractive summarization of entire ﬁction novels. Our method combines learning from human feedback with recursive task decomposition: we use models trained on smaller parts of the task to assist humans in giving feedback on the broader task. We collect a large volume of demonstrations and comparisons from human labelers, and ﬁne-tune GPT-3 using behavioral cloning and reward modeling to do summarization recursively. At inference time, the model ﬁrst summarizes small sections of the book and then recursively summarizes these summaries to produce a summary of the entire book. Our human labelers are able to supervise and evaluate the models quickly, despite not having read the entire books themselves. Our resulting model generates sensible summaries of entire books, even matching the quality of human-written summaries in a few cases (∼5% of books). We achieve state-of-the-art results on the recent BookSum dataset for book-length summarization. A zero-shot question-answering model using these summaries achieves state-of-the-art results on the challenging NarrativeQA benchmark for answering questions about books and movie scripts. We release datasets of samples from our model.2

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Wu et al. - 2021 - Recursively Summarizing Books with Human Feedback.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
