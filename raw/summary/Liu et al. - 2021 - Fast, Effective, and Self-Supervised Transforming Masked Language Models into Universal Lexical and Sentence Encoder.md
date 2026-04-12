# Liu et al. - 2021 - Fast, Effective, and Self-Supervised Transforming Masked Language Models into Universal Lexical and Sentence Encoder

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Liu et al. - 2021 - Fast, Effective, and Self-Supervised Transforming Masked Language Models into Universal Lexical and Sentence Encoder.pdf
- 全文文本：../../raw/text/Liu et al. - 2021 - Fast, Effective, and Self-Supervised Transforming Masked Language Models into Universal Lexical and Sentence Encoder.md
- 作者：Liu et al.
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

Previous work has indicated that pretrained Masked Language Models (MLMs) are not ef- fective as universal lexical and sentence en- coders off-the-shelf, i.e., without further task- speciﬁc ﬁne-tuning on NLI, sentence similar- ity, or paraphrasing tasks using annotated task data. In this work, we demonstrate that it is possible to turn MLMs into effective lexical and sentence encoders even without any addi- tional data, relying simply on self-supervision. We propose an extremely simple, fast, and ef- fective contrastive learning technique, termed Mirror-BERT, which converts MLMs (e.g., BERT and RoBERTa) into such encoders in 20–30 seconds with no access to additional external knowledge. Mirror-BERT relies on identical and slightly modiﬁed string pairs as positive (i.e., synonymous) ﬁne-tuning exam- ples, and aims to maximise their similarity dur- ing “identity ﬁne-tuning”. We report huge gains over off-the-shelf MLMs with Mirror- BERT both in lexical-level and in sentence- level tasks, across different domains and differ- ent languages. Notably, in sentence similarity (STS) and question-answer entailment (QNLI) tasks, our self-supervised Mirror-BERT model even matches the performance of the Sentence- BERT models from prior work which rely on annotated task data. Finally, we delve deeper into the inner workings of MLMs, and sug- gest some evidence on why this simple Mirror- BERT ﬁne-tuning approach can yield effective universal lexical and sentence encoders.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Liu et al. - 2021 - Fast, Effective, and Self-Supervised Transforming Masked Language Models into Universal Lexical and Sentence Encoder.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
