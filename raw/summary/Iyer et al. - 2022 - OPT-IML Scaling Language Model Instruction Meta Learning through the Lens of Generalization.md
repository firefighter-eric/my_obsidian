# Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization.pdf
- 全文文本：../../raw/text/Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization.md
- 作者：Iyer et al.
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

Recent work has shown that ﬁne-tuning large pre-trained language models on a collection of tasks described via instructions, a.k.a. instruction-tuning, improves their zero and few-shot generalization to unseen tasks. However, there is a limited understanding of the performance trade-oﬀs of diﬀerent decisions made during the instruction-tuning process. These decisions include the scale and diversity of the instruction-tuning benchmark, diﬀerent task sampling strategies, ﬁne-tuning with and without demonstrations, training using specialized datasets for reasoning and dialogue, and ﬁnally, the ﬁne-tuning objectives themselves. In this paper, we characterize the eﬀect of instruction-tuning decisions on downstream task performance when scaling both model and benchmark sizes. To this end, we create OPT-IML Bench: a large benchmark for Instruction Meta- Learning (IML) of 2000 NLP tasks consolidated into task categories from 8 existing benchmarks, and prepare an evaluation framework to measure three types of model generalizations: to tasks from fully held-out categories, to held-out tasks from seen categories, and to held-out instances from seen tasks. Through the lens of this framework, we ﬁrst present insights about instruction- tuning decisions as applied to OPT-30B and further exploit these insights to train OPT-IML 30B and 175B, which are instruction-tuned versions of OPT. OPT-IML demonstrates all three generalization abilities at both scales on four diﬀerent evaluation benchmarks with diverse tasks and input formats – PromptSource, FLAN, Super-NaturalInstructions, and UniﬁedSKG. Not only does it signiﬁcantly outperform OPT on all benchmarks but is also highly competitive with existing models ﬁne-tuned on each speciﬁc benchmark. We release OPT-IML at both scales, together with the OPT-IML Bench evaluation framework. 1. Introduction Instruction ﬁne-tuning is shown (Wei et al., 2022a; Sanh et al., 2022; Chung et al., 2022a) to sig- niﬁcantly improve the zero- and few-shot performance of large pretrained LMs (LLM). It involves ﬁne-tuning LLMs on collections of NLP tasks using instructional style input formats. Successful instruction-tuning of LLMs depends on a number of aspects such as the objectives used for ﬁne- tuning, the distribution and diversity of the ﬁne-tuning tasks, the inclusion of specialized datasets related to reasoning and dialogue, ﬁne-tuning with demonstrations, and also, the comprehensiveness of the evaluation framework. In this paper, we develop an extensive large-scale ﬁne-tuning and evaluation framework of 2000 NLP tasks (which we call OPT-IML Bench) and use it to characterize the tradeoﬀs of diﬀerent decisions relating to instruction meta-learning (IML) on the OPT models (Zhang et al., 2022). We exploit insights gathered from this process, to train OPT-IML 30B and 175B, instruction-tuned versions of OPT. There are a growing number of large meta-datasets of NLP tasks such as Super-NaturalInstructions (Wang et al., 2022), FLAN (Wei et al., 2022a) and PromptSource (Sanh et al., 2022). Recent instruction-tuning work has demonstrated success using these individual benchmarks and their com- binations (Chung et al., 2022b), with a general recommendation for scaling up the number of tasks. ∗. Equal contribution; alphabetical order. †. Work done while at Meta AI.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM预训练](../topics/LLM预训练.md)
- 综合：暂无
