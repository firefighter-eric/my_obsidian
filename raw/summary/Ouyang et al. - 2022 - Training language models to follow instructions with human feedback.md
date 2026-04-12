# Ouyang et al. - 2022 - Training language models to follow instructions with human feedback

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf
- 全文文本：../../raw/text/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.md
- 作者：Ouyang et al.
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

Making language models bigger does not inherently make them better at following a user’s intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by ﬁne-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to ﬁne-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further ﬁne-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that ﬁne-tuning with human feedback is a promising direction for aligning language models with human intent.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
