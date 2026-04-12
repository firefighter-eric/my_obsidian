# Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model.pdf
- 全文文本：../../raw/text/Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model.md
- 作者：Rafailov et al.
- 年份：2023
- 状态：已抽取全文，待精读

## 自动抽取摘要

While large-scale unsupervised language models (LMs) learn broad world knowl- edge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these prefer- ences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Prefer- ence Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sen- timent of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
