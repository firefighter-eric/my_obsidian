# Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf
- 全文文本：../../raw/text/Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.md
- 作者：Zhihong Shao et al.
- 年份：2024
- 状态：已基于 arXiv HTML 提取全文并精修 summary

## 自动抽取摘要或人工摘要

`DeepSeekMath` 的核心贡献不只是数学领域继续预训练，还包括把 `GRPO` 明确提出为面向 LLM reasoning 的高效 RL 变体。论文将数学推理提升归因于两层因素：一是通过高质量数学网页数据构建 120B token 的数学语料，二是在 instruction-tuned 模型上使用 `GRPO` 做强化学习。与传统 `PPO` 相比，`GRPO` 去掉 critic model，改用组内相对分数估计 baseline，以更低训练成本提升数学与跨域推理表现。文中还尝试用统一视角解释 `RFT / DPO / PPO / GRPO`，因此它在当前知识库里不仅是数学模型论文，也是 `reasoning-oriented RL` 方法分化的重要来源。

## 关键事实

- 论文明确提出 `GRPO`，并将其定义为 `PPO` 的变体：不再训练与策略模型同规模的 critic，而是从同组采样结果的相对得分估计 advantage。
- `GRPO` 在文中被用于提升 `DeepSeekMath-Instruct 7B` 的数学推理能力，并带来 in-domain 与 out-of-domain 数学任务增益。
- 该文把 `RFT`、`DPO`、`PPO`、`GRPO` 放在统一 RL 视角下讨论，说明偏好优化与 RL 并不是完全割裂的方法谱系。
- 这篇论文是当前知识库里 `GRPO` 的最早明确方法来源，也是后续 `DeepSeek-R1` 采用 `GRPO` 的前置节点。

## 争议与不确定点

- 该文主要在数学推理场景验证 `GRPO`，其在更通用 chat / safety / agent 任务中的泛化能力，仍需结合后续来源判断。
- 文中把多种方法统一放到 RL 范式下解释，这更像研究者的理论归纳，不应直接等同为社区已经达成共识的分类。
- `GRPO` 带来的收益与数学专门语料、指令数据设计之间存在耦合，不能把全部效果都归因于优化器本身。

## 关联页面

- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[GRPO](../../wiki/concepts/GRPO.md)
- 概念：[DeepSeek-R1](../../wiki/concepts/DeepSeek-R1.md)

