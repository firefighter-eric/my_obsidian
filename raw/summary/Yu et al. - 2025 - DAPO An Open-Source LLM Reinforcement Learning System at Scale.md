# Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale.pdf
- 全文文本：../../raw/text/Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale.md
- 作者：Qiying Yu et al.
- 年份：2025
- 状态：已基于 arXiv HTML 提取全文并精修 summary

## 自动抽取摘要或人工摘要

`DAPO` 不是只提出一个新损失，而是把“如何把 reasoning RL 真正跑起来”当成系统问题来处理。论文从 `naive GRPO` 复现效果不佳出发，指出长 CoT RL 会碰到 entropy collapse、reward noise、训练不稳定等问题，并提出 `Clip-Higher`、`Dynamic Sampling`、`Token-Level Policy Gradient Loss`、`Overlong Reward Shaping` 四类改造。它的重要性在于：如果 `DeepSeek-R1` 让社区意识到 `GRPO`/reasoning RL 的潜力，那么 `DAPO` 则试图把这条路线变成更可复现的公开工程体系。

## 关键事实

- 论文把 `DAPO` 明确定义为面向大规模 LLM RL 的 open-source system，而不是单个孤立技巧。
- `DAPO` 以 `GRPO` 为出发点，针对长链路 reasoning RL 中的训练崩塌、奖励噪声和梯度退化做系统修正。
- 文中报告基于 `Qwen2.5-32B` 的大规模 RL 结果，并将其作为可复现实验配方公开。
- 在当前知识库中，`DAPO` 代表从“提出 reasoning RL 算法”走向“把 reasoning RL 做成稳定工程系统”的下一阶段。

## 争议与不确定点

- 论文报告的优势既来自 `DAPO` 目标本身，也来自数据处理、长度控制、系统实现和超参 recipe，不能把全部提升视为纯算法差异。
- 它聚焦数学推理 / 长 CoT 情景，是否能平移到更开放的 agent / safety / multimodal RL 仍需后续来源补充。
- `DAPO` 对 `GRPO` 的批评与修正说明，reasoning RL 的关键难点已不只是“有没有 RL”，而是“RL 在长推理轨迹上的稳定实现”。

## 关联页面

- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[DAPO](../../wiki/concepts/DAPO.md)
- 概念：[GRPO](../../wiki/concepts/GRPO.md)

