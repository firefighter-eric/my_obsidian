# Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models.pdf
- 全文文本：../../raw/text/Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models.md
- 作者：Touvron et al.
- 年份：2023
- 状态：已整理

## 摘要

LLaMA 是 Meta 首次系统公开的开放基础语言模型家族之一。论文的核心主张不是单纯追求最大参数量，而是在给定推理预算下，通过使用更多公开数据与更长训练，把 `7B-65B` 模型做成“推理成本更低但能力仍强”的开放替代方案。

## 关键事实

- LLaMA 提供 `7B / 13B / 33B / 65B` 四个主要尺寸。
- 论文明确强调：模型只使用公开可得数据集训练，不依赖闭源专有语料。
- 其重要技术立场是“在推理预算敏感的场景下，较小模型训练更久，可能比单纯堆大参数更优”。
- 文中给出的代表性结果是：`LLaMA-13B` 在多数 benchmark 上超过 `GPT-3 175B`，`LLaMA-65B` 与 `Chinchilla-70B`、`PaLM-540B` 竞争。
- 在知识库语境中，LLaMA 是开放权重大模型竞争时代的关键起点之一。

## 争议与不确定点

- 论文强调开放数据和高效训练，但其开放程度主要面向研究社区，发布策略本身仍带有限制，不等于完全开放生态。
- 该文重点是 base model，不应把后续聊天、代码和安全模型能力反向投射到 LLaMA 初代上。

## 关联页面

- 概念：[Llama](../../wiki/concepts/Llama.md)
- 概念：[LLaMA](../../wiki/concepts/LLaMA.md)
- 概念：[Llama 2](../../wiki/concepts/Llama%202.md)
- 主题：[LLM预训练](../../wiki/topics/LLM%E9%A2%84%E8%AE%AD.md)
