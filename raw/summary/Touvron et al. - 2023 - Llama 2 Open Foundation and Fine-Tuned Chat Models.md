# Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models.pdf
- 全文文本：../../raw/text/Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models.md
- 作者：Touvron et al.
- 年份：2023
- 状态：已整理

## 摘要

Llama 2 是 LLaMA 初代之后的正式升级版本。它不仅继续提供基础模型，还把 `Llama 2-Chat` 作为系统 post-training 结果公开出来，因此它在开放模型语境中的意义不只是“更强 base model”，而是“开放 foundation model + fine-tuned chat model” 的成体系发布。

## 关键事实

- Llama 2 覆盖 `7B / 13B / 70B` 三个主要尺寸，并同时提供 pretrained 与 chat 版本。
- 论文明确将 `Llama 2-Chat` 描述为面向对话优化的模型家族。
- 该文系统报告了 `SFT + RLHF` 的后训练与安全改进流程，是 Llama 家族从 base model 走向可直接交互模型的重要节点。
- 文中主张 Llama 2-Chat 在多数所测 benchmark 上超过当时开放 chat 模型，并在人工 helpfulness / safety 评价中接近闭源模型替代品。
- 在知识库结构上，Llama 2 是连接早期 LLaMA 与后续 Llama 3 的中间代际。

## 争议与不确定点

- “可作为闭源模型替代品”是论文基于当时评测与人工标注给出的结论，不应直接无条件外推到所有任务。
- 该文同时讨论 foundation model、chat model 和安全流程，使用时需要区分“预训练能力”与“对齐后行为”。

## 关联页面

- 概念：[Llama](../../wiki/concepts/Llama.md)
- 概念：[LLaMA](../../wiki/concepts/LLaMA.md)
- 概念：[Llama 2](../../wiki/concepts/Llama%202.md)
- 概念：[Code Llama](../../wiki/concepts/Code%20Llama.md)
- 主题：[LLM预训练](../../wiki/topics/LLM%E9%A2%84%E8%AE%AD.md)
