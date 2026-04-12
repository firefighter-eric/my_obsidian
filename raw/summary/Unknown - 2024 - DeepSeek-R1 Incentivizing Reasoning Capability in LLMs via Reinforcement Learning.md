# Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf
- 全文文本：../../raw/text/Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.md
- 作者：Unknown
- 年份：2024
- 状态：已抽取全文，待精读

## 自动抽取摘要

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without super- vised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek- R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. To support the research community, we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama. AIME 2024 (Pass@1) Codeforces (Percentile) GPQA Diamond (Pass@1) MATH-500 (Pass@1) MMLU (Pass@1) SWE-bench Verified (Resolved) 0 20 40 60 80 100 Accuracy / Percentile (%) 79.8 96.3 71.5 97.3 90.8 49.2 79.2 96.6 75.7 96.4 91.8 48.9 72.6 90.6 62.1 94.3 87.4 36.8 63.6 93.4 60.0 90.0 85.2 41.6 39.2 58.7 59.1 90.2 88.5 42.0 DeepSeek-R1 OpenAI-o1-1217 DeepSeek-R1-32B OpenAI-o1-mini DeepSeek-V3 Figure 1 | Benchmark performance of DeepSeek-R1. Contents

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM预训练](../topics/LLM预训练.md)
- 综合：暂无
