# Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency.pdf
- 全文文本：../../raw/text/Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency.md
- 作者：Jiang et al.
- 年份：2025
- 状态：已抽取全文，待精读

## 自动抽取摘要

Answering questions with Chain-of-Thought (CoT) has significantly enhanced the reasoning ca- pabilities of Large Language Models (LLMs), yet its impact on Large Multimodal Models (LMMs) still lacks a systematic assessment and in-depth investigation. In this paper, we introduce MME- CoT, a specialized benchmark evaluating the CoT reasoning performance of LMMs, spanning six domains: math, science, OCR, logic, space-time, and general scenes. As the first comprehensive study in this area, we propose a thorough evalu- ation suite incorporating three novel metrics that assess the reasoning quality, robustness, and effi- ciency at a fine-grained level. Leveraging curated high-quality data and a unique evaluation strategy, we conduct an in-depth analysis of state-of-the- art LMMs, uncovering several key insights: 1) Models with reflection mechanism demonstrate a superior CoT quality, with Kimi k1.5 outperform- ing GPT-4o and demonstrating the highest quality results; 2) CoT prompting often degrades LMM performance on perception-heavy tasks, suggest- ing a potentially harmful overthinking behavior; and 3) Although the CoT quality is high, LMMs with reflection exhibit significant inefficiency in both normal response and self-correction phases. We hope MME-CoT serves as a foundation for advancing multimodal reasoning in LMMs. 1. Introduction The emergence of Chain-of-Thought (CoT) (Wei et al., 2022) in Large Language Models (LLMs) has demonstrated promising advances in reasoning capabilities, exemplified GPT-4o QVQ-72B Virgo-72B InternVL2-5-78B-MPO Qwen2-VL-72B Precision Recall Stability Efficacy Relevance Rate 85.4 80.2 79.5 77.3 73.6 51.2 50.5 49.2 44.2 41.1 -6.5 -3.1 -2.0 -1.7 -1.0 -2.9 -0.4 2.4 5.1 83.7 90.6 92.2 92.9 60.6 61.7 100.0 Reflection Quality 92.0 Efficiency Quality Robustness Kimi k1.5 92.0 49.3 2.9 0.0 72.2 Figure 1: Chain-of-Thought Performance of Leading LMMs in MME-CoT. Our evaluation suite assesses LMMs using three novel metrics that yield six distinct scores. Re- sults reveal that current open-source models, including those with reflection capabilities, still lag behind closed-source models like GPT-4o and Kimi k1.5 in key aspects of chain- of-thought reasoning. by the recent OpenAI o1 (OpenAI, 2024a) and DeepSeek- R1 (Guo et al., 2025a). By engaging in a more deliberate, stepwise reasoning process before reaching a final answer, this methodology presents an effective solution in tackling complex scenarios.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
