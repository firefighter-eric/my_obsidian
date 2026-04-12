# Unknown - 2024 - DeepSeek-V3 Technical Report

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Unknown - 2024 - DeepSeek-V3 Technical Report.pdf
- 全文文本：../../raw/text/Unknown - 2024 - DeepSeek-V3 Technical Report.md
- 作者：Unknown
- 年份：2024
- 状态：已抽取全文，待精读

## 自动抽取摘要

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architec- tures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3. MMLU-Pro (EM) GPQA-Diamond (Pass@1) MATH 500 (EM) AIME 2024 (Pass@1) Codeforces (Percentile) SWE-bench Verified (Resolved) 0 20 40 60 80 100 Accuracy / Percentile (%) 75.9 59.1 90.2 39.2 51.6 42.0 66.2 41.3 74.7 16.7 35.6 22.6 71.6 49.0 80.0 23.3 24.8 23.8 73.3 51.1 73.8 23.3 25.3 24.5 72.6 49.9 74.6 9.3 23.6 38.8 78.0 65.0 78.3 16.0 20.3 50.8 DeepSeek-V3 DeepSeek-V2.5 Qwen2.5-72B-Inst Llama-3.1-405B-Inst GPT-4o-0513 Claude-3.5-Sonnet-1022 Figure 1 | Benchmark performance of DeepSeek-V3 and its counterparts. Contents

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Unknown - 2024 - DeepSeek-V3 Technical Report.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM预训练](../topics/LLM预训练.md)
- 综合：暂无
