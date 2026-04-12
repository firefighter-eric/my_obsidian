# Wang et al. - 2025 - VRAG-RL Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning wit

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wang et al. - 2025 - VRAG-RL Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning wit.pdf
- 全文文本：../../raw/text/Wang et al. - 2025 - VRAG-RL Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning wit.md
- 作者：Wang et al.
- 年份：2025
- 状态：已抽取全文，待精读

## 自动抽取摘要

Effectively retrieving, reasoning and understanding visually rich information remains a challenge for traditional Retrieval-Augmented Generation (RAG) methods. On the one hand, traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As reinforcement learning (RL) has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing con- tinual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incor- porate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search en- gines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse- to-fine perspective. Furthermore, to bridge the gap between users’ original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewrit- ing and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. Extensive experiments on diverse and challenging benchmarks show that our VRAG-RL outperforms existing methods by 20% (Qwen2.5-VL-7B) and 30% (Qwen2.5-VL-3B), demonstrating the effectiveness of our approach. The code is available at https://github.com/Alibaba-NLP/VRAG.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Wang et al. - 2025 - VRAG-RL Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning wit.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
