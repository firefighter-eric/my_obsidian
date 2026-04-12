# Bai et al. - 2023 - Qwen Technical Report

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Bai et al. - 2023 - Qwen Technical Report.pdf
- 全文文本：../../raw/text/Bai et al. - 2023 - Qwen Technical Report.md
- 作者：Bai et al.
- 年份：2023
- 状态：已整理（基于 PDF 重建全文）

## 摘要

`Qwen Technical Report` 是 Qwen 家族的起点文档。它确立的不是单一 chat 模型，而是一整套“基础模型 + 对齐 chat 模型 + 代码 / 数学专门模型 + 工具 / agent 能力”的家族框架。与后续 Qwen1.5/2/2.5 相比，这一代最重要的贡献不是参数规模本身，而是先把中文/英文为核心的多语言 tokenizer、3T 级预训练、SFT+RLHF 对齐、长上下文外推，以及 tool use / code interpreter / agent 场景放进同一条产品与研究叙事里。

## 关键事实

- Qwen1 的主干基础模型公开覆盖 `1.8B / 7B / 14B` 三个尺寸，论文同时把 `Qwen-Chat`、`Code-Qwen`、`Math-Qwen-Chat` 与已发布的 `Qwen-VL` 纳入同一 lineage。
- 预训练数据规模最高到 `3T` token，其中英语与中文占核心比例，但数据集明确是多语言的，并包含 web、百科、书籍、代码等多种来源。
- tokenizer 基于 `tiktoken` 的 `cl100k_base` 扩展，最终词表约 `152K`；论文强调它在中文、英文、代码以及多种其他语言上都具有较高压缩率，这解释了 Qwen 后续一直重视多语言与部署效率。
- 架构上总体沿用 LLaMA 风格 decoder-only Transformer，但做了几项关键改造：`untied embedding`、`RoPE`、QKV bias、`RMSNorm`、`SwiGLU`，并在训练中使用 `FlashAttention`。
- 训练上下文长度是 `2048`，但论文系统讨论了用 `NTK-aware interpolation + LogN scaling + window attention` 在推理时把上下文扩展到 `8K+`，这可视为后来 Qwen 系列长上下文路线的早期工程基础。
- 对齐流程不是只做 SFT；论文明确写了 `SFT + reward model + RLHF`，并引入 `pretrained gradient` 来缓解 alignment tax。
- 论文已把 tool use、code interpreter 和 agent 能力写成正式评测项，而不是演示附录。这说明 Qwen 从第一代起就把“可行动的 chat model”作为目标，而非只做通用问答模型。
- 从家族方法论看，Qwen1 已建立一种稳定模式：先做通用底座，再通过后训练和专门数据把代码、数学、工具调用与多模态能力逐步外溢为分支。

## 争议与不确定点

- 当前全文来自本地 PDF 抽取重建，虽比原先损坏的 ar5iv 文本完整，但表格与局部排版仍可能存在抽取噪声。
- 论文所强调的 tool use / code interpreter / agent 能力，多数来自作者自建评测或当时可用 benchmark；把它直接等同于后续“成熟 agent 框架”并不稳妥。
- 这一代的多语言与长上下文能力已经明显成形，但与 Qwen2、Qwen3 相比，仍更像“为后续家族扩张打底”的阶段，而不是最终成熟形态。

## 关联页面

- 概念：[Qwen](../../wiki/concepts/Qwen.md)
- 概念：[Qwen1.5](../../wiki/concepts/Qwen1.5.md)
- 主题：[LLM 预训练](LLM%20预训练.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
