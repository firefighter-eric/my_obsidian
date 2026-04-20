# Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression.pdf
- 全文文本：../../raw/text/Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression.md
- 作者：Wei, Sun, Li
- 年份：2025
- 状态：已整理（基于 arXiv HTML 提取全文）

## 摘要

`DeepSeek-OCR` 的核心目标不是传统意义上的“更准 OCR”，而是把 OCR 变成验证视觉压缩是否能替代长文本上下文的一块实验场。它把文档文本先映射成高分辨率图像，再用 `DeepEncoder + DeepSeek3B-MoE` 做 token compression 与解码，试图证明视觉 modality 可以在长上下文问题上作为更省 token 的中介层。也因此，`DeepSeek-OCR` 在当前知识库中的意义并不只是文档解析模型，而是 OCR 与 long-context efficiency 交叉的一条特殊路线。

## 关键事实

- 论文把 `DeepSeek-OCR` 明确写成对 long-context compression 的初步研究，而不是只追求 OCR benchmark；其主问题是“视觉是否能作为文本的高压缩表示”。
- 方法由 `DeepEncoder` 与 `DeepSeek3B-MoE-A570M` 解码器组成，强调高分辨率输入下的低激活和高压缩比，以控制进入 LLM 的 vision tokens 数量。
- 论文报告当 text tokens 与 vision tokens 压缩比在 `10x` 以内时，OCR precision 可到约 `97%`；即便在 `20x` 压缩下，仍保留约 `60%` OCR 准确率。这个结论使 DeepSeek-OCR 在 OCR topic 里代表“token economy / vision-text compression”而不是普通识别路线。
- 在 `OmniDocBench` 语境下，论文强调它以极少的 vision tokens 达到强性能：相较 `GOT-OCR2.0` 与 `MinerU2.0`，其主要卖点不是绝对最强解析质量，而是更低 token 成本下的实用性能。
- 文中还把生产能力写得很重：单张 `A100-40G` 可做到 `200k+ pages/day`，20 个节点可达 `33M pages/day` 数据生成能力，这说明其定位也包括为 LLM/VLM 生产训练数据。
- 数据引擎覆盖 `OCR 1.0`、`OCR 2.0`、general vision 与 text-only data，并将文档 OCR、chart、chemical formula、plane geometry 一并纳入，说明它试图把 OCR 拓展为更广的 image-to-structured-text compression 问题。

## 争议与不确定点

- 这篇初代报告的主叙事偏“研究范式验证”而非完整 document parsing 产品，因此它的强项与 `PaddleOCR / GLM-OCR / dots.ocr` 这类面向稳定结构输出的系统并不完全同维度。
- OCR accuracy 与 compression ratio 的关系是论文最核心的论点，但不同任务对格式保真和结构保真的要求并不相同，因此不能把该结果直接外推到所有文档解析场景。
- 由于该模型很快在 `2026-01-28` 被 `DeepSeek-OCR 2` 更新，当前更稳妥的理解方式是把本页视为 DeepSeek OCR 家族的初代范式页。

## 关联页面

- 概念：[DeepSeek-OCR](../../wiki/concepts/DeepSeek-OCR.md)
- 来源后续：[Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](./Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
