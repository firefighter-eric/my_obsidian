# Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow.pdf
- 全文文本：../../raw/text/Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow.md
- 作者：Wei, Sun, Li
- 年份：2026
- 状态：已整理（基于 arXiv HTML 提取全文）

## 摘要

`DeepSeek-OCR 2` 是对初代 `DeepSeek-OCR` 的直接架构升级。它保留“vision-text compression”这一家族主轴，但把重点从单纯压缩比推进到视觉 token 是否能按语义因果流重新排序。论文认为传统 VLM 以 raster-scan 顺序处理视觉 token，并不符合人类阅读复杂页面时的语义驱动扫描，因此提出 `DeepEncoder V2` 与 causal flow query，让视觉编码先完成一次“语义排序”，再送进 LLM 解码。对 OCR topic 来说，这一页代表的是“reading order / causal visual flow 被直接写进编码器”的路线。

## 关键事实

- `DeepSeek-OCR 2` 的核心创新是 `DeepEncoder V2`，它不再满足于压缩视觉 token，而是尝试根据图像语义动态重排 token 顺序，以更接近 human-like document reading。
- 论文明确把这种方法写成“two-cascaded 1D causal reasoning structures”来逼近 2D reasoning：编码器先做 causal visual flow，解码器再做 autoregressive reasoning。
- 与初代相比，作者声称 `DeepSeek-OCR 2` 在保持压缩率与解码效率的同时，在 `OmniDocBench v1.5` 上带来 `3.73%` 的提升；这说明家族升级重点是视觉顺序建模而非单纯扩大参数。
- 模型继续限制送入 LLM 的视觉 token 预算在 `256-1120` 范围内，并明确把这一点与 `Gemini-3 Pro` 的视觉 token budget 对齐，显示其研究仍然围绕 efficiency-aware OCR / parsing 展开。
- 数据方面，`DeepSeek-OCR 2` 基本沿用初代的 `OCR 1.0 / OCR 2.0 / general vision` 数据引擎，仅做更平衡采样与标签合并，这意味着它更像架构升级版而不是完全新任务定义。
- 文中将 `Marker`、`MinerU2`、`Dolphin` 等作为比较对象，说明它要竞争的仍是 specialized OCR / parsing 系统，而不是一般 VL chat 模型。

## 争议与不确定点

- “causal flow query” 是否真正对应更强的 2D reasoning，目前仍主要由作者自己的架构解释和 benchmark 改进支撑；它很有研究吸引力，但还不是已被外部广泛确认的稳定共识。
- 这条路线和经典 OCR / parsing 工具链并不完全同题。它解决的是 OCR 中最前沿的编码器组织问题，而不是工程系统完备性；因此更适合被放在 OCR topic 的前沿 specialized model 层，而非 toolkit 层。
- 当前页与初代 `DeepSeek-OCR` 关系非常紧密，实际使用时应将二者理解成同一家族中的两个节点，而不是彼此独立的范式。

## 关联页面

- 概念：[DeepSeek-OCR](../../wiki/concepts/DeepSeek-OCR.md)
- 家族前序：[Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](./Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
