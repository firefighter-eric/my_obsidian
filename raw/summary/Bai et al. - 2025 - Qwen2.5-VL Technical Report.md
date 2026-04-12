# Bai et al. - 2025 - Qwen2.5-VL Technical Report

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Bai et al. - 2025 - Qwen2.5-VL Technical Report.pdf
- 全文文本：../../raw/text/Bai et al. - 2025 - Qwen2.5-VL Technical Report.md
- 作者：Bai et al.
- 年份：2025
- 状态：已整理（基于 PDF 重建全文）

## 摘要

`Qwen2.5-VL Technical Report` 表明 Qwen 的视觉语言路线已经从“图文问答模型”转为“原生处理文档、图表、长视频与 GUI 交互的通用多模态系统”。它的重点不只是多模态输入更多，而是用 native resolution、绝对时间建模、统一文档 HTML 表达和 agent 轨迹数据，把视觉理解、结构化解析和设备操作能力压进同一条 VL 主线。这也是 Qwen 从 LLM 分支外延走向 multimodal system 的关键节点。

## 关键事实

- 模型公开覆盖 `3B / 7B / 72B` 三个尺寸，明确面向 edge 到 flagship 的不同部署区间，而不只是单个大模型展示。
- 与 Qwen2-VL 相比，预训练数据从约 `1.2T` token 扩展到约 `4T` token，数据类型也显著扩展到 OCR、文档解析、视频定位、grounding 与 agent 轨迹。
- 视觉编码器不是沿用现成 ViT，而是从头训练 native dynamic-resolution ViT，并在大多数层使用 `Window Attention`，只保留少数全局注意力层，以降低原生分辨率带来的计算负担。
- 空间建模的核心变化是直接使用图像真实尺寸中的绝对坐标表示 bounding box 与 point，而不是标准化相对坐标；这使 grounding、counting 和文档定位更贴近真实页面尺度。
- 时间建模的核心变化是把 `MRoPE` 从 Qwen2-VL 的“按帧编号”推进到“与绝对时间对齐”，再配合动态 FPS 训练，使模型可以更稳定地理解长视频、时间戳和秒级事件定位。
- 文档路线不是简单 OCR。论文专门构造了 `QwenVL HTML` 格式，把段落、表格、图表、公式、图片说明、乐谱、化学式及其 bbox 统一表示，目标是让单模型同时完成解析、理解和格式转换。
- OCR 与文档数据覆盖多语言场景，并引入大规模图表、表格与真实文档样本；这解释了它为何在文档、图表和 diagram 理解上被作者视为主打优势。
- 视觉 agent 数据明确覆盖 mobile、web、desktop 三类界面，并把动作统一成共享函数调用空间，再配套多步轨迹与步骤级 reasoning 监督。这说明 Qwen2.5-VL 的 agent 能力来自专门数据和训练，而不是“VL 模型自然涌现”。
- 训练流程不仅有预训练，还包含 SFT 与 DPO；其中 SFT 覆盖 image-text、video、OCR、grounding、agent 等任务，DPO 进一步做偏好优化。
- 论文的总体含义是：Qwen2.5-VL 不再只是“给 LLM 加一只眼睛”，而是在把视觉理解、结构化感知与设备交互合并成更接近通用多模态执行器的路线。

## 争议与不确定点

- 当前全文来自本地 PDF 抽取重建，正文信息已可用，但表格和局部数学排版仍可能有抽取误差。
- 报告中的大量性能比较来自作者选定 benchmark；它足以说明研究方向，但不自动等于真实世界所有视觉代理任务都同样稳健。
- 论文强调 native resolution、document omni-parsing 与 visual agent 的统一性，但这条路线与后续 Qwen2.5-Omni、Qwen3.5 native multimodal agent 的边界，仍需要更多后续技术报告来进一步澄清。

## 关联页面

- 概念：[Qwen2-VL](../../wiki/concepts/Qwen2-VL.md)
- 概念：[Qwen2.5-VL](../../wiki/concepts/Qwen2.5-VL.md)
- 概念：[Qwen2.5-Omni](../../wiki/concepts/Qwen2.5-Omni.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
- 主题：[传统 CV](传统%20CV.md)
