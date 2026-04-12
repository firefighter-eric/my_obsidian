# Qwen Team - 2024 - Qwen2-VL

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen2-vl/
- 全文文本：../../raw/text/Qwen Team - 2024 - Qwen2-VL.md
- 作者：Qwen Team
- 年份：2024
- 状态：已整理

## 摘要

Qwen2-VL 是 Qwen 家族把语言底座系统扩展到视觉语言模型的重要节点。官方将其定位为既能做图像理解，也能做视频理解、视觉 agent 与文档/表格读取的通用多模态模型，而不只是图文问答模型。

## 关键事实

- Qwen2-VL 延续了 “ViT + Qwen2 LLM” 的大体结构，但对视觉分辨率和多模态位置编码做了关键升级。
- 官方强调两项核心架构改动：
  - `Naive Dynamic Resolution`：允许图像映射为动态数量的视觉 token，而不是固定分辨率裁剪。
  - `M-ROPE`：把文本、图像、视频的位置信息统一到新的多模态 rotary position embedding 中。
- Qwen2-VL 明确覆盖图片、视频、文档、表格、定位与视觉交互任务，并展示了 function calling、UI interaction、screen sharing 等 agent 化能力。
- 开源权重主要为 `2B` 与 `7B`，`72B` 通过 API 提供。
- 官方把 Qwen2-VL 看作通向更完整 omni model 的中间阶段。

## 争议与不确定点

- 当前来源是官方博客，未在本地补齐其技术报告全文。
- 视觉 agent 示例带有演示性质，真实稳定性与泛化边界仍需更细粒度 benchmark 支撑。

## 关联页面

- 概念：[Qwen2-VL](../../wiki/concepts/Qwen2-VL.md)
- 概念：[Qwen2.5-VL](../../wiki/concepts/Qwen2.5-VL.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
