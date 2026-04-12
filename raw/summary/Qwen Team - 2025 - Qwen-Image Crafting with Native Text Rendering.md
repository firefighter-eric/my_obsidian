# Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen-image/
- 全文文本：../../raw/text/Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering.md
- 作者：Qwen Team
- 年份：2025
- 状态：已整理

## 摘要

Qwen-Image 是 Qwen 家族第一次明确发布图像生成基础模型。官方把它定义为 `20B` 的 `MMDiT` image foundation model，重点强调复杂文本渲染、双语文字生成与精确图像编辑，而不是只追求通用审美质量。

## 关键事实

- 官方明确写明 Qwen-Image 是 `20B MMDiT image foundation model`。
- 核心卖点有三条：复杂文本渲染、一致性图像编辑、跨基准生成与编辑性能。
- 博客列出的公开评测覆盖 GenEval、DPG、OneIG-Bench、GEdit、ImgEdit、GSO，以及文字渲染相关的 LongText-Bench、ChineseWord、TextCraft。
- 官方直接声称 Qwen-Image 在这些 generation / editing / text rendering 基准上达到或超越现有 SOTA，特别强调中文文本渲染显著领先。
- 从示例看，Qwen-Image 把多行排版、段落级文字、海报与 PPT 生成、双语渲染都视为核心能力，而非边缘 demo。
- 官方还把 Qwen-Image 描述为支持 style transfer、additions、deletions、detail enhancement、text editing 和 pose adjustment 的通用编辑模型。

## 争议与不确定点

- 当前来源是博客，不是完整技术报告，因此训练数据、对齐方式与完整架构细节仍有限。
- 博客以官方 benchmark 和 demo 为主，后续若需要更稳健判断，应补技术报告或第三方评测来源。

## 关联页面

- 概念：[Qwen-Image](../../wiki/concepts/Qwen-Image.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md)
