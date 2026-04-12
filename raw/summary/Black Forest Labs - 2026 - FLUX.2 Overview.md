# Black Forest Labs - 2026 - FLUX.2 Overview

## 来源信息

- 类型：官方文档 / 模型总览
- 来源链接：https://docs.bfl.ai/flux_2/flux2_overview
- 全文文本：../../raw/text/Black Forest Labs - 2026 - FLUX.2 Overview.md
- 作者：Black Forest Labs
- 年份：2026
- 状态：已整理

## 摘要

FLUX.2 被官方定义为覆盖从实时生成到最高质量资产生产的图像生成与编辑家族。它的重点不只是单次文生图质量，而是把多参考图编辑、文本渲染、精确颜色控制、结构化提示和生产工作流统一进一个生产级视觉模型体系。

## 关键事实

- 官方把 FLUX.2 描述为从 sub-second inference 到最高质量输出的统一家族，覆盖 `[klein] / [max] / [pro] / [flex] / [dev]` 多个变体。
- FLUX.2 支持最多 `10` 张参考图（`klein` 较少），强调 character consistency、多参考编辑和跨场景组合能力。
- 官方明确列出 up to `4MP` 输出、任意宽高比、hex 精准颜色控制、pose guidance、structured prompting 等面向生产工作流的控制项。
- `FLUX.2 [flex]` 被官方明确定位为 typography specialist，说明文本渲染已从演示能力变成专门优化方向。
- 文档写明 `[klein] 4B` 可在约 `13GB VRAM` 消费级硬件上运行，并以 Apache 2.0 方式开放；`[klein] 9B` 则与 `8B Qwen3 text embedder` 组合。
- `FLUX.2 [max]` 还支持 grounding search，即在需要时搜索网络实时信息后再生成图像，这意味着它已开始把“世界知识接入”直接放入图像生成工作流。

## 争议与不确定点

- 该来源是产品文档，不是完整研究论文，对完整架构、训练配方与评测细节披露有限。
- 从文档可确认 FLUX.2 是当前生产级图像生成/编辑家族，但不能仅凭本页断定全部内部研究细节。

## 关联页面

- 概念：[FLUX.2](../../wiki/concepts/FLUX.2.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
