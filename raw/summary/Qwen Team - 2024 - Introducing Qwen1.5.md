# Qwen Team - 2024 - Introducing Qwen1.5

## 来源信息

- 类型：官方博客 / 发布说明
- 来源链接：https://qwenlm.github.io/blog/qwen1.5/
- 全文文本：../../raw/text/Qwen Team - 2024 - Introducing Qwen1.5.md
- 作者：Qwen Team
- 年份：2024
- 状态：已整理

## 摘要

Qwen1.5 是 Qwen 初代技术报告之后的一次系统化家族扩展。官方将其描述为面向开发者体验、模型覆盖面和对齐质量的统一升级：一方面把 Hugging Face `transformers` 原生支持、量化格式和主流部署框架接入做完整，另一方面把模型尺寸扩展到从 `0.5B` 到 `110B`，并加入 `MoE` 版本。

## 关键事实

- Qwen1.5 同时提供 base 与 chat 模型，覆盖 `0.5B / 1.8B / 4B / 7B / 14B / 32B / 72B / 110B`，并额外提供 `Qwen1.5-MoE-A2.7B`。
- 全系模型统一支持最长 `32K` 上下文。
- 官方明确把 Qwen1.5 定位为“更易开发和部署”的版本，强调其已并入 Hugging Face `transformers>=4.37.0`，不再依赖 `trust_remote_code`。
- 从博客给出的评测看，Qwen1.5 在多语言、长上下文、工具使用与 agent/RAG 方向上都比旧版 Qwen 更系统。
- Qwen1.5-72B-Chat 在官方叙述中已能与部分当时主流闭源/开源 chat 模型竞争，但仍落后于 GPT-4-Turbo。

## 争议与不确定点

- 本来源是官方发布博客，不是完整 technical report，训练细节和数据构成披露不如论文充分。
- 博客把多个能力面同时报告，部分 benchmark 提升来自预训练、对齐还是工程支持，边界并未完全拆开。

## 关联页面

- 概念：[Qwen](../../wiki/concepts/Qwen.md)
- 概念：[Qwen1.5](../../wiki/concepts/Qwen1.5.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
