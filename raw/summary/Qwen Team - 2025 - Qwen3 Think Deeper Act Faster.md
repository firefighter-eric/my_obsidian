# Qwen Team - 2025 - Qwen3 Think Deeper Act Faster

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen3/
- 全文文本：../../raw/text/Qwen Team - 2025 - Qwen3 Think Deeper Act Faster.md
- 作者：Qwen Team
- 年份：2025
- 状态：已整理

## 摘要

Qwen3 标志着 Qwen 家族从“强通用 LLM”进一步转向“可控思考预算 + agent 能力 + 更大规模 MoE”的阶段。官方把它的核心卖点概括为 hybrid thinking：同一个模型既支持深度思考模式，也支持快速直接回答模式。

## 关键事实

- Qwen3 同时开放 `2` 个 MoE 模型与 `6` 个 dense 模型：
  - MoE：`Qwen3-235B-A22B`、`Qwen3-30B-A3B`
  - Dense：`0.6B / 1.7B / 4B / 8B / 14B / 32B`
- Qwen3 支持 `thinking mode` 与 `non-thinking mode` 两种推理风格，官方把这视为代际核心变化。
- 官方写明 Qwen3 支持 `119` 种语言和方言。
- 预训练 token 规模从 Qwen2.5 的 `18T` 扩展到约 `36T`，并显式使用 Qwen2.5-VL 处理 PDF 类文档、用 Qwen2.5 / Math / Coder 生成部分高质量合成数据。
- post-training 采用四阶段流程：长 CoT 冷启动、reasoning RL、thinking/non-thinking 融合、general RL。
- Qwen3 还被明确优化为更强的 `agentic capabilities` 与 `MCP` 使用能力。

## 争议与不确定点

- 当前来源是官方博客，不是独立 technical report。
- Hybrid thinking 的收益高度依赖预算控制与具体 deployment 方式，博客展示的是官方推荐使用情境。

## 关联页面

- 概念：[Qwen3](../../wiki/concepts/Qwen3.md)
- 概念：[Qwen3.5](../../wiki/concepts/Qwen3.5.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
