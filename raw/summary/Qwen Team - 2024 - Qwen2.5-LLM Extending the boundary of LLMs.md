# Qwen Team - 2024 - Qwen2.5-LLM Extending the boundary of LLMs

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen2.5-llm/
- 全文文本：../../raw/text/Qwen Team - 2024 - Qwen2.5-LLM Extending the boundary of LLMs.md
- 作者：Qwen Team
- 年份：2024
- 状态：已整理

## 摘要

Qwen2.5 是 Qwen2 之后的一次高密度升级。官方把它描述为“扩展 LLM 边界”的版本，核心不只是继续放大模型，而是把家族补齐到更多生产友好尺寸，并在知识、代码、数学、结构化输出和长文本生成上同步强化。

## 关键事实

- Qwen2.5 是 decoder-only dense 家族，开放权重覆盖 `0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B`。
- 官方称其预训练数据从 Qwen2 的 `7T` token 扩展到最多 `18T` token。
- Qwen2.5 的公开定位不只是更大，而是补齐 `3B / 14B / 32B` 三个关键尺寸，以覆盖移动端和生产部署的成本区间。
- 相比 Qwen2，Qwen2.5 在知识、代码、数学、长文本、结构化数据理解和 JSON 输出上都被官方视为显著增强点。
- 大多数模型支持最长 `128K` 上下文，并可生成最长 `8K` token 输出。
- Qwen2.5 明显加强了“通用 LLM 能否直接承担代码、数学和结构化任务”的路线，而不是把这些能力完全外包给专用分支。

## 争议与不确定点

- 当前来源是官方博客，不是正式 technical report；训练配比、数据清洗与后训练配方披露有限。
- 博客同时讨论了 Qwen2.5、Qwen2.5-Coder、Qwen2.5-Math 与 API 模型，知识库中应将通用 LLM 主线与专用分支区分。

## 关联页面

- 概念：[Qwen](../../wiki/concepts/Qwen.md)
- 概念：[Qwen2.5](../../wiki/concepts/Qwen2.5.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
