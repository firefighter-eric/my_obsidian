# Roziere et al. - 2023 - Code Llama Open Foundation Models for Code

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Roziere et al. - 2023 - Code Llama Open Foundation Models for Code.pdf
- 全文文本：../../raw/text/Roziere et al. - 2023 - Code Llama Open Foundation Models for Code.md
- 作者：Roziere et al.
- 年份：2023
- 状态：已整理

## 摘要

Code Llama 是基于 Llama 2 继续特化出的代码大模型家族。它的意义不只是“Llama 2 会写代码”，而是把开放基础模型经由代码持续训练、长上下文扩展与 infilling 目标，变成一个面向编程任务的专门分支。

## 关键事实

- Code Llama 提供 `7B / 13B / 34B / 70B` 多个尺寸。
- 家族同时包含 `Code Llama`、`Code Llama - Python` 与 `Code Llama - Instruct` 三类变体。
- 论文强调两个关键强化方向：
  - `infilling`：支持根据前后文补全中间代码片段。
  - `long context`：在 `16K` 训练长度基础上，对输入可扩展到接近 `100K` 的长上下文。
- Code Llama 以 Llama 2 为底座而不是从零做纯代码预训练，这体现了开放基础模型向专用代码分支外溢的路线。
- 文中声称其在 HumanEval、MBPP、MultiPL-E 等代码 benchmark 上达到当时开放模型 SOTA。

## 争议与不确定点

- Code Llama 是专门代码分支，不应把其结果直接代表整个 Llama 家族的通用能力。
- 论文对“长上下文到 100K”的描述包含 RoPE 扩展与特定设置，实际可用性依赖部署和任务形式。

## 关联页面

- 概念：[Llama](../../wiki/concepts/Llama.md)
- 概念：[Llama 2](../../wiki/concepts/Llama%202.md)
- 概念：[Code Llama](../../wiki/concepts/Code%20Llama.md)
- 主题：[LLM预训练](../../wiki/topics/LLM%E9%A2%84%E8%AE%AD.md)
