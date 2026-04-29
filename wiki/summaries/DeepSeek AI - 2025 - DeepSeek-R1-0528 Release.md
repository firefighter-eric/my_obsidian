# DeepSeek AI - 2025 - DeepSeek-R1-0528 Release

## 来源信息

- 类型：官方发布页 / 模型更新资料
- 原始 HTML：[raw/html/DeepSeek AI - 2025 - DeepSeek-R1-0528 Release.html](../../raw/html/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.html)
- 全文文本：[raw/text/DeepSeek AI - 2025 - DeepSeek-R1-0528 Release.md](../../raw/text/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- 来源 URL：https://api-docs.deepseek.com/news/news250528
- 作者：DeepSeek AI
- 年份：2025
- 状态：已基于 DeepSeek 官方发布页整理

## 摘要

`DeepSeek-R1-0528` 是 `DeepSeek-R1` 的后续更新节点。它不改变 `R1` 在知识库中的核心定位：`R1` 仍代表通过大规模 RL 直接激励推理能力的 reasoning-oriented 后训练路线；但 `R1-0528` 补充说明这一模型线在发布后继续向更可用的交互接口推进，包括 benchmark 提升、前端能力增强、幻觉降低，以及 JSON output / function calling 支持。

在当前知识库中，本页主要用于修正一个容易过窄的理解：`DeepSeek-R1` 不是只停留在数学或代码 benchmark 上的研究模型，它的后续更新已经把推理模型推向更稳定的 API 与工具调用接口。不过，本页来源是短官方发布页，不能替代 `DeepSeek-R1` 技术报告来支撑 RL 训练机制的细节。

## 关键事实

- 官方发布页称 `DeepSeek-R1-0528` 提升了 benchmark performance。
- 官方发布页称该版本增强 front-end capabilities。
- 官方发布页称该版本 reduced hallucinations。
- 官方发布页称该版本支持 JSON output 与 function calling。
- 官方称 API 使用方式不变，并提供 open-source weights 链接。

## 争议与不确定点

- 发布页没有提供详细消融实验，因此不能用它来解释幻觉降低或 benchmark 提升来自哪一类训练变化。
- `JSON output / function calling` 是可用性与工具接口事实，不应直接写成 reasoning RL 算法本身的改进。
- 本页适合作为 `R1` 后续产品化与工具化更新的证据，不适合作为 `GRPO` 或 RL 机制细节的一手技术来源。

## 关联页面

- 主题：[DeepSeek 系列](../topics/DeepSeek%20系列.md)
- 主题：[LLM RL](../topics/LLM%20RL.md)
- 概念：[DeepSeek](../concepts/DeepSeek.md)
- 概念：[DeepSeek-R1](../concepts/DeepSeek-R1.md)
- 概念：[GRPO](../concepts/GRPO.md)

