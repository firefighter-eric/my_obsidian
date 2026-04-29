# DeepSeek AI - 2025 - DeepSeek-V3.2 Release

## 来源信息

- 类型：官方发布页 / 模型发布资料
- 原始 HTML：[raw/html/DeepSeek AI - 2025 - DeepSeek-V3.2 Release.html](../../raw/html/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.html)
- 全文文本：[raw/text/DeepSeek AI - 2025 - DeepSeek-V3.2 Release.md](../../raw/text/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)
- 来源 URL：https://api-docs.deepseek.com/news/news251201
- 作者：DeepSeek AI
- 年份：2025
- 状态：已基于 DeepSeek 官方发布页整理

## 摘要

`DeepSeek-V3.2` 是 DeepSeek 在 `V3.2-Exp` 之后发布的 reasoning-first 模型节点。它的意义不只是一般模型更新，而是把 `reasoning` 与 `tool-use` 明确合并到同一条 agent 训练路线里：官方发布页称 `V3.2` 是 DeepSeek 首个把 thinking 直接集成到工具使用中的模型，并支持 thinking 与 non-thinking 两种模式下的工具使用。

在当前知识库中，`DeepSeek-V3.2` 更适合作为 `DeepSeek-R1` 与 `DeepSeek-V4` 之间的桥接节点：`R1` 证明 reasoning-oriented RL 可以直接塑造推理行为，`V3.2` 则把推理行为推进到工具调用和复杂环境任务中，`V4` 再把这种长程 agent 使用与百万 token 上下文、KV cache 压缩和混合 attention 结合起来。因此本页主要支撑 `LLM RL`、`DeepSeek 系列` 和后续 agent 主题，而不应被简单写入 `LLM 预训练` 的架构事实。

## 关键事实

- 官方将 `DeepSeek-V3.2` 与 `DeepSeek-V3.2-Speciale` 定位为 reasoning-first models built for agents。
- `DeepSeek-V3.2` 是 `V3.2-Exp` 的正式后继，发布时已在 App、Web 与 API 上可用。
- `DeepSeek-V3.2-Speciale` 主打更强 reasoning 能力，官方称其在 IMO、CMO、ICPC World Finals 和 IOI 2025 达到 gold-level 结果，但它 API-only，且发布页说明暂无 tool-use 支持。
- 官方称 `DeepSeek-V3.2` 引入新的大规模 agent 训练数据合成方法，覆盖 `1,800+` environments 与 `85k+` complex instructions。
- 官方称 `DeepSeek-V3.2` 是 DeepSeek 首个把 thinking 直接集成进 tool-use 的模型，并支持 thinking / non-thinking 两种模式下的 tool-use。
- 发布页提供 `DeepSeek-V3.2` 与 `DeepSeek-V3.2-Speciale` 的 open-source model 链接和技术报告链接。

## 争议与不确定点

- 发布页的 benchmark 和 agent 能力主张属于官方口径，仍需要更多第三方复现与统一评测支撑。
- `V3.2-Speciale` 与 `V3.2` 的能力差异不能只按分数理解：前者更偏极限 reasoning，后者才是 tool-use 与日常 agent 工作流的主要节点。
- `thinking in tool-use` 应被放在 reasoning / agent 后训练语境里讨论，不能反写成预训练本身的稳定规律。

## 关联页面

- 主题：[DeepSeek 系列](../topics/DeepSeek%20系列.md)
- 主题：[LLM RL](../topics/LLM%20RL.md)
- 概念：[DeepSeek](../concepts/DeepSeek.md)
- 概念：[DeepSeek-R1](../concepts/DeepSeek-R1.md)
- 概念：[DeepSeek-V4](../concepts/DeepSeek-V4.md)

