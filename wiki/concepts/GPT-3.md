# GPT-3

## 简介

GPT-3 是 `Brown et al. 2020` 报告的大规模自回归语言模型，也是当前知识库里“通过扩大预训练规模获得 few-shot 泛化能力”的关键概念节点。

## 关键属性

- 类型：大规模自回归语言模型
- 代表来源：
  - [Brown et al. - 2020 - Language models are few-shot learners](../../raw/summary/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
  - [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- 当前角色：预训练规模化阶段的代表模型，也是后续指令对齐路线的重要基座

## 相关主张

- `Brown et al. 2020` 将 GPT-3 作为“纯提示式 few-shot 使用”的代表，强调在不做梯度更新的情况下，模型可通过上下文示例完成多类任务。
- 在当前知识库中，GPT-3 代表的是“规模扩张本身能够带来更强任务适应能力”这一主张。
- `Ouyang et al. 2022` 又把 GPT-3 作为 InstructGPT 的底座，说明后训练对齐不是替代 GPT-3，而是在其基础上修正“能力强但未必符合用户意图”的问题。

## 来源支持

- [Brown et al. - 2020 - Language models are few-shot learners](../../raw/summary/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [LLM 基础脉络](../topics/LLM%20基础脉络.md)
- [Scaling 与 compute-optimal training](../topics/Scaling%20与%20compute-optimal%20training.md)

## 关联页面

- [LLM 预训练](LLM%20预训练.md)
- [LLM 基础脉络](../topics/LLM%20基础脉络.md)
- [Scaling 与 compute-optimal training](../topics/Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)
- [InstructGPT](./InstructGPT.md)
