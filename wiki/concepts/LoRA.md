# LoRA

## 简介

LoRA 是 Low-Rank Adaptation 的缩写。在当前知识库中，它表示“以低秩增量参数替代全参数更新”的参数高效微调方法。

## 关键属性

- 类型：参数高效微调方法
- 代表来源：
  - [Hu et al. - 2021 - LoRA Low-Rank Adaptation of Large Language Models](../../raw/summary/Hu%20et%20al.%20-%202021%20-%20LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.md)
  - [Sun et al. - 2023 - A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Instruction Fo](../../raw/summary/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.md)
- 当前角色：连接基础模型与低成本指令微调实践的通用方法节点

## 相关主张

- `Hu et al. 2021` 提出冻结原模型参数、只训练低秩适配矩阵，从而显著减少可训练参数量。
- 在当前知识库里，LoRA 代表的核心不是单一模型家族，而是“让大模型后训练更便宜、更易部署”的通用技术。
- `Sun et al. 2023` 进一步说明，在中文 instruction tuning 场景下，LoRA 与全参数微调之间存在成本与效果的现实权衡。

## 来源支持

- [Hu et al. - 2021 - LoRA Low-Rank Adaptation of Large Language Models](../../raw/summary/Hu%20et%20al.%20-%202021%20-%20LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.md)
- [Sun et al. - 2023 - A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Instruction Fo](../../raw/summary/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)

## 关联页面

- [InstructGPT](./InstructGPT.md)
- [Qwen](./Qwen.md)
- [Llama 3](./Llama 3.md)
- [LLM RL](../topics/LLM%20RL.md)
- [传统 NLP](传统%20NLP.md)
