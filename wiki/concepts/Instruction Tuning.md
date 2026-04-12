# Instruction Tuning

## 简介

Instruction Tuning 指“使用带自然语言指令的数据集对预训练模型继续微调”的方法。在当前知识库中，它是连接基础预训练模型与后续对齐方法的中间层概念。

## 关键属性

- 类型：post-training / 指令微调方法
- 代表来源：
  - [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../raw/summary/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
  - [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- 当前角色：把预训练能力转成可用交互能力的早期通用方法节点

## 相关主张

- `Wei et al. 2021` 表明，在多任务自然语言指令数据上微调可以显著改善模型对未见任务的 zero-shot 泛化。
- 在当前知识库中，Instruction Tuning 不是 RLHF 的替代，而是其前置层或相邻层，用于先把模型行为拉向“遵循指令”。
- `Ouyang et al. 2022` 里的 supervised fine-tuning 阶段也可被视为更完整对齐管线中的 instruction tuning 组成部分。

## 来源支持

- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../raw/summary/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)

## 关联页面

- [InstructGPT](./InstructGPT.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)
- [LLM 预训练](LLM%20预训练.md)
