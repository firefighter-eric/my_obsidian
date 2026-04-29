# OPD

## 简介

`OPD` 是 `On-Policy Distillation` 的缩写，中文可译为“在线策略蒸馏”或“on-policy 蒸馏”。在当前知识库中，它表示一类 LLM 后训练方法：让 student model 先按自己的当前策略生成轨迹，再在这些 student-generated trajectories 上接受 teacher model 的 token 级分布监督。

它的关键位置在 `SFT`、普通知识蒸馏与 reasoning RL 之间：它不像 off-policy distillation 那样只模仿 teacher 生成的数据，也不像 `GRPO / RLVR` 那样主要依赖稀疏 outcome reward，而是试图把 **on-policy rollout** 与 **dense token-level supervision** 结合起来。

## 关键属性

- 类型：LLM post-training / on-policy distillation / dense RL-like supervision
- 代表来源：
  - [Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation](../summaries/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)
- 当前角色：连接 teacher-student distillation、on-policy training 与 reasoning RL 的方法节点

## 相关主张

- `OPD` 的核心不是“teacher 生成答案给 student 学”，而是 **student 先生成自己的轨迹，再由 teacher 在这些轨迹上提供分布监督**。
- 与 `SFT` 式 off-policy distillation 相比，`OPD` 更贴近模型推理时会遇到的自身分布，因此更适合处理 student 自己生成过程中的错误和偏移。
- 与常规 outcome-reward RL 相比，`OPD` 可以通过 teacher logits 或 log-probability 差异提供更密集的 token 级 credit assignment。
- `G-OPD` 把标准 OPD 解释为 dense KL-constrained RL 的特殊情形，并通过 reward scaling factor 与 reference model 选择把 OPD 扩展为更一般的目标。
- `ExOPD` 是 `G-OPD` 中 reward scaling factor 大于 1 的 reward extrapolation 变体；它可能提高 math reasoning 与 code generation 表现，但也会放大长度偏置、implicit reward hacking 和训练不稳定风险。

## 来源支持

- [Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation](../summaries/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)

## 关联页面

- [LLM RL](../topics/LLM%20RL.md)
- [GRPO](./GRPO.md)
- [DAPO](./DAPO.md)
- [DPO](./DPO.md)
- [Instruction Tuning](./Instruction%20Tuning.md)
