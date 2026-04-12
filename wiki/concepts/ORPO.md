# ORPO

## 简介

ORPO 是 `Odds Ratio Preference Optimization` 的缩写。在当前知识库中，它表示把 supervised fine-tuning 与 preference alignment 合并到单阶段训练中的 reference-free 偏好优化方法。

## 关键属性

- 类型：偏好优化 / reference-free alignment
- 代表来源：
  - [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../../raw/summary/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- 当前角色：`DPO` 之后进一步压缩工程复杂度的偏好优化节点

## 相关主张

- `ORPO` 把 `SFT` loss 与 odds-ratio 偏好惩罚合并，试图避免单独的 SFT warmup 与 reference model。
- 它强调“monolithic preference alignment”，即在同一训练阶段里同时完成域适配与偏好区分。
- 在当前知识库里，`ORPO` 代表“reference-free、单阶段、低资源”的偏好优化分支，而不是在线 RL 路线。

## 来源支持

- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../../raw/summary/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [LLM RL](../topics/LLM%20RL.md)

## 关联页面

- [DPO](./DPO.md)
- [KTO](./KTO.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)

