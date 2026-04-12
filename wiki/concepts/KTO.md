# KTO

## 简介

KTO 是 `Kahneman-Tversky Optimization` 的缩写。在当前知识库中，它表示一条不依赖成对 preference data、而是直接利用 desirable / undesirable 二元反馈做对齐的偏好优化路线。

## 关键属性

- 类型：偏好优化 / human-aware loss
- 代表来源：
  - [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../raw/summary/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- 当前角色：把偏好优化从 pairwise preference 扩展到 unary desirability signal 的代表方法

## 相关主张

- `KTO` 认为高质量对齐未必要求成对偏好比较，只要能判断一个响应是 desirable 还是 undesirable，就能构造有效目标。
- 该方法把 `DPO`、`PPO` 等对齐损失放进 `HALO` 框架，并以 prospect theory 作为设计 `KTO` 的理论动机。
- 在当前知识库里，`KTO` 的重要性在于它改变了后训练的数据接口假设，而不只是对 `DPO` 做小幅公式改写。

## 来源支持

- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../raw/summary/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- [LLM RL](../topics/LLM%20RL.md)

## 关联页面

- [DPO](./DPO.md)
- [ORPO](./ORPO.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)

