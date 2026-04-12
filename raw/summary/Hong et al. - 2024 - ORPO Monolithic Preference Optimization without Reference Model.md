# Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf
- 全文文本：../../raw/text/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.md
- 作者：Jiwoo Hong et al.
- 年份：2024
- 状态：已基于 arXiv HTML 提取全文并精修 summary

## 自动抽取摘要或人工摘要

`ORPO` 的问题意识是：现有 preference alignment 通常仍要先做 `SFT`，再额外跑一轮偏好优化，并且常需要 reference model。论文据此提出 monolithic 的 odds-ratio preference optimization，把 SFT 与偏好对齐揉进一个训练目标里，同时去掉 reference model。它不是回到传统 `RLHF`，也不完全等于 `DPO`，而是把“reference-free、单阶段、低资源偏好优化”作为新的工程取向。

## 关键事实

- `ORPO` 把常规 `SFT` loss 与一个 odds-ratio 形式的偏好惩罚项合并，试图在单阶段训练中同时提升 chosen response、压低 rejected response。
- 该方法显式强调不需要 reference model，也不需要额外的 preference alignment phase。
- 论文将 `ORPO` 定位为比 `RLHF`、`DPO` 更资源友好的偏好优化路线。
- 在当前知识库里，`ORPO` 是“reference-free preference optimization”分支的代表方法。

## 争议与不确定点

- `ORPO` 的收益一部分来自把 SFT 与 preference alignment 合并，另一部分来自 odds ratio 本身，两者贡献在不同模型规模上的相对作用仍需谨慎区分。
- 它减少了工程复杂度，但不意味着所有需要显式约束漂移或在线探索的场景都适合用 `ORPO` 替代。
- 论文主要验证通用对话对齐场景，其在 reasoning RL 或 agent RL 中的地位仍低于 `GRPO / DAPO` 这类在线 RL 方法。

## 关联页面

- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[ORPO](../../wiki/concepts/ORPO.md)
- 概念：[DPO](../../wiki/concepts/DPO.md)

