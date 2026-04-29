# Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation

## 来源信息

- 原始 PDF：[raw/pdf/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.pdf](../../raw/pdf/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.pdf)
- 原始 HTML：[raw/html/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.html](../../raw/html/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.html)
- 全文文本：[raw/text/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.md](../../raw/text/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)
- 来源 URL：https://arxiv.org/abs/2602.12125
- 作者：Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin
- 机构：Renmin University of China; Tencent

## 摘要

这篇论文把 `OPD`（On-Policy Distillation）放进 LLM 后训练与 reasoning RL 的交界处讨论：学生模型先从自己的当前策略采样轨迹，再在这些 student-generated trajectories 上对齐 teacher 的 logit 分布。作者认为，这种做法同时保留了 on-policy 训练与 token 级密集监督，因此不同于用 teacher-generated trajectories 做 `SFT` 的 off-policy distillation，也不同于只依赖最终正确性或 outcome reward 的常规 RL。

论文的核心贡献不是首次提出 OPD，而是把 OPD 重新解释为一种 **dense KL-constrained RL** 的特殊情形：OPD 中的 teacher / reference log-probability ratio 可以被理解成 token 级隐式 reward，且 reward 项与 KL 正则项在标准 OPD 中固定为同等权重。在此基础上，作者提出 `G-OPD`，通过引入可调 reward scaling factor 与更灵活的 reference model，把标准 OPD 扩展为更一般的后训练目标。

实验主要覆盖数学推理与代码生成。论文提出的 `ExOPD` 使用大于 1 的 reward scaling factor 做 reward extrapolation；在同尺寸多 teacher 合并和 strong-to-weak distillation 场景中，作者报告其相较标准 OPD 有更高表现，并且在某些设置下能让统一 student 超过多个 domain teacher。但论文也承认，reward correction 需要访问 teacher 的 pre-RL base model，会增加计算成本；过强 extrapolation 也可能带来隐式 reward hacking、响应长度膨胀和训练不稳定。

## 关键事实

- `OPD` 的基本形式是：学生模型使用当前策略生成轨迹，并在这些轨迹上最小化 student 与 teacher 分布之间的 reverse KL。
- 作者把 `OPD` 与 `RL` 联系起来：在近似梯度视角下，teacher 与 student 的 token-level log-probability 差异可以被视为 token 级 advantage，从而提供比 outcome reward 更密集的 credit assignment。
- 标准 `OPD` 中 reward 项与 KL 正则项的相对权重固定；`G-OPD` 通过 reward scaling factor 解除这一固定比例。
- 当 scaling factor 大于 1 时，论文称之为 `ExOPD` / reward extrapolation，主张它能在部分数学与代码任务上超过标准 OPD。
- 在 strong-to-weak distillation 中，若能使用 teacher 的 pre-RL base model 作为 reference model，作者认为 reward signal 更准确，但这依赖额外模型访问并提高计算开销。

## 争议与不确定点

- `ExOPD` 超过 teacher 的结论主要来自论文内部实验，仍需要更多独立复现和跨模型族验证。
- reward extrapolation 会放大 implicit reward，因此收益与风险并存；论文自身也指出过大 scaling factor 可能导致不稳定、reward hacking 倾向与长度偏置。
- OPD 在知识库中应被视为 reasoning RL / post-training 的相邻分支，而不是简单归入 `DPO / ORPO / KTO` 这类成对或单点偏好优化。
- strong-to-weak 场景里的 reward correction 对 teacher pre-RL model 有依赖；在闭源或只发布 chat model 的现实模型家族中，这一条件未必成立。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 概念：[OPD](../concepts/OPD.md)
- 概念：[GRPO](../concepts/GRPO.md)
- 概念：[DAPO](../concepts/DAPO.md)
- 概念：[DPO](../concepts/DPO.md)
