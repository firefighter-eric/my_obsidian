# DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence

## 来源信息

- 类型：技术报告 / 模型卡发布资料
- 原始文件：../../raw/pdf/DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.pdf
- 发布页快照：../../raw/html/DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.html
- 全文文本：../../raw/text/DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.md
- 作者：DeepSeek AI
- 年份：2026
- 状态：已基于官方技术报告与模型卡整理

## 摘要

`DeepSeek-V4` 是 DeepSeek 在 `DeepSeek-V3 / V3.2` 之后推出的预览版 MoE 语言模型系列，包含 `DeepSeek-V4-Pro` 与 `DeepSeek-V4-Flash` 两条规模路线。报告的核心不是单纯扩大参数，而是把 **百万 token 上下文、长程 agent 工作流和稀疏模型效率** 绑定在同一个架构目标里：一方面继续使用 DeepSeekMoE 扩展总容量，另一方面用 `CSA / HCA` 混合 attention 压低长上下文推理中的 FLOPs 与 `KV cache` 成本。

在当前知识库中，`DeepSeek-V4` 应被视为 `DeepSeek-V3` 后的效率架构续章：`DeepSeek-V3` 主要支撑 MoE 与 `MLA` 的高效基座叙事，`DeepSeek-V4` 则把问题推进到更长上下文、更重 agent 轨迹和更激进的 KV 压缩。其 post-training、reasoning mode 与 agent benchmark 很重要，但在主题归类上应分别连接 `LLM 预训练`、`LLM RL` 与后续 agent topic，而不能全部塞进预训练页。

## 关键事实

- `DeepSeek-V4` 预览系列包含 `DeepSeek-V4-Pro` 与 `DeepSeek-V4-Flash`：前者约 `1.6T` 总参数、`49B` 激活参数，后者约 `284B` 总参数、`13B` 激活参数。
- 两个模型都以 `1M` token 上下文为显性目标，报告称在百万上下文场景下，`DeepSeek-V4-Pro` 相比 `DeepSeek-V3.2` 仅需要约 `27%` 的单 token 推理 FLOPs 与 `10%` 的 `KV cache`。
- 架构升级包括 `Compressed Sparse Attention (CSA)`、`Heavily Compressed Attention (HCA)` 组成的混合 attention、`Manifold-Constrained Hyper-Connections (mHC)`，以及用于训练稳定性和收敛效率的 `Muon` optimizer。
- `CSA` 先把若干 token 的 `KV cache` 压缩成 compressed KV entries，再通过 `DeepSeek Sparse Attention (DSA)` 从压缩块中选择 top-k 参与核心 attention；它负责在压缩后保留相关性选择能力。
- `HCA` 使用更大的压缩率把更长跨度的 `KV cache` 合并为单个 compressed KV entry，并与 sliding window branch 配合保留近邻依赖；它负责把百万上下文下的缓存成本进一步压低。
- `mHC` 通过把 residual mapping 约束到 doubly stochastic matrices 所在的 Birkhoff polytope，改善深层 Transformer blocks 间的信号传播稳定性；报告同时说明其需要 fused kernels、recomputation 和 pipeline overlap 来控制额外开销。
- `Muon` 被用于多数模块，`AdamW` 仍用于 embedding、prediction head、`mHC` 的静态 bias / gating factors 和 `RMSNorm` 权重；报告称该优化器有助于更快收敛与训练稳定性，并为其设计了与 `ZeRO` 和 MoE 参数更新兼容的实现。
- 报告称 `DeepSeek-V4-Flash` 使用约 `32T` token 预训练，`DeepSeek-V4-Pro` 使用约 `33T` token 预训练，之后再经过 SFT、GRPO 等后训练管线。
- `DeepSeek-V4` 的发布不是纯 base model 事件：模型卡同时强调 `Non-think / Think High / Think Max` 等推理模式、agent 能力与工具调用格式，这说明该系列从发布层面就面向长程任务和 agent 使用。
- `DeepSeek-V4-Flash` 与 `DeepSeek-V4-Pro` 都有 base 与 instruct 权重入口，发布页标注 MIT license，并通过 Hugging Face collection 分发。

## 争议与不确定点

- 这是预览版本，模型卡与技术报告中的 benchmark、参数口径和能力主张仍需要后续社区复现与第三方评测确认。
- 报告同时覆盖预训练架构、后训练、reasoning mode、agent 训练基础设施与部署细节；不同主题页引用时必须拆分使用，不能把 agent benchmark 直接写成预训练规律。
- `CSA / HCA / mHC / Muon` 是否会成为 DeepSeek 之外的通用架构范式，目前还缺少更多独立实现与消融复现。
- `1M` token 上下文是能力上限与工程目标，但实际可用性依赖推理框架、显存、KV 存储策略、工具轨迹长度和任务类型。

## 关联页面

- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 概念：[DeepSeek-V4](../../wiki/concepts/DeepSeek-V4.md)
- 概念：[DeepSeek-V3](../../wiki/concepts/DeepSeek-V3.md)
- 概念：[MoE](../../wiki/concepts/MoE.md)
- 概念：[Compressed Sparse Attention](../../wiki/concepts/Compressed%20Sparse%20Attention.md)
- 概念：[Heavily Compressed Attention](../../wiki/concepts/Heavily%20Compressed%20Attention.md)
- 概念：[Manifold-Constrained Hyper-Connections](../../wiki/concepts/Manifold-Constrained%20Hyper-Connections.md)
- 概念：[Muon](../../wiki/concepts/Muon.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)
