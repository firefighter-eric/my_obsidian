# Scaling 与 compute-optimal training

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页聚焦 LLM 训练中的规模化逻辑，特别是“能力为什么随规模增长”“以及固定计算预算下模型参数量与训练 token 数如何共同配置”这两个紧密相关但不能混为一谈的问题。它是 `LLM预训练` 中最偏训练规律的一页，也是从 GPT-3 叙事过渡到 Chinchilla 修正的关键中间层。

## 核心问题

- 规模扩大为何能提升 few-shot 与跨任务泛化能力。
- 训练计算预算固定时，参数量与 token 数的平衡应如何理解。
- dense scaling 成功与 under-trained 修正之间是否存在矛盾。
- 训练最优配置是否能直接转化为部署与推理阶段的最优选择。

## 主线脉络 / 方法分层

- 规模化有效性：`Brown et al. 2020` 说明在自回归语言模型框架内，扩大规模本身就会显著改善 few-shot 能力，并使 prompt 成为任务接口。
- 规模化的系统扩展：`PaLM` 进一步展示 dense Transformer 在更大训练系统下仍可持续获得多任务收益，并把多语言、代码与推理能力纳入同一规模化语境。
- compute-optimal 修正：`Hoffmann et al. 2022` 通过大规模实验指出许多模型并不是“太小”，而是“训练 token 不够”；因此 compute-optimal 的讨论本质上是预算分配问题，而不只是模型大小问题。
- 路线结论：从当前证据出发，更准确的表述不是“GPT-3 被 Chinchilla 否定”，而是“早期 dense scaling 证明规模有用，随后 Chinchilla 修正了如何更合理地用计算预算扩规模”。

## 关键争论与分歧

- 是否应继续使用一般性的 scaling law 叙述：当前知识库更适合把重点放在 compute-optimal 训练，而不是泛泛而谈所有 scaling law。
- 模型更大还是数据更多更关键：`Hoffmann 2022` 的结论不是否定大模型，而是反对只增参数不增数据的做法。
- 训练最优是否等于产品最优：训练 FLOPs 最优与推理延迟、内存占用、部署成本之间并不天然一致，这一层在当前材料中仍未展开。
- sparse 路线是否改写 compute-optimal 讨论：MoE 可能改变“总参数规模”与“单 token 计算量”的关系，但当前 topic 还缺少直接比较 sparse 与 dense 的 summary 支撑。

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../raw/summary/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](../../raw/summary/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../raw/summary/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)

## 代表页面

- [GPT-3](../concepts/GPT-3.md)
- [PaLM](../concepts/PaLM.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [LLM 预训练](LLM%20预训练.md)

## 未解决问题

- 当前页面尚未接入更早的 Kaplan scaling law 原文，因此对 scaling law 的表述仍以 `Brown 2020` 和 `Hoffmann 2022` 之间的主线为主。
- 训练数据质量、重复率与 compute-optimal 配置之间的关系尚未形成独立证据页。
- sparse/MoE 路线如何重写 compute-optimal 讨论，需要更多直接来源而不是只靠 DeepSeek-V3 间接进入。

## 关联页面

- [LLM 基础脉络](./LLM%20基础脉络.md)
- [LLM 预训练](LLM%20预训练.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [MoE](../concepts/MoE.md)
