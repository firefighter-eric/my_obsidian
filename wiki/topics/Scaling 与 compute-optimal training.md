# Scaling 与 compute-optimal training

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页聚焦 **LLM 训练中的规模化规律**，尤其是两个紧密相连但不能混写的问题：第一，为什么能力会随着模型、数据与训练系统规模扩大而提升；第二，在 **固定计算预算** 下，参数量与 token 数应如何配置才更接近最优。也就是说，本页讨论的是 **训练规律**，而不是家族盘点，也不是完整的预训练总论。

与 [LLM 预训练](./LLM%20预训练.md) 相比，本页更窄，专门处理“规模为何有效”和“预算如何最优”这两个理论与工程中间层问题。与 `MoE` 或某些具体模型页相比，本页也更抽象；只有当具体模型能为规模化规律提供稳定证据时，才进入讨论。

当前知识库中，这一 topic 的最稳固骨架是：`GPT-3` 证明大规模 dense 自回归预训练会带来明显 few-shot 与跨任务能力；`PaLM` 说明这种收益在更大训练系统中仍持续存在；`Chinchilla` 则把“scaling”从粗糙的“继续变大”修正为 **在既定 FLOPs 下平衡参数量与训练 token 数**。因此，本页真正要解释的不是单篇论文，而是 **dense scaling 与 compute-optimal 修正之间的连续关系**。

## 核心问题

- 为什么扩大规模会提升 few-shot、跨任务泛化与通用语言能力。
- 在固定训练计算预算下，**参数量** 与 **训练 token 数** 的平衡应如何理解。
- dense scaling 的成功与 `Chinchilla` 的 under-trained 修正之间究竟是冲突关系，还是连续纠偏关系。
- 训练阶段的 compute-optimal 结论，是否能够直接外推到部署阶段的成本最优与产品最优。
- sparse/MoE 路线是否会改写当前基于 dense 模型建立的 compute-optimal 讨论。

## 主线脉络 / 方法分层

本页按 **问题演进逻辑** 分层。也就是说，不是“哪篇论文更有名”，而是“它回答了规模化问题中的哪一层”。

- **规模化有效性层**：`Brown et al. 2020` 给出的核心贡献，是让“随着模型规模扩大，few-shot 能力显著增强”成为一个可被广泛接受的事实陈述。这里最重要的不是某个单项 benchmark，而是 prompt 被证明能成为任务接口，从而让大模型具备跨任务迁移的统一表达形式。
- **系统级规模扩展层**：`PaLM` 进一步把这种规模化收益推向更大训练系统，并把多语言、代码与推理能力纳入同一 scaling 语境。它支撑的不是一个全新理论，而是一个关键经验判断：**dense Transformer 的收益并未在 GPT-3 后立即触顶。**
- **compute-optimal 修正层**：`Hoffmann et al. 2022` 是这一 topic 的真正分水岭。它表明许多模型在固定 FLOPs 下并不是“太小”，而是“训练不够久、token 不够多”。因此 compute-optimal training 的核心不是否定大模型，而是把规模化问题重写为 **预算分配问题**：在总算力既定时，应如何在参数量与训练数据上取得更优平衡。
- **路线解释层**：从现有 summary 出发，更准确的结论不是“`GPT-3` 被 `Chinchilla` 推翻”，而是：**早期 dense scaling 证明了规模有效，`Chinchilla` 修正了如何更有效地使用规模。** 这一区分很重要，因为它决定了本页应把 `Chinchilla` 写成“纠偏”，而不是“反例”。
- **外推边界层**：compute-optimal 是训练阶段命题，但产品系统关心的是推理延迟、显存占用、服务成本与吞吐。当前证据提醒我们，**训练最优并不天然等于部署最优**。这也是为什么本页必须单独保留“外推边界”这一层，而不把训练规律直接写成系统结论。
- **sparse/MoE 潜在改写层**：现有 topic 还缺少直接以 MoE 重写 compute-optimal 的强证据，但 `Mixtral`、`DBRX`、`DeepSeek-V3` 所代表的 sparse 路线已足以提出一个结构性问题：当总参数与单 token 激活参数脱钩后，原本建立在 dense 假设上的最优配置结论，在多大程度上仍然成立。当前还不能下定论，但这个问题已经构成该 topic 的自然延伸。

## 关键争论与分歧

- **是否还应使用一般性的 scaling law 叙述**：当前知识库更适合聚焦 `compute-optimal training`，而不是泛泛而谈“模型越大越强”。这一争论真正成立的前提是：必须承认 scaling 既是经验规律，也是预算配置问题，而非单一口号。
- **模型更大还是数据更多更关键**：`Hoffmann 2022` 的结论经常被误读为“数据比参数更重要”。更准确的说法是：**在固定计算预算下，只增参数而不相应增加训练 token 会导致 under-trained。** 因此这不是“参数 vs 数据”的简单二选一，而是联合配置问题。
- **训练最优是否等于产品最优**：当前 summary 并未提供足够证据把 FLOPs 最优直接转写为服务成本最优。只要推理阶段仍受显存、延迟、吞吐与硬件友好性约束，训练结论就不能无条件外推到部署层。
- **dense scaling 是否已被 sparse 路线改写**：目前还不能这么写。dense scaling 仍是理解能力增长与 compute-optimal 讨论的基础语言；MoE 更像是在工程实现上引入新的效率维度。只有在有更多直接对照 summary 后，才能更强地讨论“dense law 是否需要重写”。
- **`Chinchilla` 是否否定了早期大模型叙事**：现有证据不支持这种断裂式写法。更稳妥的说法是，`Chinchilla` 让 scaling 从“继续做大”变成“更精确地配置预算做大”，它修正的是策略，不是抹去 dense scaling 的事实基础。

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](../../wiki/summaries/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)

## 代表页面

- [GPT-3](../concepts/GPT-3.md)
- [PaLM](../concepts/PaLM.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

## 未解决问题

- 当前页面仍缺 `Kaplan scaling law` 原始节点的 summary，因此对“scaling law”本身的叙述尚未形成更长历史链；现阶段主线仍主要依赖 `Brown 2020 -> Hoffmann 2022`。
- **数据质量、重复率、去重策略** 与 compute-optimal 配置之间的关系，还没有被当前 `wiki/summaries/` 充分支撑，因此本页不能把“最优 token 数”写成脱离数据质量的纯公式问题。
- sparse/MoE 路线如何系统性重写 compute-optimal 讨论，目前仍缺直接来源支撑；因此本页只能把它写成 **待验证的结构问题**，而不是既成结论。
- 训练最优与部署最优之间的偏差，目前也缺少独立 comparison 或 summary 支撑；因此本页不能进一步外推出不同产品形态的最优模型规模策略。

## 关联页面

- [LLM 预训练](../topics/LLM%20预训练.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [MoE](../concepts/MoE.md)
