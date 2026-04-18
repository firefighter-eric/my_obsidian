# 注意力机制 Attention

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论的是 `attention` 作为现代 Transformer 与大模型核心算子的内部谱系，而不是把所有带“attention”命名的方法杂糅在一起。它要回答的不是“attention 是什么”这种概念页问题，而是：**标准 attention 为什么成为统一接口；后续方法到底在优化哪一种瓶颈；不同 attention 变体是否真的在解决同一个问题。**

因此，本页最重要的边界意识是：`efficient attention` 并不是单一问题域。长序列 encoder、长文档建模、自回归解码、`KV cache` 压缩、GPU kernel IO 优化，都会被论文写成“高效 attention”，但它们优化的对象并不相同。若不先把这些瓶颈拆开，后续所有方法分层都会变成表面分类。

从当前 evidence base 看，本页可以稳定成立的核心判断是：**attention 的研究主线不是从一种标准形式演化到另一种更先进的标准形式，而是围绕不同瓶颈分化出了几类彼此只部分可比的路线。** `Vaswani et al.` 提供统一基线；`Longformer / BigBird / Reformer` 改连接图；`Linformer / Performer / Nyströmformer` 近似完整矩阵；`MQA / GQA / MLA` 压缩解码态；`FlashAttention` 重写执行方式。

## 核心问题

- **标准 `scaled dot-product attention` 为什么能成为统一信息路由接口**，并长期作为各种近似与优化的参照物。
- **所谓 `O(n^2)` 瓶颈到底指什么**：是训练时的全矩阵显存和计算，还是推理时的 `KV cache` 带宽，还是硬件 IO 常数项。
- **后续 attention 变体到底在改变什么**：连接图、矩阵近似、状态缓存结构，还是 kernel 实现。
- **哪些方法是在近似标准 attention，哪些方法是在改变应用场景和系统接口**。

## 主线脉络 / 方法分层

从当前 summary 组合出发，本主题最适合按**被优化的瓶颈对象**来分层，而不是按论文中常见的“稀疏 / 线性 / 高效”标签直接平铺。

- **标准语义基线层**：`Attention Is All You Need` 提供了 `softmax(QK^T / sqrt(d_k))V`、`multi-head attention` 以及 encoder 双向 / decoder 因果两种基本用法。这一层的重要性在于，它建立了一个后来几乎所有工作都要回应的共同参照系。也就是说，很多后续论文并不是在重新定义注意力，而是在回答“怎样在保留这套语义的同时减少代价”。
- **长序列连接图稀疏化层**：这一层接受的前提是，并非所有 token 对都必须显式交互。`Longformer` 用**局部窗口 + 少量全局 token**处理长文档；`BigBird` 用 **global + local + random** 的混合稀疏模式试图兼顾长程信息流与理论表达性质；`Reformer` 则用 `LSH attention` 按内容相似性分桶，而不是按固定相对位置建图。三者共同代表的是**改连接图**的路线，但它们内部偏置不同：`Longformer` 偏文档任务先验，`BigBird` 偏混合稀疏与理论性质，`Reformer` 偏近邻检索式稀疏。
- **近似 full attention 矩阵层**：这条线不预设显式稀疏结构，而是把完整 attention matrix 本身视为可近似对象。`Linformer` 假设 attention 映射存在低秩结构，先压缩 `K/V`；`Performer` 用 `FAVOR+` 把 softmax kernel 线性化；`Nyströmformer` 用 landmark / Nyström 方法重建全矩阵。它们共同代表的是**保留全局交互意图，但放弃精确矩阵计算**。与前一层相比，这里的核心不是“谁跟谁连接”，而是“完整交互能否被低成本重写”。
- **解码态与 `KV cache` 压缩层**：`MQA`、`GQA`、`MLA` 虽然也经常被归入 efficient attention，但它们面对的并不是训练时 `n x n` attention matrix，而是**自回归推理不断增长的状态缓存和带宽压力**。`MQA` 通过共享单组 `K/V` 压缩缓存；`GQA` 在 `MHA` 和 `MQA` 之间做连续折中；`MLA` 进一步引入 latent 压缩来减少推理态负担。这一层与长序列 encoder 优化只能在非常宽口径上同属“高效 attention”，但不能视为同一问题的不同答案。
- **实现级与系统级优化层**：`FlashAttention` 代表的是另一种完全不同的思路。它不近似 attention，不改连接图，而是把瓶颈重述为**GPU 高带宽显存 IO**，在保持 exact attention 语义的前提下通过 tile 化、kernel fusion 和重算策略减少内存读写。这说明 efficient attention 还存在第三种逻辑：不改数学对象，只改执行路径。

据此可以得到一个更清晰的综合判断：**attention 世界里至少混杂了四类不同层次的创新：标准语义定义、连接图偏置、矩阵近似、推理态压缩、系统实现优化。** 把这些层级混在一起做“谁更先进”的比较，往往会掩盖它们各自的成立条件。

## 关键争论与分歧

- **“高效 attention”是不是单一问题**：不是，而且这正是本页需要反复强调的地方。若问题是长文档 encoder，`Longformer`、`BigBird`、`Reformer` 更可比；若问题是自回归推理，`MQA / GQA / MLA` 更 relevant；若问题是 exact attention 的速度和显存，`FlashAttention` 更 relevant。很多表面上的路线之争，其实只是问题设定不同。
- **稀疏 attention 与线性 / 低秩 attention 哪个更接近标准 attention**：现有证据不支持一个统一答案。稀疏路线保留精确局部交互但显式删边；线性 / 低秩路线保留全局交互意图但接受矩阵近似。哪一个“更接近”标准 attention，取决于你把语义保真度理解为连接图保真还是矩阵值保真。
- **`MQA / GQA / MLA` 是否应与 `Linformer / Performer / BigBird` 直接并列**：只能在极宽的“attention 变体”口径下并列，不能在“同一瓶颈的不同解法”意义上并列。前者主要优化 decoder inference，后者主要优化长序列 attention 计算。若不先说明这一点，topic 会错误暗示它们可直接横评。
- **`FlashAttention` 是否属于新的 attention 模型**：从当前来源看，不应这样表述。它改变的是执行方式和 IO 代价，而不是注意力语义本身。更准确的说法是，它把标准 attention 从“算术复杂度问题”重新表述成“内存访问问题”。
- **是否存在统一最优的 attention 路线**：现有证据不支持。不同方法在近似误差、硬件依赖、训练稳定性、部署场景与任务类型上都有明显边界。当前更稳妥的结论是：attention 研究已经从“寻找唯一最好形式”转向“为不同瓶颈选择不同结构和实现偏置”。

## 证据基础

- [Vaswani et al. - 2017 - Attention is all you need](../../wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)
- [Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer](../../wiki/summaries/Kitaev,%20Kaiser,%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md)
- [Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer](../../wiki/summaries/Beltagy,%20Peters,%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md)
- [Zaheer et al. - 2020 - Big bird Transformers for longer sequences](../../wiki/summaries/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md)
- [Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity](../../wiki/summaries/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md)
- [Choromanski et al. - 2021 - Rethinking Attention with Performers](../../wiki/summaries/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md)
- [Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention](../../wiki/summaries/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md)
- [Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need](../../wiki/summaries/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md)
- [Ainslie et al. - 2023 - GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](../../wiki/summaries/Ainslie%20et%20al.%20-%202023%20-%20GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](../../wiki/summaries/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)

## 代表页面

- [Transformer](../concepts/Transformer.md)
- [FlashAttention](../concepts/FlashAttention.md)
- [Grouped-Query Attention](../concepts/Grouped-Query%20Attention.md)
- [LLM 预训练](./LLM%20预训练.md)

## 未解决问题

- 当前页已经能较清楚地区分**长序列稀疏化、矩阵近似、推理态压缩、IO-aware 实现**四类主线，但仍缺更系统的 survey summary，因此“十多种 attention 形式的全景地图”还没有完全稳定下来。
- `cross-attention`、扩散模型中的 cross-attention、多模态 routing、检索增强中的 chunk routing 目前尚未被正式并入本页；在现有证据下，贸然并入只会让主题边界重新变宽。
- 位置建模如 `RoPE / ALiBi / YaRN` 会显著影响 attention 行为，但它们更准确地属于相邻层而非 attention 语义本体；若后续证据增长，可能需要 comparison 页专门处理“attention 与位置编码的接口”。
- `MLA` 目前仍主要通过 `DeepSeek-V3` 技术报告间接支撑；若要把 latent attention 写成更稳定的概念或分支，仍需补更直接的一手 summary。

## 关联页面

- [Transformer](../concepts/Transformer.md)
- [FlashAttention](../concepts/FlashAttention.md)
- [Grouped-Query Attention](../concepts/Grouped-Query%20Attention.md)
- [LLM 预训练](./LLM%20预训练.md)
