# 图像分层 layered

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论“图像分层 `layered`”作为**一条独立的问题域**，而不是把它当作图像生成、图像编辑、分割或深度估计的附属技巧。这里真正要处理的问题是：能否把原本纠缠在单张 `RGB` raster image 里的内容，改写成若干**可独立操作、可局部修改、可重新组合、可回放编辑过程**的结构化层。

之所以需要把它单独成题，是因为当前来源已经显示出几种**彼此相关、但不能混为一谈**的“层”概念。`Qwen-Image-Layered` 所说的层，是多张**语义解耦的 `RGBA` 图层**；Adobe `LayerDecomp` 所说的层，更接近**带视觉效果的前景/背景编辑接口**；`IntrinsicDiffusion` 所说的层，是 `albedo`、`illumination`、`geometry` 这类 **intrinsic modalities**；`SLEDGE` 所说的层，是**逐步设计过程中的原子更新**；`Illustrator's Depth` 所说的层，则是**矢量插画与平面图像中的全局 layer ordering**。它们共同指向“**为编辑拆出结构**”，但拆出的对象并不相同。

因此，本页的边界不是“所有带 layer 字样的论文”，而是那些把图像从最终像素结果重写为某种**可编辑中间表示**的工作。与 [扩散模型与文生图](./扩散模型与文生图.md) 相比，本页**不以生成质量为中心**；与 [传统 CV](./传统%20CV.md) 相比，本页**不以识别精度为中心**；与单纯的 Photoshop 工作流描述相比，本页也不是软件教程，而是研究上如何把“**层化表示**”变成模型能力的问题综述。

## 核心问题

- 为什么单张 `RGB` 图像会成为一致性编辑、局部修改和工作流复用的瓶颈。
- layered 表示究竟是导出格式、后处理接口，还是新的基础图像表示。
- 不同 layered 路线解决的是不是同一个问题，还是只是共享“可编辑”这一目标。
- 哪些 layered 结论已经相对稳定，哪些仍强依赖特定数据分布、标注方式或创作场景。

更具体地说，当前领域至少卡在四个难点上。第一，**像素层面的纠缠表示**使得局部编辑常常破坏未编辑区域，这解释了为什么很多工作要把“编辑一致性问题”前移到表示层。第二，不同工作对“层”的**语义定义并不一致**，导致表面上都叫 layered，实际输出目标和评价口径差异很大。第三，**高质量层化监督本身稀缺**，不论是 `PSD`、`SVG`、真实视觉效果对、还是 intrinsic 标注，都具有明显的分布偏置。第四，即便模型能产出某种层化结果，这种结果是否真的适合后续编辑工作流，仍不能只用传统重建指标判断，而必须看**对象移动、重组、重光照、矢量化或多步设计更新**等下游用途。

## 主线脉络 / 方法分层

从现有 summary 看，图像分层**并不是一条单线演进**，而是围绕“什么东西应该被显式拆出来”形成的几条并行支线。比较稳妥的分法**不是按论文年份，也不是按作者机构**，而是按被显式建模的“**层对象**”来分。

第一层是 **`RGBA` 表征底座层**。`AlphaVAE` 代表的是最底部的基础设施问题：如果透明图像与 `alpha` 通道本身都没有稳定潜空间，那么更高层的 layered generation 与 decomposition 很难成立。它的重要性不在于直接解决复杂图层分解，而在于把 `RGBA` 从边缘格式提升为**可训练、可评测、可继续叠加建模**的表示对象。

第二层是**多层 `RGBA` 分解层**。`Qwen-Image-Layered` 是当前仓库里最接近“**原生多层图像表示**”的节点。它不是只从文本生成一张可叠加图，而是把单张 `RGB` 图像直接分解为多个**语义解耦的 `RGBA` 层**，并明确强调**可变图层数、共享 `RGB/RGBA` 潜空间、以及从预训练生成模型逐步迁移成 layered decomposer** 的训练链条。就问题设定而言，这条线最接近“**把 Photoshop 式多层工作流原生化到模型内部**”。

第三层是**两层编辑接口层**。Adobe 的 `LayerDecomp` 与 `Qwen-Image-Layered` 有相似目标，即都想通过分层提升编辑一致性，但它解决的是**更窄也更实用**的问题：把输入图像分成一个**干净背景**和一个带阴影、反射等视觉效果的**透明前景**。它**不试图恢复完整多层语义栈**，而是直接面向 `object removal`、`spatial editing` 和无缝重组场景。因此，它更像生产编辑流程中的“**高质量前景/背景分离接口**”，而不是统一 layered world model。

第四层是 **intrinsic 分解层**。`IntrinsicDiffusion` 说明 layered 的对象不必是“可见对象层”，也可以是**材质、光照、几何等图像内在因子**。这条线与 `RGBA` layered 的相似点，在于它同样试图把编辑所需的控制变量从单张彩色图里拆出来；不同点在于它的目标**不是 PSD 式层合成**，而是 `relighting`、`retexturing` 这类对 intrinsic properties 敏感的编辑任务。把它纳入本 topic 是合理的，但前提是要明确：这里的层是“**内在属性层**”，不是“对象透明层”。

第五层是**工作流分步层**。Adobe 的 `SLEDGE` 进一步把 layered 从“单张图像表示”扩展到“**设计过程表示**”。在这条线里，层不是一次性恢复出的静态结构，而是当前 `canvas` 与下一步 `canvas` 之间那次**原子修改**本身。它关注的是如何在逐步指令下，**保留旧画布、只叠加本次更新，并同时保留可编辑 metadata**。这使 layered 在这里更像**版本化设计系统**，而不是传统意义上的像素分解。

第六层是**全局排序层**。`Illustrator's Depth` 又把 layered 引向矢量图与插画编辑场景。它提出的不是透明前景层，也不是 intrinsic maps，而是每个像素对应的**全局 `layer index`**。这个定义与普通 `monocular depth`、`panoptic segmentation` 都不同，因为它追求的是“**适合编辑与重排的排序结构**”，而非物理几何深度或纯语义区域划分。从知识组织上看，这一节点说明 layered 主题**不应被狭义地限制在 `RGBA` raster image**。

从这几条支线综合看，当前更稳妥的结论是：图像分层**并不是单一 benchmark 驱动的统一赛道**，而是一组围绕“**可编辑中间表示**”展开的子任务族。它们共享的问题意识是相通的，但在**建模对象、监督来源、评测方式和下游用途**上都存在显著差异。

## 关键争论与分歧

- **layered 是否是一条**统一主线**，而不是若干互不相干的小任务**：现有来源更支持把它视为共同问题域，因为这些工作都在试图把**编辑能力前移到表示层**；但现有证据并不支持把它们直接放进同一个单一 benchmark 下比较优劣。
- **`RGBA` layered 是否就是图像分层的主体**：从 `AlphaVAE` 与 `Qwen-Image-Layered` 看，`RGBA` 的确是当前**最贴近设计软件工作流**的一条主线；但 `IntrinsicDiffusion` 与 `Illustrator's Depth` 说明，如果把 layered 限定为透明对象图层，会遗漏**内在属性分解与全局层序恢复**这两类重要问题。
- **Qwen 与 Adobe 是否在做同一件事**：**部分重叠，但不能简单并列**。Qwen 当前的 layered 节点更像“**统一多层 `RGBA` 表示模型**”；Adobe 当前多篇论文更像围绕创作软件与设计流程，把 layered 拆成**视觉效果保留、intrinsic decomposition、逐步设计更新、矢量层序恢复**等多个问题接口。
- **layered 会不会成为**主流图像接口****：现有来源支持它的重要性正在上升，尤其在**海报、广告、电商、设计资产和矢量化场景**中更明显；但现阶段还不足以证明所有图像生成与编辑系统都会统一迁移到 layered representation。
- **真实开放世界照片能否**稳定层化****：这是当前**最大的不确定点**。许多方法依赖 `PSD`、`SVG`、`camera-captured pairs`、`design templates`、`object masks` 或受控 intrinsic 标注，这意味着它们的成功条件往往是“**工作流内可用**”，而不是“开放世界普遍成立”。

这些争论背后的结构性原因也比较清楚。其一，大家对“层”的**定义不同**，导致表面同类、实际异题。其二，许多方法的监督来源天然就是**工作流数据**，而不是自然图像全分布，因此它们很容易在目标场景中有效、在开放场景中边界不明。其三，layered 的真正价值常常要通过**下游编辑任务**体现，而不是靠单一重建或生成分数体现，这也使不同工作之间更难在同一口径下比较。

## 证据基础

- [Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](../../wiki/summaries/Wang%20et%20al.%20-%202025%20-%20AlphaVAE%20Unified%20End-to-End%20RGBA%20Image%20Reconstruction%20and%20Generation%20with%20Alpha-Aware%20Representation%20Learning.md)
- [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)
- [Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects](../../wiki/summaries/Yang%20et%20al.%20-%202025%20-%20Generative%20Image%20Layer%20Decomposition%20with%20Visual%20Effects.md)
- [Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models](../../wiki/summaries/Luo%20et%20al.%20-%202024%20-%20IntrinsicDiffusion%20Joint%20Intrinsic%20Layers%20from%20Latent%20Diffusion%20Models.md)
- [Khan et al. - 2026 - Step-by-step Layered Design Generation](../../wiki/summaries/Khan%20et%20al.%20-%202026%20-%20Step-by-step%20Layered%20Design%20Generation.md)
- [Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition](../../wiki/summaries/Maruani%20et%20al.%20-%202026%20-%20Illustrator's%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md)

## 代表页面

- [RGBA 图层图像](../concepts/RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- [Qwen-Image-Layered](../concepts/Qwen-Image-Layered.md)
- [AlphaVAE](../concepts/AlphaVAE.md)
- [扩散模型与文生图](./%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Qwen 系列](./Qwen%20%E7%B3%BB%E5%88%97.md)

## 未解决问题

- 当前知识库仍缺 `Text2Layer`、`LayerDiffuse`、`DreamLayer`、`ART`、`PrismLayers` 等公开路线的系统补齐，因此 layered 的前史到当前节点还**没有形成完整演化链**。
- `IntrinsicDiffusion` 在仓库中仍主要由 Adobe `publication-page` 级来源支撑，**架构、数据混合策略与实验设置的可追溯度**仍弱于其他几篇 `arXiv` 论文。
- `LayerDecomp` 与 `Qwen-Image-Layered` 的任务边界虽已较清楚，但两者在真实生产工作流中是**互补接口、上下游关系，还是竞争方案**，当前证据仍不足。
- `SLEDGE` 与 `Illustrator's Depth` 都强烈依赖**设计/矢量分布**，这意味着 layered 在图形设计中很可能**先成熟、在自然照片中后成熟**；但这一判断仍需要更多来源验证。
- 当前 topic 已能区分 layered 子类，但还缺 `comparison / timeline` 页来承接更细的横向对照，例如“**对象透明层 vs intrinsic layers**”“**多层分解 vs 两层接口**”“**静态层表示 vs 过程层表示**”。

## 关联页面

- [扩散模型与文生图](./%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Qwen 系列](./Qwen%20%E7%B3%BB%E5%88%97.md)
- [传统 CV](./%E4%BC%A0%E7%BB%9F%20CV.md)
- [Slide 理解与生成](./Slide%20%E7%90%86%E8%A7%A3%E4%B8%8E%E7%94%9F%E6%88%90.md)
- [RGBA 图层图像](../concepts/RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- [Qwen-Image-Layered](../concepts/Qwen-Image-Layered.md)
- [AlphaVAE](../concepts/AlphaVAE.md)
