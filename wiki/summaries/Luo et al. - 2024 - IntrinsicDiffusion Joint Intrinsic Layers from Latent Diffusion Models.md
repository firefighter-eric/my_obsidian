# Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models

## 来源信息

- 类型：论文 / Adobe Research
- 来源链接：https://research.adobe.com/publication/intrinsicdiffusion-joint-intrinsic-layers-from-latent-diffusion-models/
- 原始文件：../../raw/html/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.html
- 全文文本：../../raw/text/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.md
- 作者：Jundan Luo, Duygu Ceylan, Jae Shin Yoon, Nanxuan Zhao, Julien Philip, Anna Frühstück, Wenbin Li, Christian Richardt, Tuanfeng Y. Wang
- 年份：2024
- 状态：已整理

## 摘要

`IntrinsicDiffusion` 把“图像分层”推进到另一种更偏物理与编辑语义的层面：不是分前景/背景 `RGBA` 图层，而是联合预测 albedo、illumination、surface geometry 等 intrinsic modalities。论文的核心主张是，大规模 text-to-image foundation model 已隐式学到足够强的 intrinsic priors，可以被重新用作联合 intrinsic decomposition 的底座。

## 关键事实

- 论文把 intrinsic image decomposition 定义为对图像内在属性的联合推断，而不是传统意义上的对象层或 PSD 图层恢复。
- 方法构建在预训练 foundation image generation model 之上，加入新的 conditioning mechanism，使模型可从输入图像联合预测多个 intrinsic modalities。
- 作者强调“joint / collaborative prediction”而不是逐模态独立预测，认为多模态协同会提升整体分解质量。
- 训练设计支持混合使用“只标部分 intrinsic modalities”的数据集，这意味着它试图绕开 intrinsic 任务长期存在的标注稀缺与数据异构问题。
- 论文在 Adobe 页面上明确宣称达到 intrinsic image decomposition 的 `state-of-the-art`，并展示 relighting、retexturing 等下游编辑用途。
- 这条路线说明 Adobe 对 layered 的理解不只等于“可编辑前景层”，还包括把图像拆成材质、光照、几何等可重组的内在层。

## 争议与不确定点

- 当前仓库可用来源主要是 Adobe publication page 摘要级信息，尚不足以重建其完整架构细节、基准名称与所有实验设置。
- intrinsic layers 与 `RGBA` / PSD-style layers 不是同一种分层对象；若把两者混写，会误把物理分解问题当成设计软件图层问题。
- 虽然它显然服务编辑与重光照，但它是否应被视为“图像分层 layered”主线中的核心节点，仍取决于后续是否出现更多把 intrinsic decomposition 与可编辑工作流直接打通的来源。

## 关联页面

- 主题：[图像分层 layered](../../wiki/topics/%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)
- 主题：[传统 CV](../../wiki/topics/%E4%BC%A0%E7%BB%9F%20CV.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
