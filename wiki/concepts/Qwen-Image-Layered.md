# Qwen-Image-Layered

## 简介

Qwen-Image-Layered 是 Qwen 图像生成支线向图层化可编辑表示延伸的节点。在当前知识库里，它代表的不是“又一个文生图模型”，而是把单张 RGB 图像分解成可独立操作的多层 `RGBA` 表示，从而把编辑一致性问题前移到表示层解决。

## 关键属性

- 类型：图层分解与可编辑图像生成模型
- 代表来源：
  - [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)
- 当前角色：Qwen 图像生成支线中的 layered decomposition 节点

## 相关主张

- 它的重点不是继续提升单张 RGB 出图质量，而是把图像编辑所需的图层结构直接建模出来。
- 它说明 Qwen 图像生成路线已经从“文本渲染与编辑能力”进一步扩展到“表示级可编辑性”。
- 在方法上，`Qwen-Image-Layered` 与 `AlphaVAE` 共享一个关键判断：`RGBA` 不应只是附属通道，而应作为独立潜表示对象来训练。

## 来源支持

- [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)

## 关联页面

- [Qwen-Image](./Qwen-Image.md)
- [AlphaVAE](./AlphaVAE.md)
- [RGBA 图层图像](./RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Qwen 系列](../topics/Qwen%20%E7%B3%BB%E5%88%97.md)
