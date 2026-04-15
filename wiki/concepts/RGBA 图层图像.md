# RGBA 图层图像

## 简介

`RGBA` 图层图像指的是把透明度与图层结构一起视为一等表示对象的图像建模路线。与传统单张 `RGB` raster image 相比，这条路线更关心图像如何被分解、重建、合成和编辑，而不仅是最终视觉结果。

## 关键属性

- 类型：图像表示与生成工作流概念
- 代表来源：
  - [Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](../../wiki/summaries/Wang%20et%20al.%20-%202025%20-%20AlphaVAE%20Unified%20End-to-End%20RGBA%20Image%20Reconstruction%20and%20Generation%20with%20Alpha-Aware%20Representation%20Learning.md)
  - [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)
- 当前角色：连接透明图像表征、图层分解与可编辑生成的中间概念页

## 相关主张

- `RGBA` 不应只被理解为导出格式；它正在成为可编辑图像生成与 layered workflow 的潜表示对象。
- 这条路线至少包含两个层次：`AlphaVAE` 代表的透明图像表征与 benchmark 底座，以及 `Qwen-Image-Layered` 代表的多图层分解与编辑接口。
- 从知识组织角度看，`RGBA` 图层图像路线是当前 `扩散模型与文生图` topic 中新出现的一条支线，它与传统 text-to-image 相邻，但问题设定不同。

## 来源支持

- [Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](../../wiki/summaries/Wang%20et%20al.%20-%202025%20-%20AlphaVAE%20Unified%20End-to-End%20RGBA%20Image%20Reconstruction%20and%20Generation%20with%20Alpha-Aware%20Representation%20Learning.md)
- [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)

## 关联页面

- [AlphaVAE](./AlphaVAE.md)
- [Qwen-Image-Layered](./Qwen-Image-Layered.md)
- [Qwen-Image](./Qwen-Image.md)
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
