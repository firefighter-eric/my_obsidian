# AlphaVAE

## 简介

AlphaVAE 是面向透明图像与 `RGBA` 生成的基础表征模型。在当前知识库里，它代表的不是完整图层编辑系统，而是把透明图像的 VAE、潜空间稳定性和评测协议一起补齐的基础设施节点。

## 关键属性

- 类型：RGBA VAE / 透明图像表征模型
- 代表来源：
  - [Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](../../wiki/summaries/Wang%20et%20al.%20-%202025%20-%20AlphaVAE%20Unified%20End-to-End%20RGBA%20Image%20Reconstruction%20and%20Generation%20with%20Alpha-Aware%20Representation%20Learning.md)
- 当前角色：RGBA 潜表示与透明图像评测基座

## 相关主张

- `AlphaVAE` 的核心价值在于把透明图像从 RGB 系统的附属后处理，提升为需要独立 latent 表征的对象。
- 它补的是图像生成链路中的底层缺口，因此其意义更多在于给 layered generation、transparent generation 和后续 latent diffusion 提供更稳的编码基础。
- 与 `Qwen-Image-Layered` 相比，`AlphaVAE` 更偏底层表征与 benchmark，而不是直接解决复杂图层分解工作流。

## 来源支持

- [Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](../../wiki/summaries/Wang%20et%20al.%20-%202025%20-%20AlphaVAE%20Unified%20End-to-End%20RGBA%20Image%20Reconstruction%20and%20Generation%20with%20Alpha-Aware%20Representation%20Learning.md)

## 关联页面

- [RGBA 图层图像](./RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- [Qwen-Image-Layered](./Qwen-Image-Layered.md)
- [Qwen-Image](./Qwen-Image.md)
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
