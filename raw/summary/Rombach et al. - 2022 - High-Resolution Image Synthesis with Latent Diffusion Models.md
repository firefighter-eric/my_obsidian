# Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models

## 来源信息

- 类型：论文 / CVPR 2022
- 来源链接：https://arxiv.org/abs/2112.10752
- 全文文本：../../raw/text/Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models.md
- 作者：Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer
- 年份：2022
- 状态：已整理

## 摘要

这篇论文提出 latent diffusion models（LDM），核心贡献是把扩散过程从像素空间转移到预训练自编码器的潜空间中，从而在显著降低训练与推理成本的同时保持高保真图像生成能力。它为后续 Stable Diffusion 路线提供了直接技术基座。

## 关键事实

- 论文指出传统像素空间 diffusion model 训练和推理都极其昂贵，而 LDM 通过在潜空间建模，试图在复杂度降低与细节保留之间取得更优平衡。
- 论文显式引入 cross-attention，使模型能够接收文本、bounding boxes 等通用条件输入，而不必为每种条件重写生成器。
- 论文声称其 LDM 在 inpainting 上达到新 SOTA，并在 unconditional generation、semantic scene synthesis、super-resolution 等任务上取得强竞争力，同时显著降低算力需求。
- 从知识库视角看，Stable Diffusion 不应被理解为“凭空出现的产品名”，而应被理解为 LDM 研究路线的开放发布与工程化扩张。

## 争议与不确定点

- 本文是 latent diffusion 的研究论文，不等于 Stable Diffusion 全部产品化细节。
- 从 `LDM` 论文到 `Stable Diffusion` 家族之间还包含数据、开放发布、社区生态与后续架构演化，这些需要结合后续来源理解。

## 关联页面

- 概念：[Stable Diffusion](../../wiki/concepts/Stable%20Diffusion.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
