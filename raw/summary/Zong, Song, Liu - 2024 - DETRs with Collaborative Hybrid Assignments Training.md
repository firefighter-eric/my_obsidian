# Zong, Song, Liu - 2024 - DETRs with Collaborative Hybrid Assignments Training

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Zong, Song, Liu - 2024 - DETRs with Collaborative Hybrid Assignments Training.pdf
- 全文文本：../../raw/text/Zong, Song, Liu - 2024 - DETRs with Collaborative Hybrid Assignments Training.md
- 作者：Zong, Song, Liu
- 年份：2024
- 状态：已抽取全文，待精读

## 自动抽取摘要

In this paper, we provide the observation that too few queries assigned as positive samples in DETR with one- to-one set matching leads to sparse supervision on the en- coder’s output which considerably hurt the discriminative feature learning of the encoder and vice visa for attention learning in the decoder. To alleviate this, we present a novel collaborative hybrid assignments training scheme, namely Co-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners. This new training scheme can easily enhance the encoder’s learning ability in end-to-end detectors by training the mul- tiple parallel auxiliary heads supervised by one-to-many la- bel assignments such as ATSS and Faster RCNN. In addi- tion, we conduct extra customized positive queries by ex- tracting the positive coordinates from these auxiliary heads to improve the training efficiency of positive samples in the decoder. In inference, these auxiliary heads are discarded and thus our method introduces no additional parameters and computational cost to the original detector while re- quiring no hand-crafted non-maximum suppression (NMS). We conduct extensive experiments to evaluate the effective- ness of the proposed approach on DETR variants, including DAB-DETR, Deformable-DETR, and DINO-Deformable- DETR. The state-of-the-art DINO-Deformable-DETR with Swin-L can be improved from 58.5% to 59.5% AP on COCO val. Surprisingly, incorporated with ViT-L backbone, we achieve 66.0% AP on COCO test-dev and 67.9% AP on LVIS val, outperforming previous methods by clear mar- gins with much fewer model sizes. Codes are available at https://github.com/Sense-X/Co-DETR. 1. Introduction Object detection is a fundamental task in computer vi- sion, which requires us to localize the object and classify its category. The seminal R-CNN families [11, 14, 27] and *Corresponding author. 0 20 40 60 80 100 120 Epoch 42 44 46 48 50 52 54 AP Co-DETR DINO-Deformable-DETR H-Deformable-DETR Deformable-DETR DAB-DETR DN-DETR Faster-RCNN HTC Figure 1. Performance of models with ResNet-50 on COCO val. Co-DETR outperforms other counterparts by a large margin. a series of variants [31, 37, 44] such as ATSS [41], Reti- naNet [21], FCOS [32], and PAA [17] lead to the significant breakthrough of object detection task. One-to-many label assignment is the core scheme of them, where each ground- truth box is assigned to multiple coordinates in the detec- tor’s output as the supervised target cooperated with propos- als [11, 27], anchors [21] or window centers [32]. Despite their promising performance, these detectors heavily rely on many hand-designed components like a non-maximum suppression procedure or anchor generation [1]. To con- duct a more flexible end-to-end detector, DEtection TRans- former (DETR) [1] is proposed to view the object detection as a set prediction problem and introduce the one-to-one set matching scheme based on a transformer encoder-decoder architecture. In this manner, each ground-truth box will only be assigned to one specific query, and multiple hand- designed components that encode prior knowledge are no longer needed. This approach introduces a flexible detec- tion pipeline and encourages many DETR variants to fur- ther improve it. However, the performance of the vanilla end-to-end object detector is still inferior to the traditional detectors with one-to-many label assignments.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Zong, Song, Liu - 2024 - DETRs with Collaborative Hybrid Assignments Training.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
