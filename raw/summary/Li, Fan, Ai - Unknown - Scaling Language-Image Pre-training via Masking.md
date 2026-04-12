# Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking.pdf
- 全文文本：../../raw/text/Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking.md
- 作者：Li, Fan, Ai
- 年份：Unknown
- 状态：已抽取全文，待精读

## 自动抽取摘要

We present Fast Language-Image Pre-training (FLIP), a simple and more efﬁcient method for training CLIP [52]. Our method randomly masks out and removes a large por- tion of image patches during training. Masking allows us to learn from more image-text pairs given the same wall-clock time and contrast more samples per iteration with similar memory footprint. It leads to a favorable trade-off between accuracy and training time. In our experiments on 400 mil- lion image-text pairs, FLIP improves both accuracy and speed over the no-masking baseline. On a large diversity of downstream tasks, FLIP dominantly outperforms the CLIP counterparts trained on the same data. Facilitated by the speedup, we explore the scaling behavior of increasing the model size, data size, or training length, and report encour- aging results and comparisons. We hope that our work will foster future research on scaling vision-language learning. 1. Introduction Language-supervised visual pre-training, e.g., CLIP [52], has been established as a simple yet powerful methodology for learning representations. Pre-trained CLIP models stand out for their remarkable versatility: they have strong zero- shot transferability [52]; they demonstrate unprecedented quality in text-to-image generation (e.g., [53, 55]); the pre- trained encoder can improve multimodal and even unimodal visual tasks. Like the role played by supervised pre-training a decade ago [40], language-supervised visual pre-training is new fuel empowering various tasks today. Unlike classical supervised learning with a pre-deﬁned label set, natural language provides richer forms of supervi- sion, e.g., on objects, scenes, actions, context, and their re- lations, at multiple levels of granularity. Due to the complex nature of vision plus language, large-scale training is essen- tial for the capability of language-supervised models. For example, the original CLIP models [52] were trained on 400 million data for 32 epochs—which amount to 10,000 Ima- geNet [16] epochs, taking thousands of GPU-days [52, 36]. Even using high-end infrastructures, the wall-clock training 0 50 100 150 200 250 training time (hours) 68 69 70 71 72 73 zero-shot accuracy (%) 3.7 speedup mask 0% (our CLIP repro.) mask 50% mask 75% Figure 1. Accuracy vs. training time trade-off. With a high masking ratio of 50% or 75%, our FLIP method trains faster and is more accurate than its CLIP counterpart. All entries are bench- marked in 256 TPU-v3 cores. Training is done on LAION-400M for 6.4, 12.8, or 32 epochs, for each masking ratio. Accuracy is evaluated by zero-shot transfer on the ImageNet-1K validation set. The model is ViT-L/16 [20]. More details are in Fig. 3. As the CLIP baseline takes ∼2,500 TPU-days training, a speedup of 3.7× can save ∼1,800 TPU-days. time is still a major bottleneck hindering explorations on scaling vision-language learning. We present Fast Language-Image Pre-training (FLIP), a simple method for efﬁcient CLIP training. Inspired by the sparse computation of Masked Autoencoders (MAE) [29], we randomly remove a large portion of image patches dur- ing training. This design introduces a trade-off between “how carefully we look at a sample pair” vs. “how many sample pairs we can process”. Using masking, we can: (i) see more sample pairs (i.e., more epochs) under the same wall-clock training time, and (ii) compare/contrast more sample pairs at each step (i.e., larger batches) under simi- lar memory footprint. Empirically, the beneﬁts of process- ing more sample pairs greatly outweigh the degradation of per-sample encoding, resulting in a favorable trade-off. By removing 50%-75% patches of a training image, our method reduces computation by 2-4×; it also allows using

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
