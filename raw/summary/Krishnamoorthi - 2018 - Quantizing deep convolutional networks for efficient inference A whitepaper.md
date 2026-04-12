# Krishnamoorthi - 2018 - Quantizing deep convolutional networks for efficient inference A whitepaper

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Krishnamoorthi - 2018 - Quantizing deep convolutional networks for efficient inference A whitepaper.pdf
- 全文文本：../../raw/text/Krishnamoorthi - 2018 - Quantizing deep convolutional networks for efficient inference A whitepaper.md
- 作者：Krishnamoorthi
- 年份：2018
- 状态：已抽取全文，待精读

## 自动抽取摘要

We present an overview of techniques for quantizing convolutional neural net- works for inference with integer weights and activations. 1. Per-channel quantization of weights and per-layer quantization of activations to 8-bits of precision post-training produces classiﬁcation accuracies within 2% of ﬂoating point networks for a wide variety of CNN architectures (sec- tion 3.1). 2. Model sizes can be reduced by a factor of 4 by quantizing weights to 8- bits, even when 8-bit arithmetic is not supported. This can be achieved with simple, post training quantization of weights (section 3.1). 3. We benchmark latencies of quantized networks on CPUs and DSPs and ob- serve a speedup of 2x-3x for quantized implementations compared to ﬂoat- ing point on CPUs. Speedups of up to 10x are observed on specialized pro- cessors with ﬁxed point SIMD capabilities, like the Qualcomm QDSPs with HVX (section 6). 4. Quantization-aware training can provide further improvements, reducing the gap to ﬂoating point to 1% at 8-bit precision. Quantization-aware training also allows for reducing the precision of weights to four bits with accuracy losses ranging from 2% to 10%, with higher accuracy drop for smaller net- works (section 3.2). 5. We introduce tools in TensorFlow and TensorFlowLite for quantizing con- volutional networks (Section 3). 6. We review best practices for quantization-aware training to obtain high ac- curacy with quantized weights and activations (section 4). 7. We recommend that per-channel quantization of weights and per-layer quan- tization of activations be the preferred quantization scheme for hardware ac- celeration and kernel optimization. We also propose that future processors and hardware accelerators for optimized inference support precisions of 4, 8 and 16 bits (section 7).

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Krishnamoorthi - 2018 - Quantizing deep convolutional networks for efficient inference A whitepaper.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统NLP](../topics/传统NLP.md)
- 综合：暂无
