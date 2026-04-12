# Migacz - 2017 - Intro

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Migacz - 2017 - Intro.pdf
- 全文文本：../../raw/text/Migacz - 2017 - Intro.md
- 作者：Migacz
- 年份：2017
- 状态：已抽取全文，待精读

## 自动抽取摘要

8-bit Inference with TensorRT Szymon Migacz, NVIDIA May 8, 2017 Intro ● Goal: Convert FP32 CNNs into INT8 without significant accuracy loss. ● Why: INT8 math has higher throughput, and lower memory requirements. ● Challenge: INT8 has significantly lower precision and dynamic range than FP32. ● Solution: Minimize loss of information when quantizing trained model weights to INT8 and during INT8 computation of activations. ● Result: Method was implemented in TensorRT. It does not require any additional fine tuning or retraining. Outline ● INT8 compute ● Quantization ● Calibration ● Workflow in TensorRT ● Results INT8 Inference Challenge ● INT8 has significantly lower precision and dynamic range compared to FP32. ● Requires more than a simple type conversion from FP32 to INT8. Dynamic Range Min Positive Value FP32 -3.4 x 1038 ~ +3.4 x 1038 1.4 x 10-45 FP16 -65504 ~ +65504 5.96 x 10-8 INT8 -1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Migacz - 2017 - Intro.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统NLP](../topics/传统NLP.md)
- 综合：暂无
