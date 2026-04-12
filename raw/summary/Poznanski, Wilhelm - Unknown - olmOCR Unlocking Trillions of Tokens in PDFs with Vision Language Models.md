# Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models.pdf
- 全文文本：../../raw/text/Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models.md
- 作者：Poznanski, Wilhelm
- 年份：Unknown
- 状态：已抽取全文，待精读

## 自动抽取摘要

PDF documents have the potential to provide trillions of novel, high-quality tokens for training language models. However, these documents come in a diversity of types with differing formats and visual layouts that pose a challenge when attempting to extract and faithfully represent the underlying content for language model use. We present olmOCR, an open-source Python toolkit for processing PDFs into clean, linearized plain text in natural reading order while preserving structured content like sections, tables, lists, equations, and more. Our toolkit runs a fine-tuned 7B vision language model (VLM) trained on a sample of 260,000 pages from over 100,000 crawled PDFs with diverse properties, including graphics, handwritten text and poor quality scans. olmOCR is optimized for large-scale batch processing, able to scale flexibly to different hardware setups and convert a million PDF pages for only $190 USD. We release all components of olmOCR including VLM weights, data and training code, as well as inference code built on serving frameworks including vLLM and SGLang. Code allenai/olmocr Weights & Data allenai/olmocr Demo olmocr.allenai.org 1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
