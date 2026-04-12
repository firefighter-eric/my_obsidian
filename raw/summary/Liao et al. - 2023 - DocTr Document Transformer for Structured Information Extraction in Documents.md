# Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents.pdf
- 全文文本：../../raw/text/Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents.md
- 作者：Liao et al.
- 年份：2023
- 状态：已抽取全文，待精读

## 自动抽取摘要

We present a new formulation for structured information extraction (SIE) from visually rich documents. It aims to address the limitations of existing IOB tagging or graph- based formulations, which are either overly reliant on the correct ordering of input text or struggle with decoding a complex graph. Instead, motivated by anchor-based object detectors in vision, we represent an entity as an anchor word and a bounding box, and represent entity linking as the as- sociation between anchor words. This is more robust to text ordering, and maintains a compact graph for entity linking. The formulation motivates us to introduce 1) a DOCument TRansformer (DocTr) that aims at detecting and associating entity bounding boxes in visually rich documents, and 2) a simple pre-training strategy that helps learn entity detection in the context of language. Evaluations on three SIE bench- marks show the effectiveness of the proposed formulation, and the overall approach outperforms existing solutions. 1. Introduction Structured information extraction (SIE) from documents, as shown in Fig 1, is the process of extracting entities and their relationships, and returning them in a structured format. Structured information in a document is usually visually-rich – it is not only determined by the content of text but also the layout, typesetting, and/or figures and tables present in the document. Therefore, unlike the traditional information ex- traction task in nature language processing (NLP) [8, 3, 30] where the input is plain text (usually with a given reading order), SIE assumes the image representation of a document is available, and a pre-built optical character recognition (OCR) system may provide the unstructured text (i.e., with- out proper reading order). This is a practical assumption for day-to-day processing of business documents, where the *Corresponding author liahaofu@amazon.com †Work done at AWS AI Labs 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 CHOCOLATE MILK SHAKE

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
