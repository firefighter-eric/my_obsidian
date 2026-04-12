# Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents.pdf
- 全文文本：../../raw/text/Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents.md
- 作者：Smock, Pesala, Abraham
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

Recently, signiﬁcant progress has been made applying machine learning to the problem of table structure inference and extraction from unstructured documents. However, one of the greatest challenges remains the creation of datasets with complete, unambiguous ground truth at scale. To ad- dress this, we develop a new, more comprehensive dataset for table extraction, called PubTables-1M. PubTables-1M contains nearly one million tables from scientiﬁc articles, supports multiple input modalities, and contains detailed header and location information for table structures, making it useful for a wide variety of modeling approaches. It also addresses a signiﬁcant source of ground truth inconsistency observed in prior datasets called oversegmentation, using a novel canonicalization procedure. We demonstrate that these improvements lead to a signiﬁcant increase in training per- formance and a more reliable estimate of model performance at evaluation for table structure recognition. Further, we show that transformer-based object detection models trained on PubTables-1M produce excellent results for all three tasks of detection, structure recognition, and functional analysis without the need for any special customization for these tasks. Data and code will be released at https://github. com/microsoft/table-transformer. 1. Introduction A table is a compact, structured representation for storing data and communicating it in documents and other manners of presentation. In its presented form, however, a table, such as the one in Fig. 1, may not and often does not explicitly represent its logical structure. This is an important problem as a signiﬁcant amount of data is communicated through doc- uments, but without structure information this data cannot be used in further applications. The problem of inferring a table’s structure from its pre- sentation and converting it to a structured form is known as Figure 1. An example of a presentation table whose underlying structure must be inferred, either manually or by automated sys- tems. table extraction (TE). TE entails three subtasks [6], which we illustrate in Fig. 2: table detection (TD), which locates the table; table structure recognition (TSR), which recognizes the structure of a table in terms of rows, columns, and cells; and functional analysis (FA), which recognizes the keys and values of the table. TE is challenging for automated sys- tems [9,12,17,23] due to the wide variety of formats, styles, and structures found in presented tables. Recently, there has been a shift in the research litera- ture from traditional rule-based methods [4,11,18] for TE to data-driven methods based on deep learning (DL) [14,17,22]. The primary advantage of DL methods is that they can learn to be more robust to the wide variety of table presentation formats. However, manually annotating tables for TSR is a difﬁcult and time-consuming process [7]. To overcome this, researchers have turned recently to crowd-sourcing to con- struct larger datasets [9,22,23]. These datasets are assembled from tables appearing in documents created by thousands of authors, where an annotation for each table’s structure and content is available in a markup format such as HTML, XML, or LaTeX. While crowd-sourcing solves the problem of dataset size, repurposing annotations originally unintended for TE and automatically converting these to ground truth presents its own set of challenges with respect to completeness, consis- tency, and quality. This includes not only what information is present but how explicitly this information is represented.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
