# Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations.pdf
- 全文文本：../../raw/text/Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations.md
- 作者：Ouyang et al.
- 年份：Unknown
- 状态：已抽取全文，待精读

## 自动抽取摘要

Document content extraction is crucial in computer vi- sion, especially for meeting the high-quality data needs of large language models (LLMs) and retrieval-augmented generation (RAG) technologies. However, current docu- ment parsing methods suffer from significant limitations in terms of diversity and comprehensive evaluation. To address these challenges, we introduce OmniDocBench, a novel multi-source benchmark designed to advance auto- mated document content extraction. OmniDocBench in- cludes a meticulously curated and annotated high-quality evaluation dataset comprising nine diverse document types, such as academic papers, textbooks, slides, among others. Our benchmark provides a flexible and comprehensive eval- uation framework with 19 layout category labels and 14 at- tribute labels, enabling multi-level assessments across en- tire datasets, individual modules, or specific data types. Us- ing OmniDocBench, we perform an exhaustive compara- tive analysis of existing modular pipelines and multimodal end-to-end methods, highlighting their limitations in han- dling document diversity and ensuring fair evaluation. Om- niDocBench establishes a robust, diverse, and fair evalu- ation standard for the document content extraction field, offering crucial insights for future advancements and fos- tering the development of document parsing technologies. The codes and dataset is available in https://github. com/opendatalab/OmniDocBench. 1. Introduction Document parsing is a foundational task in computer vi- sion, focused on accurately extracting content from docu- ments [18, 36, 39, 41, 45]. High-quality document content ∗The authors contributed equally. † Project lead. ‡ Corresponding author (heconghui@pjlab.org.cn). extraction typically involves the integration of multiple al- gorithmic modules. Layout detection algorithms identify different content areas on a page, OCR technology converts images of text regions into text, while formula and table recognition models identify specific regions and transform them into corresponding source code. These modules and reading order algorithms form a comprehensive process of converting documents into machine-readable formats. With large models increasingly requiring high-quality data, the importance of document content extraction has be- come more pronounced. Although vast amounts of data are available online for training, knowledge-rich document data is relatively scarce. Documents such as academic papers and technical reports contain rich structured information that can significantly enhance the knowledge depth of large models. Moreover, the development of retrieval-augmented generation (RAG) [10, 21] technology relies on extracting accurate information from documents to improve the qual- ity and relevance of generated content. Consequently, re- search in document content extraction has intensified, lead- ing to a series of pipeline-based high-quality document ex- traction algorithms [36] and the emergence of end-to-end multimodal large model solutions [3, 5, 6, 27, 39, 40, 42]. These methods have significantly improved document con- tent parsing quality, providing robust support for the needs of large models and RAG technology. In analyzing current module-based pipeline and multi- modal end-to-end methods, we identified several limita- tions. For instance, methods like Marker and MinerU, which are mainstream pipeline methods, primarily evaluate individual modules on academic paper data, lacking doc- ument diversity and comprehensive evaluation results. Al- though MinerU considers the generalization of diverse data, it only demonstrates this through a single model and visual- ization results, lacking overall end-to-end evaluation. Mul- timodal large model methods [3, 5, 27, 39, 40], while easier to use than pipeline methods, lack performance validation on diverse documents, and some evaluation metrics are in-

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
