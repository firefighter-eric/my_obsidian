# Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection.pdf
- 全文文本：../../raw/text/Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection.md
- 作者：Wang et al.
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

Reading order detection is the cornerstone to understanding visually-rich documents (e.g., receipts and forms). Unfortunately, no existing work took advantage of advanced deep learn- ing models because it is too laborious to anno- tate a large enough dataset. We observe that the reading order of WORD documents is em- bedded in their XML metadata; meanwhile, it is easy to convert WORD documents to PDFs or images. Therefore, in an automated man- ner, we construct ReadingBank, a benchmark dataset that contains reading order, text, and layout information for 500,000 document im- ages covering a wide spectrum of document types. This ﬁrst-ever large-scale dataset un- leashes the power of deep neural networks for reading order detection. Speciﬁcally, our pro- posed LayoutReader captures the text and lay- out information for reading order prediction using the seq2seq model. It performs almost perfectly in reading order detection and signif- icantly improves both open-source and com- mercial OCR engines in ordering text lines in their results in our experiments. We will release the dataset and model at https:// aka.ms/layoutreader.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
