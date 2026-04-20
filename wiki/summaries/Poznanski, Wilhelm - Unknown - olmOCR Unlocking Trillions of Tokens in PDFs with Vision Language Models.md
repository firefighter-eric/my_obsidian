# Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models.pdf
- 全文文本：../../raw/text/Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models.md
- 作者：Poznanski, Wilhelm
- 年份：Unknown
- 状态：已抽取全文，待精读

## 摘要

`olmOCR` 代表的是 OCR / PDF parsing 向大规模数据生产基础设施演化的一条路线。它的重点不只是把页面文字识别出来，而是以开源 Python toolkit 的形式，把 PDF 线性化为自然阅读顺序文本，同时尽量保留章节、表格、列表和公式等结构信息，以便直接进入语言模型训练、RAG 与知识抽取流程。与 `Nougat` 这类更偏学术 PDF 转写模型不同，`olmOCR` 更强调多类型 PDF、成本、吞吐与批处理能力。

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models.md` 继续做深入整理。
- `olmOCR` 被定位为 open-source Python toolkit，而不是单纯 OCR 模型；其目标是把多样化 PDF 转成自然阅读顺序纯文本，同时保留结构化内容。
- 报告将 `Marker`、`MinerU` 与 `GOT-OCR 2.0` 作为主要对照对象，并通过人工 pairwise judgment 与成本估算，把 `olmOCR` 放进现实 PDF linearization 工具链竞争面。
- 文中还将 `Tesseract` 写为更早期的开源 OCR engine 里程碑，借此强调当前 PDF parsing 工具已经从传统 OCR 引擎演进到更强的结构恢复系统。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
