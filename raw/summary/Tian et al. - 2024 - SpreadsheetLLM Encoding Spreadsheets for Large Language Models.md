# Tian et al. - 2024 - SpreadsheetLLM Encoding Spreadsheets for Large Language Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Tian et al. - 2024 - SpreadsheetLLM Encoding Spreadsheets for Large Language Models.pdf
- 全文文本：../../raw/text/Tian et al. - 2024 - SpreadsheetLLM Encoding Spreadsheets for Large Language Models.md
- 作者：Tian et al.
- 年份：2024
- 状态：已抽取全文，待精读

## 自动抽取摘要

Spreadsheets are characterized by their exten- sive two-dimensional grids, flexible layouts, and varied formatting options, which pose sig- nificant challenges for large language models (LLMs). In response, we introduce SPREAD- SHEETLLM, pioneering an efficient encod- ing method designed to unleash and optimize LLMs’ powerful understanding and reason- ing capability on spreadsheets. Initially, we propose a vanilla serialization approach that incorporates cell addresses, values, and for- mats. However, this approach was limited by LLMs’ token constraints, making it im- practical for most applications. To tackle this challenge, we develop SHEETCOMPRESSOR, an innovative encoding framework that com- presses spreadsheets effectively for LLMs. It comprises three modules: structural-anchor- based compression, inverse index translation, and data-format-aware aggregation. It signif- icantly improves performance in spreadsheet table detection task, outperforming the vanilla approach by 25.6% in GPT4’s in-context learn- ing setting. Moreover, fine-tuned LLM with SHEETCOMPRESSOR has an average compres- sion ratio of 25×, but achieves a state-of-the-art 78.9% F1 score, surpassing the best existing models by 12.3%. Finally, we propose Chain of Spreadsheet for downstream tasks of spread- sheet understanding and validate in a new and demanding spreadsheet QA task. We methodi- cally leverage the inherent layout and structure of spreadsheets, demonstrating that SPREAD- SHEETLLM is highly effective across a variety of spreadsheet tasks.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Tian et al. - 2024 - SpreadsheetLLM Encoding Spreadsheets for Large Language Models.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
