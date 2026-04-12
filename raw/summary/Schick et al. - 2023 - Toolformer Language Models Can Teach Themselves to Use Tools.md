# Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.pdf
- 全文文本：../../raw/text/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.md
- 作者：Schick et al.
- 年份：2023
- 状态：已抽取全文，待精读

## 自动抽取摘要

Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or fac- tual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q&A system, a search engine, a translation system, and a calendar. Toolformer achieves substan- tially improved zero-shot performance across a variety of downstream tasks, often competi- tive with much larger models, without sacriﬁc- ing its core language modeling abilities.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统NLP](../topics/传统NLP.md)
- 综合：暂无
