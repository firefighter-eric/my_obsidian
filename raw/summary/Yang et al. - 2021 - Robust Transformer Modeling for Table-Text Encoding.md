# Yang et al. - 2021 - Robust Transformer Modeling for Table-Text Encoding

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Yang et al. - 2021 - Robust Transformer Modeling for Table-Text Encoding.pdf
- 全文文本：../../raw/text/Yang et al. - 2021 - Robust Transformer Modeling for Table-Text Encoding.md
- 作者：Yang et al.
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

Understanding tables is an important aspect of natural language understanding. Existing mod- els for table understanding require lineariza- tion of the table structure, where row or col- umn order is encoded as an unwanted bias. Such spurious biases make the model vulner- able to row and column order perturbations. Additionally, prior work has not thoroughly modeled the table structures or table-text align- ments, hindering the table-text understanding ability. In this work, we propose a robust and structurally aware table-text encoding architec- ture TABLEFORMER, where tabular structural biases are incorporated completely through learnable attention biases. TABLEFORMER is (1) strictly invariant to row and column or- ders, and, (2) could understand tables better due to its tabular inductive biases. Our eval- uations showed that TABLEFORMER outper- forms strong baselines in all settings on SQA, WTQ and TABFACT table reasoning datasets, and achieves state-of-the-art performance on SQA, especially when facing answer-invariant row and column order perturbations (6% im- provement over the best baseline), because pre- vious SOTA models’ performance drops by 4% - 6% when facing such perturbations while TABLEFORMER is not affected.1

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Yang et al. - 2021 - Robust Transformer Modeling for Table-Text Encoding.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
