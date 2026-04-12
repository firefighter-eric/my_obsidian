# Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation.pdf
- 全文文本：../../raw/text/Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation.md
- 作者：Wang et al.
- 年份：2024
- 状态：已抽取全文，待精读

## 自动抽取摘要

Formula recognition presents significant challenges due to the complicated structure and varied notation of mathemati- cal expressions. Despite continuous advancements in formula recognition models, the evaluation metrics employed by these models, such as BLEU and Edit Distance, still exhibit notable limitations. They overlook the fact that the same formula has diverse representations and is highly sensitive to the distri- bution of training data, thereby causing the unfairness in for- mula recognition evaluation. To this end, we propose a Char- acter Detection Matching (CDM) metric, ensuring the evalua- tion objectivity by designing a image-level rather than LaTex- level metric score. Specifically, CDM renders both the model- predicted LaTeX and the ground-truth LaTeX formulas into image-formatted formulas, then employs visual feature ex- traction and localization techniques for precise character- level matching, incorporating spatial position information. Such a spatially-aware and character-matching method of- fers a more accurate and equitable evaluation compared with previous BLEU and Edit Distance metrics that rely solely on text-based character matching. Experimentally, we evalu- ated various formula recognition models using CDM, BLEU, and ExpRate metrics. Their results demonstrate that the CDM aligns more closely with human evaluation standards and pro- vides a fairer comparison across different models by eliminat- ing discrepancies caused by diverse formula representations.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
