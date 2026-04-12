# Gordon, Duh, Andrews - 2020 - Compressing BERT Studying the Effects of Weight Pruning on Transfer Learning

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Gordon, Duh, Andrews - 2020 - Compressing BERT Studying the Effects of Weight Pruning on Transfer Learning.pdf
- 全文文本：../../raw/text/Gordon, Duh, Andrews - 2020 - Compressing BERT Studying the Effects of Weight Pruning on Transfer Learning.md
- 作者：Gordon, Duh, Andrews
- 年份：2020
- 状态：已抽取全文，待精读

## 自动抽取摘要

Pre-trained feature extractors, such as BERT for natural language processing and VGG for computer vision, have become effective meth- ods for improving deep learning models with- out requiring more labeled data. While ef- fective, these feature extractors may be pro- hibitively large for some deployment scenar- ios. We explore weight pruning for BERT and ask: how does compression during pre- training affect transfer learning? We ﬁnd that pruning affects transfer learning in three broad regimes. Low levels of pruning (30-40%) do not affect pre-training loss or transfer to down- stream tasks at all. Medium levels of pruning increase the pre-training loss and prevent use- ful pre-training information from being trans- ferred to downstream tasks. High levels of pruning additionally prevent models from ﬁt- ting downstream datasets, leading to further degradation. Finally, we observe that ﬁne- tuning BERT on a speciﬁc task does not im- prove its prunability. We conclude that BERT can be pruned once during pre-training rather than separately for each task without affecting performance.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Gordon, Duh, Andrews - 2020 - Compressing BERT Studying the Effects of Weight Pruning on Transfer Learning.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
