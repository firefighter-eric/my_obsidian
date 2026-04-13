# Nogueira, Cho - 2019 - Passage Re-ranking with BERT

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Nogueira, Cho - 2019 - Passage Re-ranking with BERT.pdf
- 原始 HTML：../../raw/html/Nogueira, Cho - 2019 - Passage Re-ranking with BERT.html
- 全文文本：../../raw/text/Nogueira, Cho - 2019 - Passage Re-ranking with BERT.md
- 作者：Nogueira, Cho
- 年份：2019
- 状态：已基于 arXiv HTML 整理

## 摘要

这篇论文是 transformer 进入搜索排序主线的标志性节点。它没有设计复杂新结构，而是用很直接的做法证明：把 query 和 candidate passage 拼成一个输入，使用 BERT 做相关性二分类，就足以显著刷新 MS MARCO passage reranking 的效果。这使搜索社区重新接受“重排序可以直接建模 query-document 交互”这一方向，并推动后续 `monoBERT`、`duoBERT` 和 listwise reranking 等路线。

## 关键事实

- 论文将 passage re-ranking 明确定位为典型多阶段搜索系统的第二阶段：先由 BM25 等高效召回器给出候选，再由更重的模型精排。
- 方法上采用简单的 cross-encoder 设计：query 作为 sentence A，passage 作为 sentence B，经 BERT 编码后用 `[CLS]` 表征做相关性分类。
- 文中在 MS MARCO passage ranking 上报告 `BERT Large` 相比此前最佳系统有显著提升，并给出相对 `27%` 的 `MRR@10` 改善。
- 作者还强调预训练 BERT 对排序任务的数据效率很高，即便只看训练数据中的一小部分 query-passage pair，也能达到强效果。
- 该文奠定了一个重要工程共识：cross-encoder reranker 效果通常很强，但成本高，因此适合作为多阶段架构中的后排精排器，而不是直接全库打分。

## 争议与不确定点

- 这篇论文强力证明了 BERT reranking 的效果，但并没有解决大规模索引上的实时性问题；它默认依赖 BM25 先召回候选。
- 文中任务以 passage ranking 为主，不等于长文档 web search、商品搜索或复杂工业排序都能直接照搬同一输入形式。

## 关联页面

- 主题：[搜索排序](../../wiki/topics/搜索排序.md)
- 主题：[BERT类双向Transformer语言模型](../../wiki/topics/BERT类双向Transformer语言模型.md)
- 概念：[ColBERT](../../wiki/concepts/ColBERT.md)
