# Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.pdf
- 全文文本：../../raw/text/Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.md
- 作者：Fedus, Zoph, Shazeer
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) models defy this and instead select diﬀerent parameters for each in- coming example. The result is a sparsely-activated model—with an outrageous number of parameters—but a constant computational cost. However, despite several notable suc- cesses of MoE, widespread adoption has been hindered by complexity, communication costs, and training instability. We address these with the introduction of the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques mitigate the instabilities, and we show large sparse models may be trained, for the ﬁrst time, with lower precision (bﬂoat16) formats. We design models based oﬀT5-Base and T5-Large (Raﬀel et al., 2019) to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the “Colossal Clean Crawled Corpus”, and achieve a 4x speedup over the T5-XXL model.12 Keywords: mixture-of-experts, natural language processing, sparsity, large-scale machine learning, distributed computing ∗. Equal contribution. 1. JAX code for Switch Transformer and all model checkpoints are available at https://github.com/ google-research/t5x 2. Tensorﬂow code for Switch Transformer is available at https://github.com/tensorflow/mesh/blob/ master/mesh_tensorflow/transformer/moe.py ©2022 William Fedus, Barret Zoph and Noam Shazeer. License: CC-BY 4.0, see https://creativecommons.org/licenses/by/4.0/. Attribution requirements are provided at http://jmlr.org/papers/v23/21-0998.html. arXiv:2101.03961v3 [cs.LG] 16 Jun 2022 Fedus, Zoph and Shazeer Contents

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
