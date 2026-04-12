# Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation.pdf
- 全文文本：../../raw/text/Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation.md
- 作者：Zuo et al.
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

Pre-trained language models have demon- strated superior performance in various natu- ral language processing tasks. However, these models usually contain hundreds of millions of parameters, which limits their practical- ity because of latency requirements in real- world applications. Existing methods train small compressed models via knowledge dis- tillation. However, performance of these small models drops signiﬁcantly compared with the pre-trained models due to their re- duced model capacity. We propose MoE- BERT, which uses a Mixture-of-Experts struc- ture to increase model capacity and inference speed. We initialize MoEBERT by adapt- ing the feed-forward neural networks in a pre-trained model into multiple experts. As such, representation power of the pre-trained model is largely retained. During inference, only one of the experts is activated, such that speed can be improved. We also propose a layer-wise distillation method to train MoE- BERT. We validate the efﬁciency and effec- tiveness of MoEBERT on natural language understanding and question answering tasks. Results show that the proposed method out- performs existing task-speciﬁc distillation al- gorithms. For example, our method outper- forms previous approaches by over 2% on the MNLI (mismatched) dataset. Our code is pub- licly available at https://github.com/ SimiaoZuo/MoEBERT.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
