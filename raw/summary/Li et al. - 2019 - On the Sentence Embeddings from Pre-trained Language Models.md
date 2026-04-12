# Li et al. - 2019 - On the Sentence Embeddings from Pre-trained Language Models

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Li et al. - 2019 - On the Sentence Embeddings from Pre-trained Language Models.pdf
- 全文文本：../../raw/text/Li et al. - 2019 - On the Sentence Embeddings from Pre-trained Language Models.md
- 作者：Li et al.
- 年份：2019
- 状态：已抽取全文，待精读

## 自动抽取摘要

Pre-trained contextual representations like BERT have achieved great success in natu- ral language processing. However, the sen- tence embeddings from the pre-trained lan- guage models without ﬁne-tuning have been found to poorly capture semantic meaning of sentences. In this paper, we argue that the se- mantic information in the BERT embeddings is not fully exploited. We ﬁrst reveal the the- oretical connection between the masked lan- guage model pre-training objective and the se- mantic similarity task theoretically, and then analyze the BERT sentence embeddings em- pirically. We ﬁnd that BERT always induces a non-smooth anisotropic semantic space of sentences, which harms its performance of semantic similarity. To address this issue, we propose to transform the anisotropic sen- tence embedding distribution to a smooth and isotropic Gaussian distribution through nor- malizing ﬂows that are learned with an un- supervised objective. Experimental results show that our proposed BERT-ﬂow method ob- tains signiﬁcant performance gains over the state-of-the-art sentence embeddings on a va- riety of semantic textual similarity tasks. The code is available at https://github.com/ bohanli/BERT-flow.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Li et al. - 2019 - On the Sentence Embeddings from Pre-trained Language Models.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
