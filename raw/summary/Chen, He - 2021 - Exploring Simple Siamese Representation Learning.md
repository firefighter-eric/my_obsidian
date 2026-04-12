# Chen, He - 2021 - Exploring Simple Siamese Representation Learning

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Chen, He - 2021 - Exploring Simple Siamese Representation Learning.pdf
- 全文文本：../../raw/text/Chen, He - 2021 - Exploring Simple Siamese Representation Learning.md
- 作者：Chen, He
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

Siamese networks have become a common structure in various recent models for unsupervised visual representa- tion learning. These models maximize the similarity be- tween two augmentations of one image, subject to certain conditions for avoiding collapsing solutions. In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders. Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in pre- venting collapsing. We provide a hypothesis on the impli- cation of stop-gradient, and further show proof-of-concept experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representa- tion learning. Code will be made available. 1. Introduction Recently there has been steady progress in un-/self- supervised representation learning, with encouraging re- sults on multiple visual tasks (e.g., [2, 17, 8, 15, 7]). Despite various original motivations, these methods generally in- volve certain forms of Siamese networks [4]. Siamese net- works are weight-sharing neural networks applied on two or more inputs. They are natural tools for comparing (includ- ing but not limited to “contrasting”) entities. Recent meth- ods deﬁne the inputs as two augmentations of one image, and maximize the similarity subject to different conditions. An undesired trivial solution to Siamese networks is all outputs “collapsing” to a constant. There have been several general strategies for preventing Siamese networks from collapsing. Contrastive learning [16], e.g., instantiated in SimCLR [8], repulses different images (negative pairs) while attracting the same image’s two views (positive pairs). The negative pairs preclude constant outputs from the solu- tion space. Clustering [5] is another way of avoiding con- stant output, and SwAV [7] incorporates online clustering into Siamese networks. Beyond contrastive learning and encoder f similarity encoder f predictor h stop-grad image x x1 x2 Figure 1. SimSiam architecture. Two augmented views of one image are processed by the same encoder network f (a backbone plus a projection MLP). Then a prediction MLP h is applied on one side, and a stop-gradient operation is applied on the other side. The model maximizes the similarity between both sides. It uses neither negative pairs nor a momentum encoder. clustering, BYOL [15] relies only on positive pairs but it does not collapse in case a momentum encoder is used. In this paper, we report that simple Siamese networks can work surprisingly well with none of the above strategies for preventing collapsing. Our model directly maximizes the similarity of one image’s two views, using neither neg- ative pairs nor a momentum encoder. It works with typical batch sizes and does not rely on large-batch training. We illustrate this “SimSiam” method in Figure 1. Thanks to the conceptual simplicity, SimSiam can serve as a hub that relates several existing methods. In a nut- shell, our method can be thought of as “BYOL without the momentum encoder”. Unlike BYOL but like SimCLR and SwAV, our method directly shares the weights between the two branches, so it can also be thought of as “SimCLR without negative pairs”, and “SwAV without online cluster- ing”. Interestingly, SimSiam is related to each method by removing one of its core components. Even so, SimSiam does not cause collapsing and can perform competitively. We empirically show that collapsing solutions do exist, but a stop-gradient operation (Figure 1) is critical to pre- vent such solutions. The importance of stop-gradient sug- gests that there should be a different underlying optimiza- tion problem that is being solved. We hypothesize that there are implicitly two sets of variables, and SimSiam behaves like alternating between optimizing each set. We provide proof-of-concept experiments to verify this hypothesis.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Chen, He - 2021 - Exploring Simple Siamese Representation Learning.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
