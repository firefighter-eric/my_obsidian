# He, Girshick, Dollar - 2019 - Rethinking imageNet pre-training

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/He, Girshick, Dollar - 2019 - Rethinking imageNet pre-training.pdf
- 全文文本：../../raw/text/He, Girshick, Dollar - 2019 - Rethinking imageNet pre-training.md
- 作者：He, Girshick, Dollar
- 年份：2019
- 状态：已抽取全文，待精读

## 自动抽取摘要

We report competitive results on object detection and in- stance segmentation on the COCO dataset using standard models trained from random initialization. The results are no worse than their ImageNet pre-training counterparts even when using the hyper-parameters of the baseline sys- tem (Mask R-CNN) that were optimized for ﬁne-tuning pre- trained models, with the sole exception of increasing the number of training iterations so the randomly initialized models may converge. Training from random initialization is surprisingly robust; our results hold even when: (i) us- ing only 10% of the training data, (ii) for deeper and wider models, and (iii) for multiple tasks and metrics. Experi- ments show that ImageNet pre-training speeds up conver- gence early in training, but does not necessarily provide regularization or improve ﬁnal target task accuracy. To push the envelope we demonstrate 50.9 AP on COCO ob- ject detection without using any external data—a result on par with the top COCO 2017 competition results that used ImageNet pre-training. These observations challenge the conventional wisdom of ImageNet pre-training for depen- dent tasks and we expect these discoveries will encourage people to rethink the current de facto paradigm of ‘pre- training and ﬁne-tuning’ in computer vision. 1. Introduction Deep convolutional neural networks [21, 23] revolution- ized computer vision arguably due to the discovery that fea- ture representations learned on a pre-training task can trans- fer useful information to target tasks [9, 6, 50]. In recent years, a well-established paradigm has been to pre-train models using large-scale data (e.g., ImageNet [39]) and then to ﬁne-tune the models on target tasks that often have less training data. Pre-training has enabled state-of-the-art re- sults on many tasks, including object detection [9, 8, 36], image segmentation [29, 13], and action recognition [42, 4]. A path to ‘solving’ computer vision then appears to be paved by pre-training a ‘universal’ feature representation on ImageNet-like data at massive scale [44, 30]. Attempts along this path have pushed the frontier to up to 3000× [30] the size of ImageNet. However, the success of these experiments is mixed: although improvements have been 0

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/He, Girshick, Dollar - 2019 - Rethinking imageNet pre-training.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
