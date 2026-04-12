# Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks.pdf
- 全文文本：../../raw/text/Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks.md
- 作者：Xiao et al.
- 年份：2023
- 状态：已抽取全文，待精读

## 自动抽取摘要

We introduce Florence-2, a novel vision foundation model with a unified, prompt-based representation for a va- riety of computer vision and vision-language tasks. While existing large vision models excel in transfer learning, they struggle to perform a diversity of tasks with simple in- structions, a capability that implies handling the complex- ity of various spatial hierarchy and semantic granularity. Florence-2 was designed to take text-prompt as task instruc- tions and generate desirable results in text forms, whether it be captioning, object detection, grounding or segmen- tation. This multi-task learning setup demands large- scale, high-quality annotated data. To this end, we co- developed FLD-5B that consists of 5.4 billion comprehen- sive visual annotations on 126 million images, using an it- erative strategy of automated image annotation and model refinement. We adopted a sequence-to-sequence structure to train Florence-2 to perform versatile and comprehensive vi- sion tasks. Extensive evaluations on numerous tasks demon- strated Florence-2 to be a strong vision foundation model contender with unprecedented zero-shot and fine-tuning ca- pabilities. 1. Introduction In the realm of Artificial General Intelligence (AGI) sys- tems, there has been a notable shift towards utilizing pre- trained, versatile representations, acknowledged for task- agnostic benefits accross diverse applications. This trend is evident in natural language processing (NLP), where ad- vanced models [5, 6, 19, 43, 65, 66] show adaptability with comprehensive knowledge spanning various domains and tasks with simple instructions. The success of NLP moti- vates a parallel approach in computer vision. Universal representation for diverse vision-related tasks presents unique challenges, notably the need for compre- hensive perceptual abilities. Unlike NLP, which deals Region-level Image-level Pixel-level None semantic Fine-grained semantic Coarse semantic A woman riding a bike down a street next to a red car. The image shows a person riding a red bicycle on a road with a red car in the background. The road is lined with trees on both sides and there is another person riding another bicycle in front of her. The date " 9/22/2023" is visible in the bottom. person car road red vintage car on street Spatial Hierarchy Semantic Granularity FLD-5B (Comprehensive Annotations) Florence-2 (Unified Architecture) classification caption detailed caption visual grounding & object detection regional proposal segmentation phrase segmentation visual grounding & object detection Figure 1. We aim to build a vision foundation model to en- able extensive perception capabilities including spatial hierarchy and semantic granularity. To achieve this, a single unified model Florence-2 is pre-trained on our FLD-5B dataset encompassing a total of 5.4B comprehensive annotations across 126M images, which are collected by our Florence data engine. mainly with text, computer vision requires handling in- tricate visual data like object location, masked contours, and attributes. Attaining universal representation in com- puter vision demands adept management of a spectrum of complex tasks, organized two-dimensionally as illustrated in Figure 1: • Spatial Hierarchy: The model must discern spatial details across varying scales, understanding image- level concepts and fine-grained pixel specifics. Ac- commodating the intricate spatial hierarchy within vi- sion demands the model’s proficiency in handling di- verse levels of granularity. • Semantic Granularity: Universal representation in computer vision should span a spectrum of seman- tic granularity. The model transitions from high-level captions to nuanced descriptions, enabling versatile understanding for diverse applications.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20%20理解与生成.md)
- 综合：暂无
