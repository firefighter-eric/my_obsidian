# Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World.pdf
- 全文文本：../../raw/text/Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World.md
- 作者：Peng et al.
- 年份：2023
- 状态：已抽取全文，待精读

## 自动抽取摘要

We introduce KOSMOS-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., “[text span](bounding boxes)”, where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GRIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), KOSMOS-2 integrates the grounding capability into downstream applications. We evaluate KOSMOS-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at https://aka.ms/kosmos-2. Grounding Kosmos-2: Multimodal Large Language Model [a campfire](<loc4> <loc1007>) [It](<loc44> <loc863>) sits next to Referring Figure 1: KOSMOS-2 is a multimodal large language model that has new capabilities of multimodal grounding and referring. KOSMOS-2 can understand multimodal input, follow instructions, perceive object descriptions (e.g., bounding boxes), and ground language to the visual world. ∗Equal contribution. † Corresponding author. arXiv:2306.14824v3 [cs.CL] 13 Jul 2023 Input Prompt the left eye of emoji Question: How many cows are here? Answer: Question: What does the sign say? Answer: Completion two cows present in the image. The sign says "Welcome to Carnaby Street." Input Prompt Question: What is the biggest difference betwen bottle-1 and bottle-2? Answer: Question: Why is this animal unusual? Answer: (1) (2) (3) Completion The biggest difference between bottle-1 and bottle-2 is the label. This animal is unique because it is a turtle. Turtles are known for their slow and steady pace, whereas rabbits are known to be agile and quick. It is not common for a turtle to be faster than a rabbit, especially in terms of speed. a buoy attached to the boat. (4) (5) (6) Question: What is it? Answer: Describe this image in detail: Completion A snowman sits next to a campfire in the snow. He is wearing a hat, scarf, and mittens. There are several pots nearby, likely containing a hot meal for the snowman. The scene is serene and peaceful, with the snowman looking at the fire with a warm smile. (7) Input Prompt Figure 2: Selected examples generated from KOSMOS-2. The examples include (1) visual grounding, (2)-(3) grounded question answering, (4)-(6) multimodal referring via bounding boxes, and (7) grounded image captioning. 2

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
