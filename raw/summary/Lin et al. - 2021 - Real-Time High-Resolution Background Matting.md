# Lin et al. - 2021 - Real-Time High-Resolution Background Matting

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Lin et al. - 2021 - Real-Time High-Resolution Background Matting.pdf
- 全文文本：../../raw/text/Lin et al. - 2021 - Real-Time High-Resolution Background Matting.md
- 作者：Lin et al.
- 年份：2021
- 状态：已抽取全文，待精读

## 自动抽取摘要

We introduce a real-time, high-resolution background re- placement technique which operates at 30fps in 4K resolu- tion, and 60fps for HD on a modern GPU. Our technique is based on background matting, where an additional frame of the background is captured and used in recovering the al- pha matte and the foreground layer. The main challenge is to compute a high-quality alpha matte, preserving strand- level hair details, while processing high-resolution images in real-time. To achieve this goal, we employ two neural networks; a base network computes a low-resolution result which is reﬁned by a second network operating at high- resolution on selective patches. We introduce two large- scale video and image matting datasets: VideoMatte240K and PhotoMatte13K/85. Our approach yields higher qual- ity results compared to the previous state-of-the-art in back- ground matting, while simultaneously yielding a dramatic boost in both speed and resolution. Our code and data is available at https://grail.cs.washington.edu/ projects/background-matting-v2/ 1. Introduction Background replacement, a mainstay in movie special effects, now enjoys wide-spread use in video conferencing tools like Zoom, Google Meet, and Microsoft Teams. In ad- dition to adding entertainment value, background replace- *Equal contribution. ment can enhance privacy, particularly in situations where a user may not want to share details of their location and environment to others on the call. A key challenge of this video conferencing application is that users do not typically have access to a green screen or other physical props used to facilitate background replacement in movie special effects. While many tools now provide background replacement functionality, they yield artifacts at boundaries, particu- larly in areas where there is ﬁne detail like hair or glasses (Figure 1). In contrast, traditional image matting methods [6, 16, 17, 30, 9, 2, 7] provide much higher quality re- sults, but do not run in real-time, at high resolution, and frequently require manual input. In this paper, we intro- duce the ﬁrst fully-automated, real-time, high-resolution matting technique, producing state-of-the-art results at 4K (3840×2160) at 30fps and HD (1920×1080) at 60fps. Our method relies on capturing an extra background image to compute the alpha matte and the foreground layer, an ap- proach known as background matting. Designing a neural network that can achieve real- time matting on high-resolution videos of people is ex- tremely challenging, especially when ﬁne-grained details like strands of hair are important; in contrast, the previous state-of-the-art method [28] is limited to 512×512 at 8fps. Training a deep network on such a large resolution is ex- tremely slow and memory intensive. It also requires large volumes of images with high-quality alpha mattes to gener- alize; the publicly available datasets [33, 25] are too limited.

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Lin et al. - 2021 - Real-Time High-Resolution Background Matting.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
