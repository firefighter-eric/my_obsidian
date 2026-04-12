# Lin et al. - 2022 - Robust High-Resolution Video Matting with Temporal Guidance

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Lin et al. - 2022 - Robust High-Resolution Video Matting with Temporal Guidance.pdf
- 全文文本：../../raw/text/Lin et al. - 2022 - Robust High-Resolution Video Matting with Temporal Guidance.md
- 作者：Lin et al.
- 年份：2022
- 状态：已抽取全文，待精读

## 自动抽取摘要

We introduce a robust, real-time, high-resolution hu- man video matting method that achieves new state-of-the- art performance. Our method is much lighter than previous approaches and can process 4K at 76 FPS and HD at 104 FPS on an Nvidia GTX 1080Ti GPU. Unlike most existing methods that perform video matting frame-by-frame as in- dependent images, our method uses a recurrent architecture to exploit temporal information in videos and achieves sig- niﬁcant improvements in temporal coherence and matting quality. Furthermore, we propose a novel training strat- egy that enforces our network on both matting and segmen- tation objectives. This signiﬁcantly improves our model’s robustness. Our method does not require any auxiliary in- puts such as a trimap or a pre-captured background image, so it can be widely applied to existing human matting ap- plications. Our code is available at https://peterl1n. github.io/RobustVideoMatting/ 1. Introduction Matting is the process of predicting the alpha matte and foreground color from an input frame. Formally, a frame I can be viewed as the linear combination of a foreground F and a background B through an α coefﬁcient: I = αF + (1 −α)B (1) By extracting α and F, we can composite the foreground object to a new background, achieving the background re- placement effect. Background replacement has many practical applica- tions. Many rising use cases, e.g. video conferencing and entertainment video creation, need real-time background re- placement on human subjects without green-screen props. *Work performed during an internship at ByteDance. Neural models are used for this challenging problem but the current solutions are not always robust and often gener- ate artifacts. Our research focuses on improving the matting quality and robustness for such applications. Most existing methods [18, 22, 34], despite being de- signed for video applications, process individual frames as independent images. Those approaches neglect the most widely available feature in videos: temporal information. Temporal information can improve video matting perfor- mance for many reasons. First, it allows the prediction of more coherent results, as the model can see multiple frames and its own predictions. This signiﬁcantly reduces ﬂicker and improves perceptual quality. Second, temporal information can improve matting robustness. In the cases where an individual frame might be ambiguous, e.g. the foreground color becomes similar to a passing object in the background, the model can better guess the boundary by referring to the previous frames. Third, temporal informa- tion allows the model to learn more about the background over time. When the camera moves, the background be- hind the subjects is revealed due to the perspective change. Even if the camera is ﬁxed, the occluded background still often reveals due to the subject’s movements. Having a bet- ter understanding of the background simpliﬁes the matting task. Therefore, we propose a recurrent architecture to ex- ploit the temporal information. Our method signiﬁcantly improves the matting quality and temporal coherence. It can be applied to all videos without any requirements for auxiliary inputs, such as a manually annotated trimap or a pre-captured background image. Furthermore, we propose a new training strategy to en- force our model on both matting and semantic segmen- tation objectives simultaneously. Most existing methods

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/Lin et al. - 2022 - Robust High-Resolution Video Matting with Temporal Guidance.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[传统CV](../topics/传统CV.md)
- 综合：暂无
