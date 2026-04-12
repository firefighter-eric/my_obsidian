# 传统 CV

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页汇总多模态大模型成为主流之前，以及与其并行发展的视觉方法主线，重点包括视觉 Transformer、目标检测、OCR 与文档理解。这里的“传统CV”不是指只收录卷积网络时代材料，而是指不以通用多模态 LLM 为中心组织的视觉研究脉络。

## 核心问题

- Transformer 如何从 NLP 架构转化为视觉基础架构。
- 视觉任务为何会从手工 pipeline 走向 end-to-end set prediction 或统一预训练。
- OCR 与文档理解为何逐渐从专门模块化系统走向预训练基础模型。
- 通用视觉、文档视觉与多模态视觉的边界应如何划分。

## 主线脉络 / 方法分层

- 视觉基础模型化：`ViT` 证明纯 Transformer 可以直接服务图像分类，从而把“视觉是否必须依赖 CNN”改写为开放问题。
- end-to-end 检测主线：`DETR` 把目标检测重写为 set prediction，弱化了传统检测 pipeline 中大量手工组件的必要性。
- 文档 AI 预训练：`LayoutLMv3` 说明文档理解需要同时建模文字、版面与图像，并用统一 masking 目标学习多模态表示。
- OCR 的预训练化：`TrOCR` 把图像编码与文本生成统一到 Transformer 编解码框架中，代表 OCR 从任务专用 pipeline 走向预训练生成模型。
- 与多模态 LLM 的连接：虽然本页不以通用 MLLM 为中心，但文档理解路线已经明显朝向更生成式、更统一的模型接口推进。

## 关键争论与分歧

- Transformer 是否完全取代经典视觉结构：当前证据足以说明其已成为基础架构，但并不足以得出所有视觉任务都完成统一的结论。
- DETR 式 end-to-end 框架是否优于传统检测体系：其优点在于简化 pipeline，但训练效率和变体演化问题仍需更多材料才能全面比较。
- 文档理解应归入通用 CV 还是独立知识线：当前页面把它视为传统CV中的重要分支，但随着页数增长，未来可能拆成独立 topic。
- OCR 与文档理解是否应并入多模态主题：从当前证据看，两者仍有强任务特性，不宜直接并入“通用多模态”而失去问题边界。

## 证据基础

- [Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale](../../raw/summary/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)
- [Carion et al. - 2020 - End-to-End Object Detection with Transformers](../../raw/summary/Carion%20et%20al.%20-%202020%20-%20End-to-End%20Object%20Detection%20with%20Transformers.md)
- [Huang et al. - 2022 - LayoutLMv3 Pre-training for Document AI with Unified Text and Image Masking](../../raw/summary/Huang%20et%20al.%20-%202022%20-%20LayoutLMv3%20Pre-training%20for%20Document%20AI%20with%20Unified%20Text%20and%20Image%20Masking.md)
- [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](../../raw/summary/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)

## 代表页面

- [ViT](../concepts/ViT.md)
- [Transformer](../concepts/Transformer.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [TrOCR](../concepts/TrOCR.md)
- [DocLLM](../concepts/DocLLM.md)

## 未解决问题

- 当前页面对视频、neural rendering、talking head 等视觉生成分支仍未纳入正式主线。
- 表格结构识别、文档版面数据集与 OCR 之间的关系尚未整理成更细的比较层。
- 视觉基础模型与通用多模态 LLM 的接口层仍缺少专门页面承接。

## 关联页面

- [Slide 理解与生成](Slide%20理解与生成.md)
- [ViT](../concepts/ViT.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [TrOCR](../concepts/TrOCR.md)
