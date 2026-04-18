# 传统 CV

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论的是**不以通用多模态 LLM 为中心组织**的视觉研究主线。这里的“传统 CV”不是怀旧式标签，也不等于“卷积时代残余方法”；更准确地说，它指的是这样一类研究共同体：问题定义首先来自视觉任务本身，模型接口围绕分类、检测、OCR、文档解析、版面建模等任务目标展开，而不是先假设一个统一的对话式多模态代理，再把视觉能力嵌进去。

因此，本页的边界需要收紧到三个层面。第一，它讨论的是**视觉任务如何形成基础表征、任务接口与文档理解能力**，而不是所有非 LLM 论文的杂项汇编。第二，它把目标检测视为已经足够成熟、足够独立的一条主线，因此只保留其在整体视觉谱系中的位置，不在本页内部重新展开。第三，它把 OCR 与文档 AI 留在本页，不是因为它们与通用视觉完全同质，而是因为当前知识库证据更支持它们作为“视觉结构化理解”路线的一部分，而不是已经可以稳定拆成两个独立正式 topic。

从现有 summary 出发，本页最稳妥的定位是：它描述的是**视觉研究从任务专用 pipeline 走向统一 Transformer 表征，再进一步走向 layout-aware、generative 接口**的过渡地带。这个判断目前主要由 `ViT`、`LayoutLMv3`、`TrOCR` 三类来源共同支撑。

## 核心问题

- **视觉基础架构是否必须以卷积归纳偏置为核心**，还是可以被 patch 化、token 化的统一 Transformer 表示改写。
- **文档与 OCR 为什么没有停留在“检测 + 识别 + 规则后处理”**，而是逐步转向预训练与生成式接口。
- **通用视觉、文档视觉、OCR、版面理解之间究竟共享多少表示层**，又在哪些任务边界上仍然必须分开讨论。
- **视觉研究的“统一化”究竟指什么**：是统一 backbone、统一预训练目标，还是统一为自然语言驱动的生成接口。

## 主线脉络 / 方法分层

从当前证据看，本主题不宜按“模型家族名”来分，而应按**视觉对象和任务接口被如何重写**来分层。这样做的原因是：`ViT`、`LayoutLMv3`、`TrOCR` 虽然都与 Transformer 有关，但它们解决的并不是同一个问题。

- **视觉表征基础层**：`ViT` 的意义不只是提出一个新分类器，而是把“图像可以被切成 token 序列，并直接送入 Transformer”变成可行命题。它真正改写的是视觉基础表征的组织方式，把“视觉模型是否必须显式保留 CNN 式局部归纳偏置”从默认前提改成开放问题。也正因为如此，`ViT` 在本页里不是一篇普通分类论文，而是后续文档视觉、跨模态视觉与视觉 Transformer 家族的共同起点之一。
- **文档多模态建模层**：`LayoutLMv3` 代表的是另一类问题设定。它不是要证明 Transformer 能否处理图像，而是要证明**文字、版面和图像区域可以在统一预训练目标下共同学习**。这条线的价值在于，它把文档 AI 从“先 OCR，再喂给下游模型”的松耦合流程，推进到 layout-aware 的统一表示学习框架。这里的核心对象不再是自然图像，而是带有强空间结构约束的视觉文档。
- **识别到生成接口层**：`TrOCR` 的关键不是把 OCR 精度再抬高一点，而是把文本识别从 `CNN/RNN + LM` 组合系统，改写成**图像编码器加文本生成器的端到端序列生成问题**。这意味着 OCR 在接口层上开始向生成模型靠拢，其输出不再只是中间模块结果，而是可以被更大生成式工作流吸收的自然语言序列。
- **专门任务向独立 topic 外溢层**：目标检测在当前知识库里已经形成从 `Faster R-CNN` 到 `DETR / RT-DETR / Co-DETR` 的独立方法主线，因此本页只保留它作为传统视觉谱系中的关键支柱。保留这一定位是必要的，因为检测仍然是视觉任务结构化输出最典型的接口之一；但继续在本页中细讲，会削弱本页围绕“视觉表征统一化”和“文档视觉生成化”的主线。

把这几层放在一起，当前可以得到一个比原页更稳的综述判断：**传统 CV 在本库中的主线，不是“旧方法大全”，而是视觉任务接口如何从手工模块拼接，转向更统一的 token 表示、layout-aware 预训练和生成式解码。** 其中 `ViT` 解决基础表征问题，`LayoutLMv3` 解决结构化文档问题，`TrOCR` 解决识别接口生成化问题，而检测则已经外溢成独立主题。

## 关键争论与分歧

- **Transformer 是否已经“统一了视觉”**：现有证据只支持它已改写视觉基础架构与文档建模方式，不支持“所有视觉子任务都已被同一种训练范式稳定统一”。`ViT` 支撑的是基础表征层转向，`LayoutLMv3` 和 `TrOCR` 支撑的是部分任务接口统一，不能机械外推为全部视觉问题都已收敛。
- **文档 AI 应否继续留在传统 CV 中**：当前这样组织是合理的，因为 `LayoutLMv3` 与 `TrOCR` 仍然体现出强视觉结构约束与任务专用接口；但若后续知识库补入更多 `DocLLM`、通用文档 agent、版面生成与文档问答来源，文档 AI 可能更适合升级为独立 topic。也就是说，这个争论目前的成立条件是**证据面是否仍主要围绕版面与识别，而非围绕通用多模态推理**。
- **OCR 是否已经从识别任务变成纯生成任务**：`TrOCR` 证明生成式接口在 OCR 中可行且有效，但现有证据不足以说明 OCR 的评测逻辑、错误模式和数据依赖已经完全等同于通用文本生成。更稳妥的说法是：OCR 的**模型接口生成化了**，而问题本体并未因此消失。
- **“传统 CV”这个总题是否过宽**：是的，而且这个宽度本身就是当前页面的风险。它现在仍然能成立，是因为知识库里尚缺足够多的正式 topic 去分别承接文档 AI、OCR、表格理解、视觉基础模型等子线。一旦这些子线补齐，本页应进一步收缩为更强的总览页，而不是继续承载细节。

## 证据基础

- [Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale](../../wiki/summaries/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)
- [Huang et al. - 2022 - LayoutLMv3 Pre-training for Document AI with Unified Text and Image Masking](../../wiki/summaries/Huang%20et%20al.%20-%202022%20-%20LayoutLMv3%20Pre-training%20for%20Document%20AI%20with%20Unified%20Text%20and%20Image%20Masking.md)
- [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](../../wiki/summaries/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)

## 代表页面

- [ViT](../concepts/ViT.md)
- [Transformer](../concepts/Transformer.md)
- [CLIP](../concepts/CLIP.md)
- [Faster R-CNN](../concepts/Faster%20R-CNN.md)
- [DETR](../concepts/DETR.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [DocLayNet](../concepts/DocLayNet.md)
- [PubTables-1M](../concepts/PubTables-1M.md)
- [TrOCR](../concepts/TrOCR.md)
- [DocLLM](../concepts/DocLLM.md)
- [目标检测](目标检测.md)

## 未解决问题

- 当前页面对**视觉基础模型、文档 AI、OCR 生成化**三条子线的总结已经初步成形，但证据面仍偏薄，尚不足以支撑更细粒度的稳定断言，例如“统一预训练已成为文档理解唯一主流”。
- `DocLLM`、`OmniDocBench` 等来源虽已在库中出现，但当前页的核心判断仍主要由 `ViT / LayoutLMv3 / TrOCR` 支撑；若要把“文档理解正在从视觉模型转向 layout-aware 语言模型”写成稳定结论，必须先补更强的 `wiki/summaries/`。
- 表格解析、版面分析、OCR、文档问答之间目前还缺 comparison 页，因此本页只能把它们写成相邻路线，尚不能给出更严格的边界裁剪。
- 视频、neural rendering、talking-head、3D 视觉生成等内容仍未纳入本页主线；在现有证据下，把它们直接吸进“传统 CV”只会稀释页面边界。

## 关联页面

- [Slide 理解与生成](./Slide%20理解与生成.md)
- [目标检测](目标检测.md)
- [ViT](../concepts/ViT.md)
- [Transformer](../concepts/Transformer.md)
- [CLIP](../concepts/CLIP.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [DocLayNet](../concepts/DocLayNet.md)
- [PubTables-1M](../concepts/PubTables-1M.md)
- [Faster R-CNN](../concepts/Faster%20R-CNN.md)
- [DETR](../concepts/DETR.md)
- [TrOCR](../concepts/TrOCR.md)
- [Kosmos-2](../concepts/Kosmos-2.md)
- [Kosmos-2.5](../concepts/Kosmos-2.5.md)
- [MiniCPM-V](../concepts/MiniCPM-V.md)
- [OFA](../concepts/OFA.md)
- [data2vec](../concepts/data2vec.md)
- [HuBERT](../concepts/HuBERT.md)
- [Tip-Adapter](../concepts/Tip-Adapter.md)
