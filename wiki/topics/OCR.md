# OCR

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论 OCR 这一类任务：把图像、扫描页、PDF 页面或其他文字密集视觉输入，转成**机器可用、顺序正确、结构尽量保真的文本表示**。它当然包含狭义的字符识别，但在当前知识库可支撑的范围内，OCR 已不能被理解成单纯“看清字形”的局部识别问题；一旦对象从单行文字扩展到整页文档，OCR 就同时碰到阅读顺序、版面结构、公式表格表示以及输出格式保真的问题。

topic 层面最需要澄清的，是 OCR 与相邻问题的边界。第一，OCR 不等于整个文档理解；问答、抽取、推理可以建立在 OCR 之上，但不自动属于 OCR 本体。第二，OCR 也不等于普通目标检测，虽然文本框检测和版面分块常被用作前置模块，但 OCR 的最终目标是**可读文本转写**而不是只输出位置框。第三，当前 OCR 的重要变化并不是“字符识别已经 solved”，而是**输出对象从 plain text 走向 markup、坐标化文本块与结构化页面表示**，这让 OCR 与 document parsing、multimodal reading 的边界显著靠近。

从现有 summary 组合看，本页最稳的主题定义不是“所有文字识别方法大全”，而是：**OCR 如何从局部字符识别流水线，走向端到端生成式识别，再走向文档级结构化转写与多模态系统内嵌能力。**

## 核心问题

- **OCR 的真正输出是什么**：只是字符序列、自然阅读顺序下的纯文本，还是带结构、样式、坐标与层级的文档表示。
- **版面信息在 OCR 中处于什么地位**：它究竟只是后处理辅助，还是决定整页 OCR 质量的核心建模对象。
- **端到端生成式接口是否优于传统模块流水线**：它带来统一训练与更强表达能力，但也引入格式漂移、幻觉和评测复杂度。
- **OCR 应该作为独立系统存在，还是被文档模型与通用多模态模型吸收为一种能力**。
- **OCR 该如何评测**：字符级正确率不足以覆盖阅读顺序、公式等价表达、表格结构和跨页文档保真度。

## 主线脉络 / 方法分层

从当前 evidence base 看，OCR 主线不宜按“模型名字”或“是否用 Transformer”来分，而应按**模型把难点放在什么层级**来分层。

- **流水线重写层**：`TrOCR` 的重要性不在于又一次刷新文字识别精度，而在于它明确把 OCR 从常见的 `CNN/RNN + language model` 组合系统，改写成预训练视觉编码器加文本解码器的端到端生成问题。这个转向很关键，因为它意味着 OCR 不再天然被拆成“视觉特征提取、字符序列建模、语言纠错”几个分离模块，而可以在统一预训练和统一解码接口下完成。
- **版面与顺序显式建模层**：`LayoutReader` 与 `LayoutLMv3` 共同说明，整页 OCR 的核心瓶颈并不只在字形识别。`LayoutReader` 直接把阅读顺序检测单列出来，并证明顺序错误会显著伤害下游文档任务；`LayoutLMv3` 则进一步说明，文字、版面与图像区域应在统一预训练目标下共同学习。二者合在一起指向一个稳定判断：**当 OCR 对象变成 visually rich document 时，layout 不是附属信息，而是识别结果可用性的组成部分。**
- **结构化转写层**：`Nougat` 与 `Kosmos-2.5` 把 OCR 往前推进了一步。这里的任务不再是把页面“抄成纯文本”，而是把学术 PDF 或文字密集图像直接转写成 markdown、带结构的文本块或其他机器可消费格式。这个节点特别重要，因为它把 OCR 从“识别字”推进到“重建文档表达形式”。也正因此，公式、表格、章节层次和样式信息不再只是外围细节，而成为输出接口的一部分。
- **专门文档 VLM 层**：`dots.ocr`、`DeepSeek-OCR` 家族与 `GLM-OCR` 代表了 OCR 在 2025 年 12 月到 2026 年 3 月之间非常明显的一次 specialized model 密集演化。`dots.ocr` 试图把 layout、recognition 与 reading order 真正统一到单一 end-to-end 文档 VLM；`DeepSeek-OCR` 则把 OCR 当作 vision-text compression 与 causal visual flow 的实验场，重点是 token economy 与编码顺序重写；`GLM-OCR` 则以 `0.9B` 紧凑规模、`MTP` 和两阶段 layout + region recognition 设计，走向更强的现实部署与吞吐折中。三者共同说明，OCR 前沿已经不只是“更强 OCR 工具”，而是在快速分化出不同问题意识的专门文档模型路线。
- **开源工具与工程系统层**：`PaddleOCR 3.0`、`olmOCR` 与更早作为 baseline 出现的 `Tesseract` 一起说明，OCR 的竞争对象早已不只是论文模型，而是完整工具链。`PaddleOCR 3.0` 把多语言识别、文档解析、KIE、部署与 `MCP` 接口收敛成生产级 toolkit；`olmOCR` 把 PDF 线性化、成本与吞吐写成核心目标；而 `Tesseract` 则在 `TrOCR` 与 `LayoutReader` 中更多以开源 OCR engine baseline 角色出现。这个层次的重要性在于：**OCR 的现实价值越来越取决于能否提供可部署、可组合、可进入 RAG / agent 工作流的系统接口，而不只是单点评测分数。**
- **文档级系统化层**：`DocLLM`、`Marker`、`MinerU`、`OmniDocBench` 与上述工具链一起说明 OCR 正在进入文档级系统语境。`DocLLM` 把 OCR 上游产物与 layout-aware 语言模型接口更紧地绑在一起；`Marker / MinerU` 则在 `olmOCR` 与 `OmniDocBench` 语境下代表现实 PDF parsing pipeline 工具；`OmniDocBench` 进一步提醒我们，现实系统比较的对象已经不是单个识别器，而是模块化 pipeline 与端到端多模态方法在多种文档上的整体表现。
- **能力内嵌层**：`Qwen2.5-VL` 表明 OCR 已被吸收到更通用的多模态系统能力栈中。这里 OCR 不再单独以一个服务名出现，而是作为文档 HTML 表示、图表表格解析、多语言文本密集图像理解的一部分被联合训练。这个变化很大，但它不等于“独立 OCR 问题消失了”；更准确的说法是，**OCR 的接口正在内嵌进通用模型，而不是它的保真要求被自动解决。**

如果把这几层连起来，当前最稳的综合判断是：**OCR 的长期演化，不是从识别模型线性升级到更大模型，而是不断把“什么算输出”“哪些结构必须保留”“layout 应否进入主体模型”“系统应如何部署和接入下游”这些问题重新定义。** `TrOCR` 重写了识别接口，`LayoutReader / LayoutLMv3` 重写了页面级约束，`Nougat / Kosmos-2.5` 重写了输出格式，`dots.ocr / DeepSeek-OCR / GLM-OCR` 把 specialized document VLM 路线迅速拉开，`PaddleOCR / olmOCR / DocLLM / Qwen2.5-VL` 则进一步把 OCR 推进到更大的文档系统、工程工具链与通用多模态能力框架里。

## 关键争论与分歧

- **OCR 是否已经被通用多模态模型“吃掉”**：现有证据不支持这么写。`Qwen2.5-VL` 确实说明 OCR 能力可以被统一训练并内嵌进 VL 系统，但 `olmOCR`、`OmniDocBench` 和 `LayoutReader` 都提示，真实文档场景仍然非常依赖阅读顺序、结构保留、长文档稳定性与系统评测，这些问题并不会因为模型更通用就自动消失。
- **OCR 是否只是字符识别任务**：现有 evidence base 明确不支持这种缩写法。至少在整页文档、学术 PDF 与 text-intensive image 场景下，阅读顺序、段落层级、公式表格和结构化输出已经成为问题本体的一部分。更稳妥的表述是：**字符识别是 OCR 的局部能力，但不是 OCR topic 的完整边界。**
- **生成式 OCR 是否天然优于模块化 pipeline / toolkit**：`TrOCR`、`Nougat`、`Kosmos-2.5` 让生成式路线非常有吸引力，因为它统一了接口并扩大了输出表达力；但 `PaddleOCR 3.0`、`OmniDocBench` 与 `olmOCR` 又提醒我们，模块化方法、toolkit 化 pipeline 与端到端方法在不同文档类型和部署约束下各有局限，现实比较尚未收敛。因此当前更合理的判断是：**生成式路线是主线增量，但不是已经无条件取代 pipeline / toolkit 的共识。**
- **OCR 评测是否还能靠字符级文本相似度解决**：`CDM` 对公式识别评测的修正已经清楚表明，不同等价表示会让纯文本指标失真；`OmniDocBench` 也说明 document parsing 需要多层次、跨模块评测。因此 OCR 的评测争议不是附属问题，而是方法路线分歧本身的一部分。
- **文档 OCR 与 document understanding 是否应合并成同一主题**：当前还不宜直接合并。现有 summary 足以支持 OCR 独立成题，因为识别、阅读顺序与转写格式已经形成独立主线；但同时也能看出，OCR 正在不断向 document understanding 边界推进。更准确的组织方式是：**先把 OCR 作为独立入口写稳，再把文档理解视为其上层邻接问题。**

## 证据基础

- [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](../../wiki/summaries/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)
- [Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection](../../wiki/summaries/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.md)
- [Huang et al. - 2022 - LayoutLMv3 Pre-training for Document AI with Unified Text and Image Masking](../../wiki/summaries/Huang%20et%20al.%20-%202022%20-%20LayoutLMv3%20Pre-training%20for%20Document%20AI%20with%20Unified%20Text%20and%20Image%20Masking.md)
- [Blecher et al. - 2023 - Nougat Neural Optical Understanding for Academic Documents](../../wiki/summaries/Blecher%20et%20al.%20-%202023%20-%20Nougat%20Neural%20Optical%20Understanding%20for%20Academic%20Documents.md)
- [Lv et al. - 2023 - Kosmos-2.5 A Multimodal Literate Model](../../wiki/summaries/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.md)
- [Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model](../../wiki/summaries/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md)
- [Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
- [Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)
- [Duan et al. - 2026 - GLM-OCR Technical Report](../../wiki/summaries/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md)
- [Wang et al. - 2023 - DocLLM A layout-aware generative language model for multimodal document understanding](../../wiki/summaries/Wang%20et%20al.%20-%202023%20-%20DocLLM%20A%20layout-aware%20generative%20language%20model%20for%20multimodal%20document%20understanding.md)
- [Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation](../../wiki/summaries/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.md)
- [Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations](../../wiki/summaries/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md)
- [Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models](../../wiki/summaries/Poznanski,%20Wilhelm%20-%20Unknown%20-%20olmOCR%20Unlocking%20Trillions%20of%20Tokens%20in%20PDFs%20with%20Vision%20Language%20Models.md)
- [PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report](../../wiki/summaries/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md)
- [Bai et al. - 2025 - Qwen2.5-VL Technical Report](../../wiki/summaries/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md)

## 代表页面

- [TrOCR](../concepts/TrOCR.md)
- [dots.ocr](../concepts/dots.ocr.md)
- [DeepSeek-OCR](../concepts/DeepSeek-OCR.md)
- [GLM-OCR](../concepts/GLM-OCR.md)
- [PaddleOCR](../concepts/PaddleOCR.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [DocLLM](../concepts/DocLLM.md)
- [Kosmos-2.5](../concepts/Kosmos-2.5.md)
- [Qwen2.5-VL](../concepts/Qwen2.5-VL.md)
- [传统 CV](./传统%20CV.md)

## 未解决问题

- 当前页已经能较稳定地写出 **生成式识别、阅读顺序、结构化转写、专门文档 VLM、工程工具链、能力内嵌与评测升级** 七层结构，但表格 OCR、手写场景、低资源多语言与端侧部署等子线仍缺更直接 summary，因此还不宜展开成更细的正式子 topic。
- `dots.ocr / DeepSeek-OCR / GLM-OCR` 已经构成一条很值得单独比较的 2025-12 到 2026-03 specialized OCR VLM 子线，但当前库里还没有 comparison 页系统拆开它们在 unified parsing、token compression、compact deployment 三个维度上的差异。
- `PaddleOCR 3.0` 已经把 OCR toolkit、document parsing、KIE 与部署接口写成一条工程主线，但当前库里还缺对 `Tesseract / EasyOCR / GOT-OCR / Mathpix / Marker / MinerU / PaddleOCR` 的专门 comparison 页，因此工具链层的优劣仍主要以 topic 综合方式呈现。
- `Nougat`、`Kosmos-2.5`、`Qwen2.5-VL` 都把 OCR 推向 markdown / HTML / 坐标化文本块等 richer output，但当前库里还缺独立 comparison 页来系统比较这些输出接口的保真度、可逆性与下游兼容性。
- `OmniDocBench` 与 `CDM` 已经说明评测问题很关键，但当前知识库仍缺对 CER/WER、结构保真、公式等价性、阅读顺序一致性等指标体系的统一整理。
- `DocLLM`、`olmOCR` 与更通用的 VL 系统之间目前更像相邻路线而非完全可替代路线；若后续补入更多文档 agent 与 PDF parsing summary，本页可能需要再拆出 `文档解析 / document parsing` 的独立 topic。

## 关联页面

- [传统 CV](./传统%20CV.md)
- [Slide 理解与生成](./Slide%20理解与生成.md)
- [TrOCR](../concepts/TrOCR.md)
- [dots.ocr](../concepts/dots.ocr.md)
- [DeepSeek-OCR](../concepts/DeepSeek-OCR.md)
- [GLM-OCR](../concepts/GLM-OCR.md)
- [PaddleOCR](../concepts/PaddleOCR.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [DocLLM](../concepts/DocLLM.md)
- [Kosmos-2.5](../concepts/Kosmos-2.5.md)
- [Qwen2.5-VL](../concepts/Qwen2.5-VL.md)
- [MiniCPM-V](../concepts/MiniCPM-V.md)
- [PubTables-1M](../concepts/PubTables-1M.md)
- [DocLayNet](../concepts/DocLayNet.md)
