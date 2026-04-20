# Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model.pdf
- 全文文本：../../raw/text/Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model.md
- 作者：Li et al.
- 年份：2025
- 状态：已整理（基于 arXiv HTML 提取全文）

## 摘要

`dots.ocr` 代表的是把文档版面检测、内容识别与阅读关系理解统一进单一文档 VLM 的路线。它反对 `MinerU / PaddleOCR` 这类多阶段 pipeline 将 layout、recognition、reading order 分开做的方式，主张在单次端到端生成里 jointly 学习三类任务，并借此减少误差级联。与 `Nougat` 这类偏 academic PDF OCR 的模型相比，`dots.ocr` 更明确地把自己定义成 multilingual document layout parsing 模型，而不仅是文本转写器。

## 关键事实

- 论文把 document parsing 明确拆成三项核心能力：layout detection、content recognition、relational understanding；`dots.ocr` 的主张是这三者应在单一 end-to-end VLM 中共同学习，而不是继续拆成多阶段流水线。
- 作者将 `MinerU`、`PaddleOCR` 等方法归为 fragmented multi-stage pipeline，并把 error propagation 与 task synergy 缺失写成其主要问题；因此 `dots.ocr` 在 OCR topic 中代表的是对 pipeline 范式的直接统一化挑战。
- `dots.ocr` 的核心定位不是单语 OCR，而是 multilingual document parsing。论文同时引入 `XDocParse` 基准，覆盖 `126` 种语言，说明作者把语言覆盖面当成方法主张的一部分，而不是附属卖点。
- 在 `OmniDocBench` 上，论文报告 `dots.ocr` 取得新的 SOTA 分数，并给出 `87.5 (EN)`、`84.0 (CH)` 的整体表现；这使其成为当前 OCR / document parsing topic 中一条必须纳入的 specialized document VLM 路线。
- 模型不是小型轻量工具，而是较大的专门文档 VLM：其架构由 `1.2B` Vision Encoder 与约 `1.7B` Language Decoder 构成，强调高分辨率文档解析与统一生成，而不是端侧部署优先。
- 论文明确把 `olmOCR`、`Nougat`、`MonkeyOCR`、`Dolphin` 等近期 specialized document VLM 当作相关路线，并强调这些方法仍然要么省略关键任务、要么内部仍是 staged design；这进一步凸显 `dots.ocr` 想占据的是“真正 unified parsing”位置。

## 争议与不确定点

- 论文中的统一化主张很强，但其主要优势建立在作者自己的架构与数据引擎叙事之上；要判断它是否稳定优于多阶段 pipeline，仍需更多外部复现与跨数据验证。
- 模型参数规模不小，且 heavily 依赖内部数据合成引擎；因此它在“开源 specialized model”里的意义更偏方法路线节点，不自动等于最容易部署的工程方案。
- 当前全文来自 arXiv HTML 提取，核心叙事已清晰，但更细的 benchmark 表格与局部数值仍建议回看原文核对。

## 关联页面

- 概念：[dots.ocr](../../wiki/concepts/dots.ocr.md)
- 概念：[PaddleOCR](../../wiki/concepts/PaddleOCR.md)
- 概念：[GLM-OCR](../../wiki/concepts/GLM-OCR.md)
- 概念：[DeepSeek-OCR](../../wiki/concepts/DeepSeek-OCR.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
