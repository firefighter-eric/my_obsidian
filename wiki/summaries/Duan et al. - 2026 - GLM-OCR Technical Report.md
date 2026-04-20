# Duan et al. - 2026 - GLM-OCR Technical Report

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Duan et al. - 2026 - GLM-OCR Technical Report.pdf
- 全文文本：../../raw/text/Duan et al. - 2026 - GLM-OCR Technical Report.md
- 作者：Duan et al.
- 年份：2026
- 状态：已整理（基于 arXiv HTML 提取全文）

## 摘要

`GLM-OCR` 代表的是紧凑型、多任务、可生产部署的 document understanding OCR 模型。它不像 `DeepSeek-OCR` 那样把重点放在 token compression，也不像 `dots.ocr` 那样主打统一端到端 parsing，而是试图在较小参数规模下，把文本、公式、表格、KIE 与部署速度一起做好。它通过 `0.9B` 模型规模、`MTP` 多 token 预测，以及 `PP-DocLayout-V3 + region-level recognition` 的两阶段设计，在 OCR topic 中占据的是“compact but production-minded specialized OCR model”位置。

## 关键事实

- `GLM-OCR` 是 `0.9B` 参数紧凑型多模态 OCR 模型，由 `0.4B` CogViT visual encoder 与 `0.5B` GLM decoder 组成，明确强调效率与识别性能的平衡。
- 为解决 OCR 中标准 autoregressive decoding 过慢的问题，论文引入 `Multi-Token Prediction (MTP)`，训练时每步预测多个 token，推理平均每步生成约 `5.2` 个 token，并声称带来约 `50%` throughput 提升。
- 在系统层面，`GLM-OCR` 采取两阶段设计：先由 `PP-DocLayout-V3` 做 layout analysis，再进行 parallel region-level recognition。也就是说，它不是纯 end-to-end 单模型，而是有意识地在 layout 与 recognition 之间分层，以换取稳定性与并行效率。
- 论文报告其在 `OmniDocBench v1.5` 上达到 `94.6`，并在 text、formula、table、KIE 等多个公开和工业场景 benchmark 上表现很强；这使它成为 OCR topic 中当前必须纳入的高性能紧凑路线。
- 作者明确把 real-world deployment 写成主问题：支持 `vLLM`、`SGLang`、`Ollama` 部署，并提供 fine-tuning 能力，说明该模型面向的不只是 benchmark，而是可服务化的文档生产场景。
- 从方法结构看，`GLM-OCR` 与 `PaddleOCR` 之间存在明显连接：它自身使用 `PP-DocLayout-V3` 做 layout analysis，这说明开源 OCR toolkit 与 specialized OCR VLM 在现实系统中并非互斥，而是会互相借力。

## 争议与不确定点

- `GLM-OCR` 的高性能部分建立在 `PP-DocLayout-V3 + GLM-OCR Core` 的组合上，因此若要与纯 end-to-end 模型做公平对比，需要明确它并不是完全单阶段体系。
- 论文同时覆盖 benchmark、部署、SDK、MaaS 与 fine-tuning，边界已超出狭义 OCR；它更准确地说是 document understanding OCR 系统，而不是只做文本识别。
- 当前页基于 arXiv HTML 抽取，核心结论清晰，但若后续需要精确整理 benchmark 排名或部署吞吐细节，仍建议回看 PDF 表格。

## 关联页面

- 概念：[GLM-OCR](../../wiki/concepts/GLM-OCR.md)
- 概念：[GLM](../../wiki/concepts/GLM.md)
- 概念：[PaddleOCR](../../wiki/concepts/PaddleOCR.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
