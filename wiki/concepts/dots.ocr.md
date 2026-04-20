# dots.ocr

## 简介

`dots.ocr` 是 multilingual document layout parsing 专门模型的代表之一。在当前知识库中，它表示把版面检测、内容识别和阅读关系统一进单一文档 VLM 的路线。

## 关键属性

- 类型：文档 OCR / 文档解析模型
- 代表来源：[Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model](../../wiki/summaries/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md)
- 当前角色：统一式 multilingual document parsing 路线代表页

## 相关主张

- `dots.ocr` 主张 layout detection、content recognition 与 relational understanding 应共同学习，而不是继续分散在 pipeline 的不同阶段。
- 在当前知识库里，它是 specialized OCR document VLM 主线中的重要节点。

## 来源支持

- [Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model](../../wiki/summaries/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md)

## 关联页面

- [OCR](../topics/OCR.md)
- [PaddleOCR](./PaddleOCR.md)
- [GLM-OCR](./GLM-OCR.md)
- [DeepSeek-OCR](./DeepSeek-OCR.md)
- [传统 CV](../topics/传统%20CV.md)
