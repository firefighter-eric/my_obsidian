# Qwen Team - 2024 - Hello Qwen2

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen2/
- 全文文本：../../raw/text/Qwen Team - 2024 - Hello Qwen2.md
- 作者：Qwen Team
- 年份：2024
- 状态：已整理

## 摘要

Qwen2 是 Qwen 家族从 Qwen1.5 进一步走向“多语言、高上下文、强代码与数学能力”的关键代际。官方把它描述为从 Qwen1.5 演进而来的新一代语言模型家族，并强调更广的语言覆盖、更强的长上下文与更成熟的 post-training。

## 关键事实

- Qwen2 提供 `0.5B / 1.5B / 7B / 57B-A14B / 72B` 五个主要尺寸，同时覆盖 base 与 instruct 版本。
- 官方明确写出 Qwen2 在英语和中文之外额外训练了 `27` 种语言。
- Qwen2-7B-Instruct 与 Qwen2-72B-Instruct 在官方评估中支持最长 `128K` 上下文。
- Qwen2 把 `GQA` 扩展到全尺寸模型，强调更快推理和更低显存占用。
- 官方把 post-training 描述为“可扩展、低人工标注”的组合流程，包含 `SFT`、reward model、online DPO 与 alignment tax 控制。
- 博客明确把 Qwen2 看作后续多模态扩展的底座，并预告将向视觉和音频方向延伸。

## 争议与不确定点

- 虽然博客末尾给出 `Qwen2 Technical Report` 引用信息，但当前知识库这里使用的是博客来源，不等同于直接摄入论文全文。
- 57B-A14B 的 MoE 路线与 dense 路线如何在训练成本和能力上对比，博客只给出结果级描述。

## 关联页面

- 概念：[Qwen](../../wiki/concepts/Qwen.md)
- 概念：[Qwen2](../../wiki/concepts/Qwen2.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
