# Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen2.5-omni/
- 全文文本：../../raw/text/Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All.md
- 作者：Qwen Team
- 年份：2025
- 状态：已整理

## 摘要

Qwen2.5-Omni 是 Qwen 家族第一次明确把文本、图像、音频、视频与语音输出统一进一个 end-to-end 模型。官方强调其不是“多个单模态模块的拼装”，而是具备实时流式输入输出能力的 omni 模型。

## 关键事实

- Qwen2.5-Omni 支持同时理解 `text / image / audio / video`，并能同时输出文本与自然语音。
- 官方把其架构概括为 `Thinker-Talker`：Thinker 负责理解与高层表示，Talker 负责流式语音生成。
- 新的位置编码 `TMRoPE` 用于对齐视频时间戳与音频时间轴。
- 官方声称 Qwen2.5-Omni 在多模态整合任务上达到 SOTA，并在同尺寸比较下优于 Qwen2-Audio、接近 Qwen2.5-VL-7B。
- 这一代的核心变化是把“视觉语言模型 + 音频模型”推进到统一 omni 模型。

## 争议与不确定点

- 当前页基于官方博客，而非完整论文 PDF。
- “接近或优于单模态专门模型”的说法来自官方 benchmark 选择，仍需外部复核。

## 关联页面

- 概念：[Qwen2.5-Omni](../../wiki/concepts/Qwen2.5-Omni.md)
- 概念：[Qwen3.5-Omni](../../wiki/concepts/Qwen3.5-Omni.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
