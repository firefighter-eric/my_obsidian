# MoE

## 简介

MoE 是 Mixture-of-Experts 的缩写。在当前知识库中，它表示“只激活部分专家参数以提升容量与效率”的稀疏模型架构路线。

## 关键属性

- 类型：模型架构 / 稀疏激活方法
- 代表来源：
  - [Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../raw/summary/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)
  - [Unknown - 2024 - DeepSeek-V3 Technical Report](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
  - [Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation](../../raw/summary/Zuo%20et%20al.%20-%202022%20-%20MoEBERT%20from%20BERT%20to%20Mixture-of-Experts%20via%20Importance-Guided%20Adaptation.md)
- 当前角色：连接大规模稀疏训练与开源模型效率工程的结构概念

## 相关主张

- `Fedus et al. 2022` 强调 MoE 通过稀疏激活在近似固定计算成本下扩展总参数规模。
- 在当前知识库里，MoE 不只属于早期 Switch Transformer 路线，也已通过 `DeepSeek-V3` 进入当代开源大模型主线。
- `Zuo et al. 2022` 说明 MoE 也可作为从致密模型迁移到专家结构的一种改造思路，而不只服务于超大预训练模型。

## 来源支持

- [Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../raw/summary/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation](../../raw/summary/Zuo%20et%20al.%20-%202022%20-%20MoEBERT%20from%20BERT%20to%20Mixture-of-Experts%20via%20Importance-Guided%20Adaptation.md)
- [LLM 预训练](LLM%20预训练.md)

## 关联页面

- [DeepSeek](./DeepSeek.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [LLM 预训练](LLM%20预训练.md)
- [传统 NLP](传统%20NLP.md)
