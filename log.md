# Wiki Log

本页是 LLM Wiki 的追加式操作日志。

## [2026-04-16] ingest | 扩充开放模型主线与 GLM Kimi 对照

涉及页面：

- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- [开放模型家族与中国重要家族对照](./wiki/comparisons/%E5%BC%80%E6%94%BE%E6%A8%A1%E5%9E%8B%E5%AE%B6%E6%97%8F%E4%B8%8E%E4%B8%AD%E5%9B%BD%E9%87%8D%E8%A6%81%E5%AE%B6%E6%97%8F%E5%AF%B9%E7%85%A7.md)
- [BLOOM](./wiki/concepts/BLOOM.md)
- [Mistral 7B](./wiki/concepts/Mistral%207B.md)
- [Mixtral](./wiki/concepts/Mixtral.md)
- [Gemma](./wiki/concepts/Gemma.md)
- [Gemma 2](./wiki/concepts/Gemma%202.md)
- [DBRX](./wiki/concepts/DBRX.md)
- [OpenELM](./wiki/concepts/OpenELM.md)
- [Phi-3](./wiki/concepts/Phi-3.md)
- [OLMo 2](./wiki/concepts/OLMo%202.md)
- [GLM](./wiki/concepts/GLM.md)
- [Kimi](./wiki/concepts/Kimi.md)
- [index](./index.md)

关键变更：

- 新增 `BLOOM / MPT / Mistral 7B / Mixtral / Gemma / Gemma 2 / StarCoder2 / DBRX / OpenELM / Phi-3 / OLMo 2 / Falcon 3 / GLM / Kimi` 的原始来源链路，并补齐对应 `raw/html`、`raw/pdf` 与 `raw/text`
- 为上述模型家族新增对应 `wiki/summaries/` 与 `wiki/concepts/` 页面，统一明确 `open-weight`、`开放发布但限制较多`、`API 或闭源` 的边界
- 在 `LLM 预训练` 中补入开放模型家族阶段与中国重要家族对照节点的叙述，避免把 `GLM` 与 `Kimi` 混进同一种开放性类别
- 新增 comparison 页 [开放模型家族与中国重要家族对照](./wiki/comparisons/%E5%BC%80%E6%94%BE%E6%A8%A1%E5%9E%8B%E5%AE%B6%E6%97%8F%E4%B8%8E%E4%B8%AD%E5%9B%BD%E9%87%8D%E8%A6%81%E5%AE%B6%E6%97%8F%E5%AF%B9%E7%85%A7.md)，承接开放性与家族角色的横向整理
- 更新根级 `index.md`，把 summary 计数增至 `224`，并补齐新增 summary、concept 与 comparison 导航

## [2026-04-15] ingest | 并入并移除 LLM 基础脉络

涉及页面：

- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- [指令对齐与 post-training](./wiki/topics/%E6%8C%87%E4%BB%A4%E5%AF%B9%E9%BD%90%E4%B8%8E%20post-training.md)
- [LLM RL](./wiki/topics/LLM%20RL.md)
- [Scaling 与 compute-optimal training](./wiki/topics/Scaling%20与%20compute-optimal%20training.md)
- [注意力机制 Attention](./wiki/topics/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%20Attention.md)
- [Qwen 系列](./wiki/topics/Qwen%20系列.md)
- [GPT-3](./wiki/concepts/GPT-3.md)
- [PaLM](./wiki/concepts/PaLM.md)
- [DeepSeek](./wiki/concepts/DeepSeek.md)
- [Llama 3](./wiki/concepts/Llama%203.md)
- [Vaswani et al. - 2017 - Attention is all you need](./wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)

## [2026-04-15] lint | 三层结构与证据层回正

- 将全部 `raw/summary/` 迁移到 `wiki/summaries/`，明确 summary 属于 wiki 的来源摘要层，而非 raw 原始层
- 合并 `Trans-Encoder`、`PPTAgent`、`DETRs with Collaborative Hybrid Assignments Training` 的重复 `(2)` 来源链，并统一回收引用
- 重写 `AGENTS.md`，回正为 `raw -> wiki -> schema` 三层结构，允许基于现有 `wiki/summaries/` 在 query 阶段沉淀 topic / comparison / timeline
- 清理 concept 页中的证据层与导航层混用，把 topic 链接从 `来源支持` 移回 `关联页面`
- 重写 `作者分析` 与 `机构分析`，去除脏统计与非实体噪声
- 新增 comparison 页 [RLHF vs DPO vs ORPO vs KTO](./wiki/comparisons/RLHF%20vs%20DPO%20vs%20ORPO%20vs%20KTO.md)
- 新增 timeline 页 [Qwen 系列演进](./wiki/timelines/Qwen%20系列演进.md)
- [index](./index.md)

关键变更：

- 将 `LLM 基础脉络` 中仍有价值的骨干叙事并入 `LLM 预训练`
- 在 `LLM 预训练` 中补入“能力形成 vs 行为塑形”的阶段切分，使其可独立承担 LLM 主线入口
- 回收 `指令对齐与 post-training`、`LLM RL` 与相关 concept/topic/summary 中对 `LLM 基础脉络` 的导航依赖
- 从 `index.md` 主导航中移除 `LLM 基础脉络`，并删除该 topic 页面

后续建议：

- 若后续需要更高层的 LLM 总览，应优先做 comparison 或 timeline，而不是恢复一个与子 topic 重叠的总论页
- 在继续扩展 LLM 主线时，把新增内容优先沉淀到 `LLM 预训练`、`指令对齐与 post-training`、`LLM RL` 这三个边界更清晰的页面

## [2026-04-14] lint | 全量扫描 topic / concept 关系并补链

涉及页面：

- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- `LLM 基础脉络`（现已并入 `LLM 预训练`）
- [LLM RL](./wiki/topics/LLM%20RL.md)
- [指令对齐与 post-training](./wiki/topics/%E6%8C%87%E4%BB%A4%E5%AF%B9%E9%BD%90%E4%B8%8E%20post-training.md)
- [传统 CV](./wiki/topics/%E4%BC%A0%E7%BB%9F%20CV.md)
- [传统 NLP](./wiki/topics/%E4%BC%A0%E7%BB%9F%20NLP.md)
- [FLAN](./wiki/concepts/FLAN.md)
- [LoRA](./wiki/concepts/LoRA.md)
- [OPT-IML](./wiki/concepts/OPT-IML.md)
- [Prompt Tuning](./wiki/concepts/Prompt%20Tuning.md)
- [Toolformer](./wiki/concepts/Toolformer.md)
- [Llama Guard](./wiki/concepts/Llama%20Guard.md)
- [CLIP](./wiki/concepts/CLIP.md)
- [Kosmos-2](./wiki/concepts/Kosmos-2.md)
- [Kosmos-2.5](./wiki/concepts/Kosmos-2.5.md)
- [MiniCPM](./wiki/concepts/MiniCPM.md)
- [MiniCPM-V](./wiki/concepts/MiniCPM-V.md)
- [OFA](./wiki/concepts/OFA.md)
- [data2vec](./wiki/concepts/data2vec.md)
- [HuBERT](./wiki/concepts/HuBERT.md)
- [DocLayNet](./wiki/concepts/DocLayNet.md)
- [PubTables-1M](./wiki/concepts/PubTables-1M.md)
- [T5](./wiki/concepts/T5.md)
- [Switch Transformer](./wiki/concepts/Switch%20Transformer.md)
- [OPT](./wiki/concepts/OPT.md)
- [mT5](./wiki/concepts/mT5.md)
- [Gemma 3](./wiki/concepts/Gemma%203.md)

关键变更：

- 对全部 `wiki/topics/` 与 `wiki/concepts/` 执行关系扫描，统计 concept 是否被任何 topic 引用
- 按 concept 页已有 `来源支持 / 关联页面` 反查 topic，优先补齐确定性漏链，而不凭空扩写新结论
- 在 `指令对齐与 post-training`、`LLM RL` 中补入 `FLAN`、`LoRA`、`OPT-IML`、`Prompt Tuning`、`Toolformer`、`Llama Guard` 等后训练相关概念的导航
- 在 `传统 CV` 中补入 `CLIP`、`DocLayNet`、`PubTables-1M`、`Kosmos-2`、`Kosmos-2.5`、`MiniCPM-V`、`OFA`、`data2vec`、`HuBERT`、`Tip-Adapter` 等视觉 / 文档 / 多模态相关概念的导航
- 在 `LLM 预训练` 与当时的 `LLM 基础脉络` 中补入 `OPT`、`T5`、`Switch Transformer`、`mT5`、`Gemma 3`、`MiniCPM`、`PaLM`、`DeepSeek` 等预训练主线相关概念
- 扫描完成后，当前 `0` 个 concept 处于“未被任何 topic 引用”状态

后续建议：

- 后续新增 concept 时，可默认复用本次扫描思路：先看 concept 页的 `来源支持 / 关联页面`，再决定应补哪些 topic
- 若后续新增多模态或语音 topic，可把当前暂挂在 `传统 CV` 下的部分概念再细分迁移，减少总览页负担

## [2026-04-14] ingest | 新增搜索排序 topic

涉及页面：

- [搜索排序](./wiki/topics/%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F.md)
- [ColBERT](./wiki/concepts/ColBERT.md)
- [传统 NLP](./wiki/topics/%E4%BC%A0%E7%BB%9F%20NLP.md)
- [BERT类双向Transformer语言模型](./wiki/topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [Mitra, Craswell - 2019 - A Deep Look into Neural Ranking Models for Information Retrieval](./wiki/summaries/Mitra,%20Craswell%20-%202019%20-%20A%20Deep%20Look%20into%20Neural%20Ranking%20Models%20for%20Information%20Retrieval.md)
- [Nogueira, Cho - 2019 - Passage Re-ranking with BERT](./wiki/summaries/Nogueira,%20Cho%20-%202019%20-%20Passage%20Re-ranking%20with%20BERT.md)
- [Khattab, Zaharia - 2020 - ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](./wiki/summaries/Khattab,%20Zaharia%20-%202020%20-%20ColBERT%20Efficient%20and%20Effective%20Passage%20Search%20via%20Contextualized%20Late%20Interaction%20over%20BERT.md)
- [Nogueira et al. - 2020 - Pretrained Transformers for Text Ranking BERT and Beyond](./wiki/summaries/Nogueira%20et%20al.%20-%202020%20-%20Pretrained%20Transformers%20for%20Text%20Ranking%20BERT%20and%20Beyond.md)
- [index](./index.md)

关键变更：

- 新增 4 篇搜索排序相关原始来源，并补齐对应 `raw/pdf/`、`raw/html/`、`raw/text/` 与精修版 `wiki/summaries/`
- 新建正式 topic《搜索排序》，按 `neural ranking 背景 -> BERT reranking -> dense retrieval -> late interaction` 组织主线
- 新增 `ColBERT` 概念页，承接搜索排序中介于 cross-encoder 与双塔之间的 late interaction 路线
- 更新 `传统 NLP` 与 `BERT类双向Transformer语言模型` 的关联导航，避免新 topic 成为孤立页
- 更新根级 `index.md`，补 summary 总数、新来源记录、topic 导航与 concept 导航

后续建议：

- 若继续补齐搜索排序主线，可优先加入 `RankNet / LambdaMART`、`ANCE`、`monoT5` 与更直接的 listwise reranking 来源
- 若后续 summary 足够，可把“召回 vs 重排 vs late interaction”再拆成独立 comparison 页，避免 topic 过宽

## [2026-04-13] ingest | 新增注意力机制 Attention topic

涉及页面：

- [注意力机制 Attention](./wiki/topics/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%20Attention.md)
- [FlashAttention](./wiki/concepts/FlashAttention.md)
- [Grouped-Query Attention](./wiki/concepts/Grouped-Query%20Attention.md)
- [Transformer](./wiki/concepts/Transformer.md)
- [Vaswani et al. - 2017 - Attention is all you need](./wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)
- [Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need](./wiki/summaries/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md)
- [Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer](./wiki/summaries/Kitaev,%20Kaiser,%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md)
- [Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer](./wiki/summaries/Beltagy,%20Peters,%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md)
- [Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity](./wiki/summaries/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md)
- [Zaheer et al. - 2020 - Big bird Transformers for longer sequences](./wiki/summaries/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md)
- [Choromanski et al. - 2021 - Rethinking Attention with Performers](./wiki/summaries/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md)
- [Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention](./wiki/summaries/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md)
- [Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](./wiki/summaries/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)
- [Ainslie et al. - 2023 - GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](./wiki/summaries/Ainslie%20et%20al.%20-%202023%20-%20GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](./wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [index](./index.md)

关键变更：

- 新增 8 篇 attention 相关原始来源，并补齐对应 `raw/pdf/`、`raw/html/`、`raw/text/` 与精修版 `wiki/summaries/`
- 精修 `Attention Is All You Need`、`BigBird` 与 `DeepSeek-V3` 的 summary，使其可直接支撑 attention topic 中的标准 attention、混合稀疏 attention 与 `MLA` 路线
- 新建正式 topic《注意力机制 Attention》，按“标准全连接 attention -> 长序列稀疏化 -> 线性/低秩近似 -> `KV cache` 优化 -> 实现级优化”组织主线
- 新增 `FlashAttention` 与 `Grouped-Query Attention` 两个 concept 页，分别承接 exact attention 的系统优化路线与 `MHA / MQA / GQA` 的推理解码折中路线
- 更新 `Transformer` 概念页与根级 `index.md`，把 attention topic 接入主导航，并把 summary 总数更新为 `207`

后续建议：

- 若后续继续扩展 attention 主线，可优先补 `cross-attention`、`RoPE / ALiBi`、`Hybrid Attention` 与 `DeepSeek-V2 MLA` 的更直接来源
- 若希望把“长序列 efficient attention”与“解码态 `KV cache` 优化”进一步拆开，可新增比较页，避免当前 topic 继续变得过宽

## [2026-04-13] ingest | 新增目标检测 topic

涉及页面：

- [目标检测](./wiki/topics/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B.md)
- [传统 CV](./wiki/topics/%E4%BC%A0%E7%BB%9F%20CV.md)
- [Faster R-CNN](./wiki/concepts/Faster%20R-CNN.md)
- [DETR](./wiki/concepts/DETR.md)
- [Ren et al. - 2015 - Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks](./wiki/summaries/Ren%20et%20al.%20-%202015%20-%20Faster%20R-CNN%20Towards%20Real-Time%20Object%20Detection%20with%20Region%20Proposal%20Networks.md)
- [index](./index.md)

关键变更：

- 新增 `Faster R-CNN` 原始来源，补齐 `raw/pdf/`、`raw/html/`、`raw/text/` 与精修版 `wiki/summaries/`
- 新建 `目标检测` 正式 topic，按 `proposal-based -> set prediction -> 实时化 -> 混合 assignment` 组织检测主线
- 新增 `Faster R-CNN` 与 `DETR` 两个 concept 页，作为检测 topic 的稳定导航节点
- 更新 `传统 CV`，将目标检测从总览页中的并列分支拆为独立 topic，并保留与视觉总线的关系
- 更新根级 `index.md`，补 summary 计数、新来源记录、topic 导航与 concept 导航

后续建议：

- 继续补 `YOLO`、`RetinaNet`、`Deformable DETR`、`DINO` 等关键来源，避免当前 topic 仍偏向 `Faster R-CNN / DETR` 双主线
- 若后续证据足够，可新增 `two-stage vs one-stage vs end-to-end detection` 比较页

## [2026-04-13] ingest | 扩展目标检测中的 YOLO 全系列

涉及页面：

- [目标检测](./wiki/topics/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B.md)
- [YOLO](./wiki/concepts/YOLO.md)
- [Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection](./wiki/summaries/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md)
- [Redmon, Farhadi - 2016 - YOLO9000 Better Faster Stronger](./wiki/summaries/Redmon,%20Farhadi%20-%202016%20-%20YOLO9000%20Better%20Faster%20Stronger.md)
- [Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement](./wiki/summaries/Redmon,%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md)
- [Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection](./wiki/summaries/Bochkovskiy,%20Wang,%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md)
- [Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection](./wiki/summaries/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md)
- [Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond](./wiki/summaries/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md)
- [Ultralytics - 2026 - Ultralytics YOLO Docs Home](./wiki/summaries/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md)
- [index](./index.md)

关键变更：

- 新增 7 个 `YOLO` 相关来源页，并补齐对应 `raw/pdf/`、`raw/html/`、`raw/text/`
- 新建 `YOLO` 概念页，作为 one-stage 检测家族从 `YOLOv1` 到 `YOLO26` 的总入口
- 重写 `目标检测` 中的主线分层，把 `YOLO` 从旁支对照对象提升为正式 one-stage 主线
- 在 topic 中明确区分：`YOLOv1-v4` 的 classic one-stage 演化、`YOLOv5-v9` 的综述级脉络、`YOLOv10` 的 `NMS-free` 新阶段，以及截至 `2026-04-13` 官方 `YOLO11 / YOLO26` 状态
- 更新根级 `index.md` 的 summary 计数、传统 CV 分组记录与 concept 导航

后续建议：

- 继续补 `YOLOv5`、`YOLOv7`、`YOLOv8`、`YOLOv9` 的原始论文或官方技术说明，减少当前对综述与官方首页的依赖
- 后续可新增 `YOLO vs DETR vs Faster R-CNN` 比较页，专门整理三条检测路线在接口、监督、后处理与部署上的差异

## [2026-04-12] ingest | 扩展 LLM RL 方法谱系

涉及页面：

- [LLM RL](./wiki/topics/LLM%20RL.md)
- [RLHF](./wiki/concepts/RLHF.md)
- [GRPO](./wiki/concepts/GRPO.md)
- [ORPO](./wiki/concepts/ORPO.md)
- [KTO](./wiki/concepts/KTO.md)
- [DAPO](./wiki/concepts/DAPO.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](./wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](./wiki/summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](./wiki/summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](./wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- [index](./index.md)

关键变更：

- 补入 4 篇 `LLM RL` 关键来源：`DeepSeekMath`、`ORPO`、`KTO`、`DAPO`，并同步保存 `raw/pdf/`、`raw/html/`、`raw/text/`
- 新增 `GRPO`、`ORPO`、`KTO`、`DAPO` 四个 concept 页，使 `LLM RL` 不再只停留在 `RLHF / DPO / DeepSeek-R1` 三节点结构
- 重写 `LLM RL` 的方法分层，把主题明确拆成 `经典 RLHF`、`偏好优化分支`、`reasoning-oriented 在线 RL`、`大规模 reasoning RL 工程化`
- 更新 `RLHF` 概念页，补足其与后续 preference optimization 与 reasoning RL 分支的关系
- 更新根级 `index.md`，补 summary 计数、`LLM RL` 主题摘要与新增 concept 导航

后续建议：

- 下一步优先补 `PPO`、reward model、RLAIF、process supervision、online iterative RLHF 等概念页，避免 `LLM RL` 继续把关键中间层折叠掉
- 若后续继续扩展，可新增 `RLHF vs DPO vs ORPO vs KTO` 比较页，以及 `GRPO -> DAPO` 的 reasoning RL 工程演化页

## [2026-04-12] ingest | 扫描 PDF 并导入可用 arXiv HTML

涉及页面：

- [log](./log.md)
- [download_arxiv.py](./scripts/download_arxiv.py)
- [import_arxiv_html_for_pdfs.py](./scripts/import_arxiv_html_for_pdfs.py)
- `raw/html/` 中新增的 arXiv HTML 原件
- `raw/text/` 中由 PDF 抽取版切换为 arXiv HTML 导出版的全文 markdown 页面

关键变更：

- 扫描 `raw/pdf/` 下全部 `175` 个 PDF，优先从文件名和 PDF 前两页提取 arXiv id，少数缺 id 的条目再用标题到 arXiv API 做谨慎补判
- 对命中的 `165` 个 PDF 补齐 `raw/html/*.html`，并用 arXiv HTML 导出的 markdown 覆盖对应 `raw/text/*.md`
- 为 `DeepSeek-V3`、`DeepSeek-R1` 与重复文件 `DETRs with Collaborative Hybrid Assignments Training(2)` 增加确定性映射，避免标题补判漏掉已知 arXiv 来源
- 保持 `raw/pdf/` 原件不变，只迁移全文层的来源优先级，使后续 summary / topic 默认回到 HTML 文本层

未命中条目：

- `Ahead - 2024 - Leopold Aschenbrenner S I T U AT I O N A L AWA R E N E S S The Decade Ahead`
- `Corporation - 2022 - NVIDIA DGX A100 The Universal System for AI Infrastructure`
- `Dean, Scientist, Deepmind - Unknown - Important Trends in AI How Did We Get Here , What Can We Do Now and How Can We Shape AI ’ s Fut`
- `Ding et al. - 2024 - Using the divergent association task to measure divergent thinking in Chinese elementary school students`
- `Dumas, Organisciak, Doherty - 2020 - Measuring Divergent Thinking Originality With Human Raters and Text-Mining Models A Psychometric Co`
- `Migacz - 2017 - Intro`
- `Nvidia - 2022 - Nvidia Ada Gpu Architecture`
- `Pradhan, Moschitti, Uryupina - 2012 - CoNLL-2012 Shared Task Modeling Multilingual Unrestricted Coreference in OntoNotes`
- `StandfordUniversity - 2023 - Artificial Intelligence Index Report Introduction to the AI Index Report 2023 GP-003`
- `Xu et al. - 2016 - Review on knowledge graph techniques`

后续建议：

- 对剩余 `10` 个未命中 PDF 单独核对其是否本来就不是 arXiv 来源；若是其他网页、报告站或会议官网来源，应按普通网页 ingest 流程补 `raw/html/` 与 `raw/text/`
- 后续新增 PDF 时，优先先跑 `scripts/import_arxiv_html_for_pdfs.py`，再考虑回退到 `scripts/extract_pdf_text.py`

## [2026-04-12] lint | 扫描并补齐 html/pdf 原件与全文来源标注

涉及页面：

- [AGENTS](./AGENTS.md)
- [log](./log.md)
- [fetch_web_text.py](./scripts/fetch_web_text.py)
- [extract_pdf_text.py](./scripts/extract_pdf_text.py)
- [download_arxiv.py](./scripts/download_arxiv.py)
- `raw/html/` 中新增的原始 HTML 页面
- `raw/text/` 中批量修正来源头的全文 markdown 页面

关键变更：

- 对全部 `wiki/summaries/*.md -> raw/text/*.md` 链路执行扫描，检查是否存在 `Source PDF / Source HTML`
- 批量修正旧 `raw/text` 中错误的 `Source PDF` 路径
- 为当前 12 个网页 / HTML 来源补齐 `raw/html/*.html` 原件
- 为所有网页来源的 `raw/text/*.md` 补齐 `Source HTML` 头标记
- 确认当前 `187` 篇 summary 全部具备可回溯的 `raw/text`，且其来源头不再缺失

## [2026-04-12] lint | 执行 arXiv HTML / markdown 全文层规则

涉及页面：

- [AGENTS](./AGENTS.md)
- [log](./log.md)
- `raw/text/` 中新增的网页 markdown 全文页
- [Hello Qwen2](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)
- [Introducing Qwen1.5](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Introducing%20Qwen1.5.md)
- [Qwen2-VL](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Qwen2-VL.md)
- [Qwen2.5-LLM Extending the boundary of LLMs](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Qwen2.5-LLM%20Extending%20the%20boundary%20of%20LLMs.md)
- [Qwen2.5-Omni See Hear Talk Write Do It All](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md)
- [Qwen3 Think Deeper Act Faster](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
- [Qwen3.5 Towards Native Multimodal Agents](./wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)
- [Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](./wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)
- [Qwen-Image Crafting with Native Text Rendering](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)
- [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](./wiki/summaries/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)
- [Stability AI - 2022 - Stable Diffusion Launch Announcement](./wiki/summaries/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)
- [Black Forest Labs - 2026 - FLUX.2 Overview](./wiki/summaries/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md)
- [fetch_web_text.py](./scripts/fetch_web_text.py)

关键变更：

- 将 ingest 规则落到项目层：网页来源也统一补齐 `raw/text/*.md` markdown 全文层
- 新增 `scripts/fetch_web_text.py`，支持 `HTML -> markdown`
- 对当前缺少 `全文文本` 的来源页完成审计与补链
- 对 `qwen.ai` 动态页面使用浏览器渲染提取，避免规则只覆盖静态 HTML
- 修复 `qwenlm.github.io/blog/*` 跳转页只抓到 redirect 提示的问题，改为优先抽取页面内 `articleBody`
- 补入 `uv + Python 3.12` 项目配置，并为 HTML 来源恢复 `raw/html/` 原件保存层

后续建议：

- 后续 ingest 对网页来源先补 `raw/text`，再写 `wiki/summaries`
- 若动态站点继续增多，可把 `playwright` 流程进一步脚本化

## [2026-04-12] ingest | 接入 diffusion 文生图主线

涉及页面：

- [扩散模型与文生图](./wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Stable Diffusion](./wiki/concepts/Stable%20Diffusion.md)
- [FLUX.2](./wiki/concepts/FLUX.2.md)
- [Qwen-Image](./wiki/concepts/Qwen-Image.md)
- [Qwen 系列](./wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md)
- [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](./wiki/summaries/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)
- [Stability AI - 2022 - Stable Diffusion Launch Announcement](./wiki/summaries/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)
- [Black Forest Labs - 2026 - FLUX.2 Overview](./wiki/summaries/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md)
- [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)
- [index](./index.md)

关键变更：

- 新增 `扩散模型与文生图` 正式 topic，补齐从 latent diffusion 到生产级文生图控制的研究主线
- 新增 `Stable Diffusion`、`FLUX.2`、`Qwen-Image` 三个 concept 页
- 新增 4 个 summary 页，分别承接 latent diffusion 论文、Stable Diffusion 开放发布、FLUX.2 官方总览与 Qwen-Image 官方发布
- 同步更新 `Qwen 系列`，将 `Qwen-Image` 纳入家族分支脉络
- 更新根级 `index.md` 导航与 summary 计数

后续建议：

- 继续补 `SDXL`、`ControlNet`、`DALL·E`、`Imagen` 等关键节点，完善文生图时间线与比较层
- 若后续获取 `Qwen-Image` 技术报告或 `FLUX.2` 研究论文，应回写 topic 中关于架构与训练路线的不确定点

## [2026-04-12] bootstrap | 初始化最小骨架

涉及页面：

- [AGENTS](./AGENTS.md)
- [index](./index.md)

关键变更：

- 固定基础结构：`raw/pdf/`、`wiki/`、`AGENTS.md`
- 初始化首批 `wiki/summaries/` 与 `wiki/topics/` 目录
- 建立索引与操作日志入口

后续建议：

- 选择一个 `raw/pdf/` 来源做首个真实 ingest
- 从 ingest 结果反向微调模板和命名约定

## [2026-04-12] ingest | 首轮 LLM 基础脉络铺开

涉及页面：

- `LLM 基础脉络`（该页后续已并入 `LLM 预训练`）
- [Scaling 与 compute-optimal training](./wiki/topics/Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./wiki/topics/指令对齐与%20post-training.md)
- [GPT-3](./wiki/concepts/GPT-3.md)
- [Llama 3](./wiki/concepts/Llama%203.md)
- [Brown et al. - 2020 - Language models are few-shot learners](./wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](./wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](./wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [index](./index.md)

关键变更：

- 建立 3 个主题页，形成 LLM 基础脉络、compute-optimal 训练、指令对齐三段主线
- 建立 3 个summary 页，作为 Brown 2020、Hoffmann 2022、Ouyang 2022 的首轮证据基座
- 建立 `GPT-3` 与 `Llama 3` 两个实体页，其中 `Llama 3` 暂作为后续 wave 的轻量入口
- 将根级 `index.md` 从空占位升级为真实导航页

后续建议：

- 继续 ingest `PaLM`、`Qwen`、`Llama 3`、`DPO` 等代表性来源，补齐基础脉络
- 视页面增长情况决定是否建立 `Chinchilla`、`InstructGPT` 独立实体页
- 在来源覆盖更充分后，再沉淀 `comparisons/` 或 `timelines/` 页面

## [2026-04-12] ingest | 批量抽取 PDF 全文并重写主题页

涉及页面：

- [AGENTS](./AGENTS.md)
- `LLM 基础脉络`（该页后续已并入 `LLM 预训练`）
- [Scaling 与 compute-optimal training](./wiki/topics/Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./wiki/topics/指令对齐与%20post-training.md)

关键变更：

- 为 `raw/pdf/` 下全部 PDF 批量生成 `raw/text/*.md` 全文文本文件
- 新增 `scripts/extract_pdf_text.py` 作为可重复执行的全文抽取脚本
- 将现有 3 个主题页从摘要层重写为基于正文可支持信息的主线页

后续建议：

- 用 `raw/text/` 中的全文内容重写首轮 3 个summary 页，使其不再只依赖摘要
- 继续把 `PaLM`、`Qwen`、`Llama 3`、`DPO` 等已可读全文接入 `wiki/summaries/`
- 若后续页数明显增长，可为 `raw/text/` 增加索引或批量校验工具

## [2026-04-12] ingest | 重组主题为五大类

涉及页面：

- [传统 NLP](./wiki/topics/传统%20NLP.md)
- [传统 CV](./wiki/topics/传统%20CV.md)
- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- [LLM RL](./wiki/topics/LLM%20RL.md)
- [Slide相关](./wiki/topics/Slide相关.md)
- [index](./index.md)

关键变更：

- 新增 5 个总主题页，按 `传统NLP / 传统CV / LLM预训练 / LLM RL / Slide相关` 重组主题层
- 将现有 LLM 专题页保留为工作中的子专题页，并在新总主题页中建立挂接关系
- 更新根级 `index.md`，使主题导航优先按这 5 个大类展开

后续建议：

- 逐步为 `传统NLP`、`传统CV`、`Slide相关` 各接入首批正式summary 页
- 将 `PaLM`、`Qwen`、`Llama 3` 归入 `LLM预训练`，将 `DPO` 等归入 `LLM RL`

## [2026-04-12] ingest | 接入 DeepSeek-V3 与 DeepSeek-R1

涉及页面：

- [Unknown - 2024 - DeepSeek-V3 Technical Report](./wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](./wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek](./wiki/concepts/DeepSeek.md)
- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- [LLM RL](./wiki/topics/LLM%20RL.md)
- [index](./index.md)

关键变更：

- 将 `DeepSeek-V3` 正式接入 `LLM预训练`
- 将 `DeepSeek-R1` 正式接入 `LLM RL`
- 新增 `DeepSeek` 实体页，连接预训练与推理强化学习两条主线

后续建议：

- 继续接入 `Qwen`、`PaLM`、`DPO` 等来源，补齐与 DeepSeek 的对照关系
- 视后续页数决定是否把 `DeepSeek-V3` 和 `DeepSeek-R1` 拆成独立专题页

## [2026-04-12] ingest | 重建全部 summary

涉及页面：

- [index](./index.md)
- `wiki/summaries/` 下全部summary 页

关键变更：

- 删除旧的 5 个试验版summary 页
- 按 `raw/pdf/` 与 `raw/text/` 全量重建 172 个统一格式的summary 页
- `Summary` 索引改为按候选主题分组展示

后续建议：

- 优先精修主干来源，而不是一次性精修全部 172 篇
- 对 `LLM预训练 / LLM RL / Slide相关` 继续补更多高价值summary 页和实体页

## [2026-04-12] ingest | 对齐 knowledge-base 目录结构

涉及页面：

- [AGENTS](./AGENTS.md)
- [index](./index.md)
- [log](./log.md)
- `raw/pdf/`
- `raw/text/`
- `raw/assets/`
- `wiki/summaries/`
- `wiki/topics/`
- `wiki/concepts/`
- `wiki/authors/`
- `wiki/comparisons/`
- `wiki/timelines/`

关键变更：

- 将原始 PDF 统一收敛到 `raw/pdf/`
- 将 PDF 全文文本统一收敛到 `raw/text/`
- 将总索引与操作日志固定在根级 `index.md` 与 `log.md`
- 将 `wiki/` 目录固定为 `sources / topics / concepts / authors / comparisons / timelines`
- 重新生成全部 `wiki/summaries/` 页面，使其引用新的 `raw/pdf/` 与 `raw/text/` 路径

后续建议：

- 后续新增 PDF 时先放入 `raw/pdf/`，再运行全文抽取与 ingest 脚本
- 作者、比较、时间线三类目录当前仍为空，可在下一轮按主题逐步补齐

## [2026-04-12] ingest | 作者与机构分析初始化

涉及页面：

- [作者分析](./wiki/authors/作者分析.md)
- [机构分析](./wiki/authors/机构分析.md)
- [index](./index.md)

关键变更：

- 新增 `scripts/rebuild_author_analysis.py`，从 `raw/text/` 首页文本启发式抽取作者与机构
- 在 `wiki/authors/` 下生成 `作者分析` 与 `机构分析` 两个汇总页
- 将根级 `index.md` 的 `Authors` 区从空占位改成实际入口

后续建议：

- 对高价值来源做人工校正，逐步把启发式抽取结果沉淀为更稳定的作者与机构页
- 后续可继续做机构归并，例如统一 `Google Research` / `Google DeepMind` / `Google`

## [2026-04-12] ingest | 完善 concept 层

涉及页面：

- [GPT-3](./wiki/concepts/GPT-3.md)
- [PaLM](./wiki/concepts/PaLM.md)
- [Chinchilla](./wiki/concepts/Chinchilla.md)
- [Qwen](./wiki/concepts/Qwen.md)
- [Llama 3](./wiki/concepts/Llama%203.md)
- [DeepSeek](./wiki/concepts/DeepSeek.md)
- [DeepSeek-V3](./wiki/concepts/DeepSeek-V3.md)
- [DeepSeek-R1](./wiki/concepts/DeepSeek-R1.md)
- [InstructGPT](./wiki/concepts/InstructGPT.md)
- [DPO](./wiki/concepts/DPO.md)
- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- [LLM RL](./wiki/topics/LLM%20RL.md)
- [index](./index.md)

关键变更：

- 将原本较轻的 concept 层扩成模型家族与方法概念并行的结构
- 为预训练主线补入 `PaLM`、`Chinchilla`、`Qwen`、`Llama 3`、`DeepSeek-V3`
- 为对齐主线补入 `InstructGPT`、`DPO`、`DeepSeek-R1`
- 同步更新 `LLM预训练`、`LLM RL` 和根级 `index.md` 的链接入口

后续建议：

- 下一步可继续为 `comparison` 层建立如 `GPT-3 vs Chinchilla`、`RLHF vs DPO` 之类的横向页
- 若后续继续精读来源，应优先把当前 concept 页中的保守表述升级成更细的结构化事实

## [2026-04-12] ingest | 扩展通用方法概念

涉及页面：

- [RLHF](./wiki/concepts/RLHF.md)
- [Instruction Tuning](./wiki/concepts/Instruction%20Tuning.md)
- [LoRA](./wiki/concepts/LoRA.md)
- [MoE](./wiki/concepts/MoE.md)
- [LLM 预训练](./wiki/topics/LLM%20预训练.md)
- [LLM RL](./wiki/topics/LLM%20RL.md)
- [index](./index.md)

关键变更：

- 新增 `RLHF`、`Instruction Tuning`、`LoRA`、`MoE` 四个通用方法与架构概念页
- 将概念层从“以模型家族为主”扩展到“模型概念 + 方法概念 + 架构概念”并行
- 在 `LLM预训练` 与 `LLM RL` 两个主题页中补入这些新概念的入口
- 更新根级 `index.md`，让 Concepts 区覆盖更完整的主干术语

后续建议：

- 下一步可把 `RLHF vs DPO`、`dense vs MoE`、`full fine-tuning vs LoRA` 沉淀成比较页
- 若后续继续补概念，可优先接入 `reward model`、`PPO`、`RLAIF`、`scaling laws`

## [2026-04-12] ingest | 批量扩展 concept 到 50 个

涉及页面：

- `wiki/concepts/` 下新增 36 个 concept 页
- [index](./index.md)
- [log](./log.md)

关键变更：

- 在原有 14 个 concept 基础上，新增 36 个模型、方法、架构、数据集概念页
- 将 concept 层覆盖范围从 LLM 主干扩展到传统 NLP、文档理解、多模态、视觉与检索
- 将根级 `index.md` 的 `Concepts` 区扩展到 50 个条目

后续建议：

- 下一步可按 `预训练 / 对齐 / 多模态 / 检索 / 文档理解` 为 concept 层建立比较页或时间线页
- 若继续扩 concept，优先补 `RAG`、`PPO`、`reward model`、`Nougat`、`DocTR`、`LayoutReader`

## [2026-04-12] ingest | 升级 topic 专业版规范

涉及页面：

- [AGENTS](./AGENTS.md)
- [log](./log.md)

关键变更：

- 将 `wiki/topics/` 从最小结构要求升级为研究综述型强规范
- 为 topic 引入 `正式 topic / 待建设 topic` 两种成熟度状态
- 明确 topic 的核心论断必须优先回溯到 `wiki/summaries/`，并区分证据层与导航层
- 在 `Ingest / Query / Lint / index.md` 规则中补入 topic 专项约束

后续建议：

- 下一步按新规范审视现有 `wiki/topics/`，将弱占位页降级或重写
- 优先重写 `传统NLP`、`传统CV`、`Slide相关` 这类 summary 支撑不足的 topic

## [2026-04-12] ingest | 按专业规范重写全部 topic

涉及页面：

- `wiki/topics/` 下全部 8 个 topic 页
- [index](./index.md)
- [log](./log.md)

关键变更：

- 将全部 topic 页升级为正式 topic 结构，补入主题定义、核心问题、主线脉络、关键争论、证据基础、代表页面与未解决问题
- 为 `传统NLP`、`传统CV`、`Slide相关` 补入可追溯的 summary 基座，不再保留占位式写法
- 将根级 `index.md` 的 Topics 区摘要更新为研究综述型入口表述

后续建议：

- 下一步优先建立 `RLHF vs DPO`、`dense vs MoE`、`文档理解 vs slide 理解` 等 comparison 页
- 随着更多 summary 精修，可继续把 topic 中的保守主张升级为更细的结构化比较

## [2026-04-12] ingest | 扩展 Qwen 家族到 3.5 并建立家族 topic

涉及页面：

- `wiki/summaries/` 下新增 8 个 Qwen 家族来源页
- `wiki/concepts/` 下新增 7 个 Qwen 相关 concept 页，并更新 [Qwen](./wiki/concepts/Qwen.md) 与 [Qwen2.5-VL](./wiki/concepts/Qwen2.5-VL.md)
- [Qwen 系列](./wiki/topics/Qwen%20系列.md)
- [index](./index.md)
- [log](./log.md)

关键变更：

- 补齐 Qwen1.5、Qwen2、Qwen2.5、Qwen2-VL、Qwen2.5-Omni、Qwen3，以及截至 `2026-04-12` 可确认的 Qwen3.5、Qwen3.5-Omni 来源页
- 将原先较抽象的 [Qwen](./wiki/concepts/Qwen.md) 总页扩展为“家族总入口”，并拆出代际与模态节点
- 新建正式 topic [Qwen 系列](./wiki/topics/Qwen%20系列.md)，明确 LLM 主线、VL 分支、Omni 分支与 native multimodal agent 转向
- 更新根级 [index](./index.md)，让 Summary / Topics / Concepts 三层都能直接导航到 Qwen 家族

后续建议：

- 若后续补到 Qwen3 / Qwen3.5 的完整技术报告，可进一步把当前依赖官方摘要的判断升级为更强证据链
- 下一步可单独建立 `Qwen vs Llama vs DeepSeek` comparison 页，承接开放模型家族横向对照

## [2026-04-12] ingest | 补齐 Llama 家族主干论文

涉及页面：

- `raw/pdf/` 下新增 3 篇 Llama 家族论文 PDF
- `raw/text/` 下新增 3 篇对应全文文本
- `wiki/summaries/` 下新增 3 个 Llama 家族来源页
- `wiki/concepts/` 下新增 [Llama](./wiki/concepts/Llama.md)、[LLaMA](./wiki/concepts/LLaMA.md)、[Llama 2](./wiki/concepts/Llama%202.md)、[Code Llama](./wiki/concepts/Code%20Llama.md)
- 更新 [Llama 3](./wiki/concepts/Llama%203.md)
- 更新 [LLM预训练](./wiki/topics/LLM%E9%A2%84%E8%AE%AD.md)
- 更新 [index](./index.md)

关键变更：

- 补入 `LLaMA`、`Llama 2`、`Code Llama` 三篇主干论文，使 Llama 线不再只有 `Llama 3`
- 将 Llama 家族组织为连续代际与代码分支，而不再只保留单个 `Llama 3` 概念入口
- 在 `LLM预训练` 中补入 LLaMA/Llama 2/Code Llama 证据链，使开放模型家族主线更完整

后续建议：

- 若后续继续补 Llama，可优先接入 `Llama 3.1/3.2`、`Llama Guard 2/3`、以及独立的 multimodal/agentic 技术报告
- 可进一步建立 `Llama vs Qwen vs DeepSeek` 比较页，承接开放模型家族横向比较

## [2026-04-12] ingest | 深化 Qwen 系列 topic 与关键来源页

涉及页面：

- 重建 `raw/text/` 下 [Bai et al. - 2023 - Qwen Technical Report](./raw/text/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md) 与 [Bai et al. - 2025 - Qwen2.5-VL Technical Report](./raw/text/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md) 的 PDF 文本层
- 更新 [Bai et al. - 2023 - Qwen Technical Report](./wiki/summaries/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
- 更新 [Bai et al. - 2025 - Qwen2.5-VL Technical Report](./wiki/summaries/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md)
- 更新正式 topic [Qwen 系列](./wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md)

关键变更：

- 将原先损坏的 ar5iv 文本层回退为本地 PDF 抽取版，恢复 Qwen1 与 Qwen2.5-VL 的可读全文基座
- 把两个 “待精读” summary 精修为可直接支撑 topic 的来源页，补入 tokenizer、长上下文外推、RLHF、native dynamic resolution、文档 omni-parsing、GUI agent 数据等关键事实
- 重写 [Qwen 系列](./wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md) 的主线结构，明确区分 LLM 主线、VL 感知路线、Omni 路线、图像生成支线，以及 Qwen3/3.5 的 agent 化转向
- 在 topic 中加入编者归纳层，显式区分来源直接支持的事实与基于多篇 summary 得出的结构性判断

后续建议：

- 下一步优先补 `Qwen3` 与 `Qwen3.5` 的独立 technical report / 正式 summary，以进一步收紧当前 agent 相关结论的证据链
- 可单独建立 `Qwen vs Llama vs DeepSeek` comparison 页，承接开放模型家族在多语言、长上下文、agent 与多模态上的横向对照

## [2026-04-12] query | Slide相关改为 Slide 理解与生产

涉及页面：

- 新增正式 topic [Slide 理解与生产](./wiki/topics/Slide%20理解与生产.md)
- 保留兼容入口 [Slide相关](./wiki/topics/Slide相关.md)
- 更新 [传统 CV](./wiki/topics/传统%20CV.md)
- 更新 [index](./index.md)

关键变更：

- 将原 `Slide相关` 正式 topic 重构为 `Slide 理解与生产`，明确主题边界从“泛 slide 相关”收紧到“理解、评测与生成”三条主线
- 在新 topic 中补入底层文档能力、多模态 slide 理解、设计缺陷评测、编辑式生成与跨页 coherence 的结构化分层
- 用 `LayoutLMv3`、`DocLLM`、`OmniDocBench`、`Lee et al. 2022`、`SlideAudit`、`PPTAgent` 组成更完整的 evidence base
- 将索引与交叉链接切换到新名称，同时保留旧页作为兼容入口，避免历史链接与旧 summary 引用断裂

后续建议：

- 可继续补 `slide understanding vs document understanding`、`text-to-slides vs edit-based generation`、`automatic slide evaluation vs human review` 三类 comparison 页
- 若后续接入更多商业汇报或学术汇报来源，可进一步把教育课件与通用 presentation 的边界写得更稳

## [2026-04-12] query | Slide 理解与生产改为 Slide  理解与生成

涉及页面：

- 新增正式 topic [Slide 理解与生成](./wiki/topics/Slide%20理解与生成.md)
- 保留兼容入口 [Slide 理解与生产](./wiki/topics/Slide%20理解与生产.md) 与 [Slide相关](./wiki/topics/Slide相关.md)
- 更新 [传统 CV](./wiki/topics/传统%20CV.md)
- 更新 [index](./index.md)

关键变更：

- 按字面名称将正式 topic 从 `Slide 理解与生产` 调整为 `Slide  理解与生成`
- 同步将主题表述从“生产”统一收紧为“生成”，以保持标题、正文与方法分层一致
- 保留两层兼容入口，避免新旧 topic 名称之间的历史链接断裂

## [2026-04-12] query | 新建类 BERT 双向 Transformer 语言模型 topic

涉及页面：

- 新增正式 topic [BERT类双向Transformer语言模型](./wiki/topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- 更新 [传统 NLP](./wiki/topics/传统%20NLP.md)
- 更新 [BERT](./wiki/concepts/BERT.md)
- 更新 [RoBERTa](./wiki/concepts/RoBERTa.md)
- 更新 [SpanBERT](./wiki/concepts/SpanBERT.md)
- 更新 [Sentence-BERT](./wiki/concepts/Sentence-BERT.md)
- 更新 [SimCSE](./wiki/concepts/SimCSE.md)
- 更新 [XLM-R](./wiki/concepts/XLM-R.md)
- 更新 [index](./index.md)

关键变更：

- 新建正式 topic，把当前知识库中分散的 `BERT`、`RoBERTa`、`SpanBERT`、`Sentence-BERT`、`SimCSE`、`XLM-R` 等节点收束成一个可导航的双向编码器主题页
- 在正文中明确区分原始 MLM 编码器、训练配方优化、结构感知预训练、句向量与检索、多语言扩展五条稳定主线，而不是机械罗列几十个变体名
- 显式标出当前知识库尚未补齐 `ALBERT`、`ELECTRA`、`DeBERTa`、`MPNet`、`DistilBERT` 等常见变体的 summary/概念页，避免把不完整覆盖写成完整家谱
- 将新 topic 接入 `index.md` 与相关 concept/topic 页的交叉链接，避免新增页面成为孤立入口

后续建议：

- 可继续按同一主题补入 `ALBERT`、`ELECTRA`、`DeBERTa`、`MPNet`、`DistilBERT` 的 summary 与 concept，逐步把当前“代表性分层”升级成更完整家族图谱
- 若后续需要回答“哪类 BERT 适合 embedding / reranking / multilingual / extractive QA”，可单独再建 comparison 页承接

## [2026-04-13] query | 对客服问题做一个 topic，加入很多篇论文分析

涉及页面：

- 新增正式 topic [客服问题](./wiki/topics/%E5%AE%A2%E6%9C%8D%E9%97%AE%E9%A2%98.md)
- 更新 [index](./index.md)

关键变更：

- 新建“客服问题”正式 topic，把知识库中分散的 FAQ 检索、多轮对话、Dense Retrieval、instruction tuning、RLHF、DPO、护栏模型与工具调用来源收束成一条客服系统主线
- 在 topic 中显式区分 FAQ 检索、多轮上下文建模、知识增强生成、行为对齐、工具调用与安全门控六层，而不是把客服问题简化成“聊天机器人”
- 将 `Sakata 2019`、`Vlasov 2019`、`Karpukhin 2020`、`Oğuz 2021`、`Wei 2021`、`Iyer 2022`、`Ouyang 2022`、`Liang 2022`、`Rafailov 2023`、`Schick 2023`、`Inan 2023`、`PIKE-RAG` 逐篇写入主线分析与证据基础
- 明确标出当前 topic 的证据边界：现有来源足以支撑“AI 客服系统分层”这一正式结构，但对工单闭环、人工转接策略和业务指标仍缺少更直接 summary 支撑

## [2026-04-13] query | 主要是和 AI 有关的智能问答、智能客服

涉及页面：

- 新增正式 topic [AI 智能问答与智能客服](./wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)
- 更新 [index](./index.md)

关键变更：

- 将原“客服问题”正式 topic 收紧为更准确的 `AI 智能问答与智能客服`，明确本页主轴是问答系统、客服机器人与 LLM support assistant，而不是泛客服管理问题
- 在正式页中把问题表述统一调整为智能问答 / 智能客服口径，使 FAQ 检索、RAG、指令微调、偏好优化、工具调用与安全护栏都围绕 AI assistant 主线展开

## [2026-04-13] query | 清理旧入口，遵守 AGENTS 新要求

涉及页面：

- 删除旧入口 `wiki/topics/客服问题.md`
- 保留正式 topic [AI 智能问答与智能客服](./wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)

关键变更：

- 按新要求清理旧入口文件，不再保留兼容跳转页
- 仓库中关于该主题的唯一正式入口改为 `AI 智能问答与智能客服`

## [2026-04-13] ingest | AI 智能问答与智能客服补 raw summary 基座

涉及页面：

- 更新 `wiki/summaries/Sakata et al. - 2019 - FAQ retrieval using query-question similarity and BERT-based query-answer relevance.md`
- 更新 `wiki/summaries/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.md`
- 更新 `wiki/summaries/Oğuz et al. - 2021 - Domain-matched Pre-training Tasks for Dense Retrieval.md`
- 更新 `wiki/summaries/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.md`
- 更新 `wiki/summaries/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.md`
- 更新 `wiki/summaries/Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations.md`

关键变更：

- 将一批原本仍停留在批量抽取状态的来源页升级为可复用的结构化 summary
- 使 `AI 智能问答与智能客服` topic 的 FAQ、检索、对齐、工具调用与安全护栏层都能明确回溯到 `wiki/summaries`
- 统一把这些来源的关联页面接回该 topic，而不是继续停留在不准确的自动归类结果

## [2026-04-13] query | 将新增 topic 必须先保留原始文件写入 AGENTS

涉及页面：

- 更新 [AGENTS](./AGENTS.md)

关键变更：

- 在 ingest 的 topic 补充规则中明确：新增 topic 若使用新材料，必须先确认 `raw/html` 或 `raw/pdf` 中已保留原始文件
- 若原始 HTML / PDF 缺失，先下载原始文件，再生成 `raw/text` 与 `wiki/summaries`，最后再纳入正式 topic

## [2026-04-15] ingest | 补入 Qwen-Image-Layered 与 AlphaVAE

涉及页面：

- 新增 `raw/pdf/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.pdf`
- 新增 `raw/html/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.html`
- 新增 `raw/text/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.md`
- 新增 `wiki/summaries/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.md`
- 新增 `raw/pdf/Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning.pdf`
- 新增 `raw/html/Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning.html`
- 新增 `raw/text/Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning.md`
- 新增 `wiki/summaries/Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning.md`
- 新增 `wiki/concepts/Qwen-Image-Layered.md`
- 新增 `wiki/concepts/AlphaVAE.md`
- 新增 `wiki/concepts/RGBA 图层图像.md`
- 更新 `wiki/concepts/Qwen-Image.md`
- 更新 `wiki/topics/扩散模型与文生图.md`
- 更新 `wiki/topics/Qwen 系列.md`
- 更新 `index.md`

关键变更：

- 按 `raw/pdf or raw/html -> raw/text -> wiki/summaries` 顺序补齐两篇新来源的原始层与全文层
- 将 `Qwen-Image-Layered` 组织为 Qwen 图像生成支线向 `RGBA` 图层分解扩展的节点
- 将 `AlphaVAE` 组织为透明图像与 `RGBA` 潜表示学习的基础设施来源
- 新增 `RGBA 图层图像` 概念页，把透明图像表征、图层分解与可编辑生成写成可复用中间层
- 回写 `扩散模型与文生图` 与 `Qwen 系列`，使新来源不只停留在单篇 summary

## [2026-04-15] query | 将 authors 目录收敛为可求证的全名作者与机构页面

涉及页面：

- 删除一批 `et al.` / 姓氏聚合式作者页
- 新增多个可求证全名作者页与机构页面
- 更新 [index](./index.md)

关键变更：

- 不再保留 `Wang et al.`、`Li et al.` 这类无法映射到单一真实作者的伪作者页
- 将可稳定求证的人名改为全名作者页，例如 `Junyang Lin`、`Jingren Zhou`、`Hugo Touvron`、`Joseph Redmon`、`Ali Farhadi`
- 表格结构识别相关作者页拆成 `Brandon Smock`、`Rohith Pesala`、`Robin Abraham` 三个独立作者页
- 同步更新 `index.md` 的 Authors 区，使根级导航直接指向这些可求证的真实作者与机构页面

## [2026-04-18] ingest | 补入 Seedance 2.0

涉及页面：

- 新增 `raw/pdf/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.pdf`
- 新增 `raw/html/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.html`
- 新增 `raw/text/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.md`
- 新增 `wiki/summaries/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.md`
- 新增 `wiki/concepts/Seedance 2.0.md`
- 新增 `wiki/authors/ByteDance Seed.md`
- 新增 `wiki/topics/视频生成.md`
- 更新 `index.md`

关键变更：

- 按 `raw/pdf + raw/html -> raw/text -> wiki/summaries` 顺序把 `Seedance 2.0` 接入知识库
- 将其组织为视频生成从单轮短片走向多模态参考、编辑、续写与音视频联合生成的代表节点
- 新增 `视频生成` 待建设 topic，避免把视频模型错误挂到图像生成主线下
- 新增 `ByteDance Seed` 机构页，使该来源不只停留在单篇 summary

## [2026-04-18] ingest | 补入 Sora Veo Kling Wan Vidu 视频生成主线

涉及页面：

- 新增 `raw/html/OpenAI - 2025 - Sora 2 is here.html`
- 新增 `raw/text/OpenAI - 2025 - Sora 2 is here.md`
- 新增 `raw/html/Google DeepMind - 2026 - Veo.html`
- 新增 `raw/text/Google DeepMind - 2026 - Veo.md`
- 新增 `raw/html/Kuaishou Technology - 2026 - Kling VIDEO 3.0 Omni Model User Guide.html`
- 新增 `raw/text/Kuaishou Technology - 2026 - Kling VIDEO 3.0 Omni Model User Guide.md`
- 新增 `raw/html/Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos.html`
- 新增 `raw/text/Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos.md`
- 新增 `raw/html/Vidu - 2026 - Pricing.html`
- 新增 `raw/text/Vidu - 2026 - Pricing.md`
- 新增 `wiki/summaries/OpenAI - 2025 - Sora 2 is here.md`
- 新增 `wiki/summaries/Google DeepMind - 2026 - Veo.md`
- 新增 `wiki/summaries/Kuaishou Technology - 2026 - Kling VIDEO 3.0 Omni Model User Guide.md`
- 新增 `wiki/summaries/Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos.md`
- 新增 `wiki/summaries/Vidu - 2026 - Pricing.md`
- 新增 `wiki/concepts/Sora 2.md`
- 新增 `wiki/concepts/Veo 3.1.md`
- 新增 `wiki/concepts/Kling VIDEO 3.0 Omni.md`
- 新增 `wiki/concepts/Wan2.6.md`
- 新增 `wiki/concepts/Vidu Q2-Pro.md`
- 新增 `wiki/authors/Kuaishou Technology.md`
- 新增 `wiki/authors/Alibaba Group.md`
- 新增 `wiki/authors/ShengShu Technology.md`
- 更新 `wiki/topics/视频生成.md`
- 更新 `index.md`

关键变更：

- 为 `Sora / Veo / Kling / Wan / Vidu` 分别补入官方来源层与结构化 summary
- 将这些路线组织成独立 concept，而不是只在 topic 中口头提及
- 把 `视频生成` 从待建设骨架升级为正式 topic，明确 world simulation、reference-to-video、导演式控制与 API 工作流几条主线
- 为 `Kuaishou Technology`、`Alibaba Group`、`ShengShu Technology` 新建机构页，补齐视频生成方向的导航层

## [2026-04-18] query | 图像分层 layered

涉及页面：

- 新增 `raw/pdf/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.pdf`
- 新增 `raw/html/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.html`
- 新增 `raw/text/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.md`
- 新增 `raw/html/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.html`
- 新增 `raw/text/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.md`
- 新增 `raw/pdf/Khan et al. - 2026 - Step-by-step Layered Design Generation.pdf`
- 新增 `raw/html/Khan et al. - 2026 - Step-by-step Layered Design Generation.html`
- 新增 `raw/text/Khan et al. - 2026 - Step-by-step Layered Design Generation.md`
- 新增 `raw/pdf/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.pdf`
- 新增 `raw/html/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.html`
- 新增 `raw/text/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.md`
- 新增 `wiki/summaries/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.md`
- 新增 `wiki/summaries/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.md`
- 新增 `wiki/summaries/Khan et al. - 2026 - Step-by-step Layered Design Generation.md`
- 新增 `wiki/summaries/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.md`
- 新增 `wiki/topics/图像分层 layered.md`
- 更新 `wiki/topics/扩散模型与文生图.md`
- 更新 `wiki/topics/Qwen 系列.md`
- 更新 `wiki/concepts/RGBA 图层图像.md`
- 更新 `index.md`

关键变更：

- 围绕用户提出的 `Qwen-Image-Layered + Adobe papers` 查询，补齐 Adobe 相关一手来源的 raw 层与全文层
- 新增正式 topic `图像分层 layered`，把 `RGBA` 图层、`LayerDecomp`、`IntrinsicDiffusion`、`SLEDGE`、`Illustrator's Depth` 与 `Qwen-Image-Layered` 放进同一条可追溯主线
- 明确区分 layered 主题中的四类问题：`RGBA` 多层表示、保留视觉效果的前景/背景分解、intrinsic layers、设计/矢量工作流中的层级更新与排序
- 回写 `扩散模型与文生图`、`Qwen 系列` 与 `RGBA 图层图像`，避免 layered 相关结论只停留在新 topic 单页中

## [2026-04-18] query | 按最新 AGENTS 要求重整 图像分层 layered

涉及页面：

- 更新 `wiki/topics/图像分层 layered.md`
- 更新 `log.md`

关键变更：

- 按当前 `AGENTS.md` 的正式 topic 模板重写正文，使其更接近 paper 级综述而不是 extended notes
- 强化 `主题定义` 的边界说明，明确 layered 不是单一任务名，而是围绕可编辑中间表示形成的问题域
- 重写 `主线脉络 / 方法分层`，改为按“层对象”而不是按论文名简单并列
- 在 `关键争论与分歧` 中补充争议来源与成立条件，减少无约束判断
- 在 `未解决问题` 中明确仓库当前证据缺口与后续应补 comparison / timeline 的位置

## [2026-04-18] query | 统一其他 topic 的高亮写法

涉及页面：

- 更新 `wiki/topics/AI 智能问答与智能客服.md`
- 更新 `wiki/topics/BERT类双向Transformer语言模型.md`
- 更新 `wiki/topics/LLM RL.md`
- 更新 `wiki/topics/LLM 预训练.md`
- 更新 `wiki/topics/Qwen 系列.md`
- 更新 `wiki/topics/Scaling 与 compute-optimal training.md`
- 更新 `wiki/topics/Slide 理解与生成.md`
- 更新 `wiki/topics/传统 CV.md`
- 更新 `wiki/topics/传统 NLP.md`
- 更新 `wiki/topics/扩散模型与文生图.md`
- 更新 `wiki/topics/指令对齐与 post-training.md`
- 更新 `wiki/topics/搜索排序.md`
- 更新 `wiki/topics/注意力机制 Attention.md`
- 更新 `wiki/topics/目标检测.md`
- 更新 `wiki/topics/视频生成.md`
- 更新 `log.md`

关键变更：

- 按 `图像分层 layered` 页的视觉风格，对其余正式 topic 做一轮统一高亮
- 主要在 `主线脉络 / 方法分层`、`关键争论与分歧` 等章节中，将带标签的论点条目统一改为加粗标签
- 保持 `证据基础` 仍只列 `wiki/summaries/`，不把视觉高亮改成事实层混乱
- 本轮以风格统一为主，不等于逐页重写正文结构；若后续需要，可继续按同一标准逐页提升到更强的 paper 级综述密度
