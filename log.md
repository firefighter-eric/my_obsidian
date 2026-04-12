# Wiki Log

本页是 LLM Wiki 的追加式操作日志。

## [2026-04-12] ingest | 扩展 LLM RL 方法谱系

涉及页面：

- [LLM RL](./wiki/topics/LLM%20RL.md)
- [RLHF](./wiki/concepts/RLHF.md)
- [GRPO](./wiki/concepts/GRPO.md)
- [ORPO](./wiki/concepts/ORPO.md)
- [KTO](./wiki/concepts/KTO.md)
- [DAPO](./wiki/concepts/DAPO.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](./raw/summary/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](./raw/summary/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](./raw/summary/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](./raw/summary/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- [index](./index.md)

关键变更：

- 补入 4 篇 `LLM RL` 关键来源：`DeepSeekMath`、`ORPO`、`KTO`、`DAPO`，并同步保存 `raw/pdfs/`、`raw/html/`、`raw/text/`
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

- 扫描 `raw/pdfs/` 下全部 `175` 个 PDF，优先从文件名和 PDF 前两页提取 arXiv id，少数缺 id 的条目再用标题到 arXiv API 做谨慎补判
- 对命中的 `165` 个 PDF 补齐 `raw/html/*.html`，并用 arXiv HTML 导出的 markdown 覆盖对应 `raw/text/*.md`
- 为 `DeepSeek-V3`、`DeepSeek-R1` 与重复文件 `DETRs with Collaborative Hybrid Assignments Training(2)` 增加确定性映射，避免标题补判漏掉已知 arXiv 来源
- 保持 `raw/pdfs/` 原件不变，只迁移全文层的来源优先级，使后续 summary / topic 默认回到 HTML 文本层

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

- 对全部 `raw/summary/*.md -> raw/text/*.md` 链路执行扫描，检查是否存在 `Source PDF / Source HTML`
- 批量修正旧 `raw/text` 中错误的 `Source PDF` 路径
- 为当前 12 个网页 / HTML 来源补齐 `raw/html/*.html` 原件
- 为所有网页来源的 `raw/text/*.md` 补齐 `Source HTML` 头标记
- 确认当前 `187` 篇 summary 全部具备可回溯的 `raw/text`，且其来源头不再缺失

## [2026-04-12] lint | 执行 arXiv HTML / markdown 全文层规则

涉及页面：

- [AGENTS](./AGENTS.md)
- [log](./log.md)
- `raw/text/` 中新增的网页 markdown 全文页
- [Hello Qwen2](./raw/summary/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)
- [Introducing Qwen1.5](./raw/summary/Qwen%20Team%20-%202024%20-%20Introducing%20Qwen1.5.md)
- [Qwen2-VL](./raw/summary/Qwen%20Team%20-%202024%20-%20Qwen2-VL.md)
- [Qwen2.5-LLM Extending the boundary of LLMs](./raw/summary/Qwen%20Team%20-%202024%20-%20Qwen2.5-LLM%20Extending%20the%20boundary%20of%20LLMs.md)
- [Qwen2.5-Omni See Hear Talk Write Do It All](./raw/summary/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md)
- [Qwen3 Think Deeper Act Faster](./raw/summary/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
- [Qwen3.5 Towards Native Multimodal Agents](./raw/summary/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)
- [Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](./raw/summary/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)
- [Qwen-Image Crafting with Native Text Rendering](./raw/summary/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)
- [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](./raw/summary/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)
- [Stability AI - 2022 - Stable Diffusion Launch Announcement](./raw/summary/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)
- [Black Forest Labs - 2026 - FLUX.2 Overview](./raw/summary/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md)
- [fetch_web_text.py](./scripts/fetch_web_text.py)

关键变更：

- 将 ingest 规则落到项目层：网页来源也统一补齐 `raw/text/*.md` markdown 全文层
- 新增 `scripts/fetch_web_text.py`，支持 `HTML -> markdown`
- 对当前缺少 `全文文本` 的来源页完成审计与补链
- 对 `qwen.ai` 动态页面使用浏览器渲染提取，避免规则只覆盖静态 HTML
- 修复 `qwenlm.github.io/blog/*` 跳转页只抓到 redirect 提示的问题，改为优先抽取页面内 `articleBody`
- 补入 `uv + Python 3.12` 项目配置，并为 HTML 来源恢复 `raw/html/` 原件保存层

后续建议：

- 后续 ingest 对网页来源先补 `raw/text`，再写 `raw/summary`
- 若动态站点继续增多，可把 `playwright` 流程进一步脚本化

## [2026-04-12] ingest | 接入 diffusion 文生图主线

涉及页面：

- [扩散模型与文生图](./wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Stable Diffusion](./wiki/concepts/Stable%20Diffusion.md)
- [FLUX.2](./wiki/concepts/FLUX.2.md)
- [Qwen-Image](./wiki/concepts/Qwen-Image.md)
- [Qwen 系列](./wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md)
- [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](./raw/summary/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)
- [Stability AI - 2022 - Stable Diffusion Launch Announcement](./raw/summary/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)
- [Black Forest Labs - 2026 - FLUX.2 Overview](./raw/summary/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md)
- [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](./raw/summary/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)
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

- 固定基础结构：`raw/pdfs/`、`wiki/`、`AGENTS.md`
- 初始化首批 `raw/summary/` 与 `wiki/topics/` 目录
- 建立索引与操作日志入口

后续建议：

- 选择一个 `raw/pdfs/` 来源做首个真实 ingest
- 从 ingest 结果反向微调模板和命名约定

## [2026-04-12] ingest | 首轮 LLM 基础脉络铺开

涉及页面：

- [LLM 基础脉络](./wiki/topics/LLM%20基础脉络.md)
- [Scaling 与 compute-optimal training](./wiki/topics/Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./wiki/topics/指令对齐与%20post-training.md)
- [GPT-3](./wiki/concepts/GPT-3.md)
- [Llama 3](./wiki/concepts/Llama%203.md)
- [Brown et al. - 2020 - Language models are few-shot learners](./raw/summary/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](./raw/summary/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](./raw/summary/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
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
- [LLM 基础脉络](./wiki/topics/LLM%20基础脉络.md)
- [Scaling 与 compute-optimal training](./wiki/topics/Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./wiki/topics/指令对齐与%20post-training.md)

关键变更：

- 为 `raw/pdfs/` 下全部 PDF 批量生成 `raw/text/*.md` 全文文本文件
- 新增 `scripts/extract_pdf_text.py` 作为可重复执行的全文抽取脚本
- 将现有 3 个主题页从摘要层重写为基于正文可支持信息的主线页

后续建议：

- 用 `raw/text/` 中的全文内容重写首轮 3 个summary 页，使其不再只依赖摘要
- 继续把 `PaLM`、`Qwen`、`Llama 3`、`DPO` 等已可读全文接入 `raw/summary/`
- 若后续页数明显增长，可为 `raw/text/` 增加索引或批量校验工具

## [2026-04-12] ingest | 重组主题为五大类

涉及页面：

- [传统 NLP](传统%20NLP.md)
- [传统 CV](传统%20CV.md)
- [LLM 预训练](LLM%20预训练.md)
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

- [Unknown - 2024 - DeepSeek-V3 Technical Report](./raw/summary/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [Unknown - 2024 - DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](./raw/summary/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek](./wiki/concepts/DeepSeek.md)
- [LLM 预训练](LLM%20预训练.md)
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
- `raw/summary/` 下全部summary 页

关键变更：

- 删除旧的 5 个试验版summary 页
- 按 `raw/pdfs/` 与 `raw/text/` 全量重建 172 个统一格式的summary 页
- `Summary` 索引改为按候选主题分组展示

后续建议：

- 优先精修主干来源，而不是一次性精修全部 172 篇
- 对 `LLM预训练 / LLM RL / Slide相关` 继续补更多高价值summary 页和实体页

## [2026-04-12] ingest | 对齐 knowledge-base 目录结构

涉及页面：

- [AGENTS](./AGENTS.md)
- [index](./index.md)
- [log](./log.md)
- `raw/pdfs/`
- `raw/text/`
- `raw/assets/`
- `raw/summary/`
- `wiki/topics/`
- `wiki/concepts/`
- `wiki/authors/`
- `wiki/comparisons/`
- `wiki/timelines/`

关键变更：

- 将原始 PDF 统一收敛到 `raw/pdfs/`
- 将 PDF 全文文本统一收敛到 `raw/text/`
- 将总索引与操作日志固定在根级 `index.md` 与 `log.md`
- 将 `wiki/` 目录固定为 `sources / topics / concepts / authors / comparisons / timelines`
- 重新生成全部 `raw/summary/` 页面，使其引用新的 `raw/pdfs/` 与 `raw/text/` 路径

后续建议：

- 后续新增 PDF 时先放入 `raw/pdfs/`，再运行全文抽取与 ingest 脚本
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
- [LLM 预训练](LLM%20预训练.md)
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
- [LLM 预训练](LLM%20预训练.md)
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
- 明确 topic 的核心论断必须优先回溯到 `raw/summary/`，并区分证据层与导航层
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

- `raw/summary/` 下新增 8 个 Qwen 家族来源页
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

- `raw/pdfs/` 下新增 3 篇 Llama 家族论文 PDF
- `raw/text/` 下新增 3 篇对应全文文本
- `raw/summary/` 下新增 3 个 Llama 家族来源页
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
- 更新 [Bai et al. - 2023 - Qwen Technical Report](./raw/summary/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
- 更新 [Bai et al. - 2025 - Qwen2.5-VL Technical Report](./raw/summary/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md)
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
- 更新 [传统 CV](传统%20CV.md)
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

- 新增正式 topic [Slide 理解与生成](Slide%20理解与生成.md)
- 保留兼容入口 [Slide 理解与生产](./wiki/topics/Slide%20理解与生产.md) 与 [Slide相关](./wiki/topics/Slide相关.md)
- 更新 [传统 CV](传统%20CV.md)
- 更新 [index](./index.md)

关键变更：

- 按字面名称将正式 topic 从 `Slide 理解与生产` 调整为 `Slide  理解与生成`
- 同步将主题表述从“生产”统一收紧为“生成”，以保持标题、正文与方法分层一致
- 保留两层兼容入口，避免新旧 topic 名称之间的历史链接断裂

## [2026-04-12] query | 新建类 BERT 双向 Transformer 语言模型 topic

涉及页面：

- 新增正式 topic [BERT类双向Transformer语言模型](./wiki/topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- 更新 [传统 NLP](传统%20NLP.md)
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
