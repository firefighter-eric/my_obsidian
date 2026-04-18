# AGENTS.md

本文件定义当前知识库仓库的固定结构和维护规则。后续会话默认遵守本文件，除非用户明确要求偏离。

## 来源与定位

本文件是以下概念文档在当前仓库中的落地版本：

- [LLM Wiki](./LLM%20Wiki.md)
- [LLM Wiki_zh](./LLM%20Wiki_zh.md)

它们负责解释方法；本文件负责把方法收敛成当前仓库可执行的规则。

优先级规则：

- `LLM Wiki.md` 与 `LLM Wiki_zh.md` 是当前仓库的最高层方法来源。
- 关于 `ingest / query / lint` 的设计与执行，最重要的判断依据始终是 `LLM Wiki` 的核心方法，而不是局部流程措辞。
- `AGENTS.md` 的职责不是替代 `LLM Wiki`，而是把 `LLM Wiki` 在本仓库中实例化为可执行 schema、目录约束与 workflow。
- 当前仓库已有的流程规范可以参考、继承和本地化，但只能作为 `LLM Wiki` 方法在本仓库中的实现版本，不能凌驾于 `LLM Wiki` 之上。
- 若后续会话中出现局部规则歧义，优先回到 `LLM Wiki` 的核心原则理解：构建并维护一个持续累积、持续修订、可回写的 persistent wiki。
- 若 `AGENTS.md` 某条局部表述与 `LLM Wiki` 的核心方法明显冲突，应优先按 `LLM Wiki` 的方法意图修正 `AGENTS.md`，而不是机械坚持冲突条目。

## 1. 三层结构

当前仓库严格收敛为三层：

- `raw/`：原始来源与可重建全文层。
- `wiki/`：所有 LLM 生成、维护和持续沉淀的知识产物。
- `AGENTS.md`：本仓库的本地 schema / workflow 规则。

核心原则：

- `raw` 不是知识组织层，只保存来源文件与可重建全文。
- `wiki` 承载 summary、topic、concept、comparison、timeline、author 等全部知识页面。
- 跨页面结论必须能回溯到一个或多个 `wiki/summaries/` 页面。
- 页面正文默认中文；专有名词保留英文原名。

## 2. 仓库结构

当前仓库固定为：

- `raw/pdf/`：原始 PDF 来源。只读。
- `raw/html/`：原始 HTML 来源。只读。
- `raw/text/`：来源全文的 markdown 文本层。优先保存 arXiv HTML 提取结果；若只有 PDF，则保存从 PDF 提取的 markdown。可重建，不是事实源头。
- `raw/assets/`：图片、附件等辅助原始资源。
- `wiki/summaries/`：单篇资料的结构化 summary 页，属于 wiki 的来源摘要层。
- `wiki/topics/`：主题页。
- `wiki/concepts/`：概念 / 模型 / 数据集 / 工具等页面。
- `wiki/authors/`：作者与机构聚合页。
- `wiki/comparisons/`：比较页。
- `wiki/timelines/`：时间线页。
- `index.md`：根级总索引。
- `log.md`：根级操作日志。
- `AGENTS.md`：schema / workflow 规范。

默认原则：

- 绝不修改 `raw/pdf/` 与 `raw/html/` 中的原始资料。
- `raw/text/` 只用于辅助读取全文；如需重建，应优先从可用的 HTML 重新提取 markdown，若无 HTML，再从 `raw/pdf/` 重抽。
- 新增来源时，`raw` 链路必须完整：先有 `raw/html/` 或 `raw/pdf/` 中的原始文件，再有 `raw/text/` 中的 markdown 全文层；之后才允许创建或更新 `wiki/summaries/`。
- 所有稳定知识组织优先写入 `wiki/`，而不是停留在对话里。

## 3. 页面职责

### `wiki/summaries/`

单篇资料的结构化 summary 页。它不是原文，也不是全文抽取文本，而是介于两者之间、可被 topic / concept / comparison / timeline 复用的来源摘要层。至少包含：

- 来源信息
- 摘要
- 关键事实
- 争议与不确定点
- 关联页面

summary 状态约定：

- 精修 summary：可直接支撑正式 topic 的核心论断。
- 待精读自动摘要：只作为整理入口，不应直接支撑正式 topic 的关键结论。

### `wiki/topics/`

围绕一个研究方向或问题的研究综述型汇总页。`wiki/topics/` 不是概念罗列页，也不是论文目录页；它的职责是对一个主题给出问题定义、内部结构、主线脉络、关键分歧与证据基础。

对正文质量的默认要求是：`wiki/topics/` 中的**正式 topic 必须按“paper 级综述”标准书写**，而不是停留在简短介绍、要点堆砌或来源列表。这里的 paper 级综述，指页面应接近一篇成熟 survey / review article 的综述密度：不仅说明“有哪些工作”，还要说明“它们为什么这样分层、彼此解决的究竟是不是同一问题、关键分歧来自哪里、证据支持到什么程度、哪些结论已经相对稳定、哪些判断仍然不确定”。**topic 页必须体现综合、比较、抽象、取舍与判断**，而不是把 summary 改写成更长的列表。

topic 写作的最低质量线：

- **不能只做论文罗列、概念摘抄或按年份堆积工作。**
- **不能只写“有哪些方法”**，还必须解释这些方法在问题设定、假设条件、能力边界、失败模式或评测口径上的结构性差异。
- **不能只重复各 `wiki/summaries/` 的局部结论**，必须在 topic 层形成更高一层的归纳、对照与综合判断。
- **应明确区分**“相对稳定的共识”“依赖特定设定的结论”“仍有争议的问题”“当前证据不足的推测”。
- 若作者尚无法写出具有主线、分层、比较与判断的综述正文，则该页应**降级为“待建设 topic”**，而不是以简短内容充当正式 topic。

#### topic 页面成熟度

`wiki/topics/` 页面分为两种状态：

- **正式 topic**：满足完整专业模板，且核心论断有足够 `wiki/summaries/` 支撑；正文质量达到 paper 级综述标准，能够体现主线组织、横向比较、证据约束与综合判断，而不是简单罗列材料；可作为稳定入口出现在 `index.md` 主导航。
- **待建设 topic**：资料不足、主线尚未形成或 summary 支撑不足的页面。可以保留短页，但必须明确标注其待建设性质。

#### 正式 topic 的固定章节模板

**正式 topic 至少包含以下固定章节：**

- 主题定义
- 核心问题
- 主线脉络 / 方法分层
- 关键争论与分歧
- 证据基础
- 代表页面
- 未解决问题
- 关联页面

除章节存在外，还要求这些章节**具备综述写作功能，而不是形式化占位**：

- `主题定义` 不只给术语解释，还应交代边界、相邻问题及为何值得单独成题。
- `核心问题` 不只列问题名，还应指出该领域真正卡住研究进展的难点。
- `主线脉络 / 方法分层` 应体现作者对领域结构的归纳，而不是照搬论文自带分类。
- `关键争论与分歧` 应解释争议产生的原因、双方证据及各自成立条件。
- `证据基础` **不只是参考列表，而是 topic 核心判断的可追溯支撑面。**
- `未解决问题` 应是从现有证据与分歧中推导出的真实开放问题，而不是泛泛而谈的 future work。

#### topic 的证据与可追溯性

- **topic 的核心论断必须能回溯到一个或多个 `wiki/summaries/` 页面。**
- `wiki/concepts/`、`wiki/comparisons/`、`wiki/timelines/`、`wiki/authors/` 只能作为组织与导航层，不是一级事实来源。
- **`证据基础` 只列 `wiki/summaries/` 页面**，不混入 topic / concept / author / comparison / timeline。
- 若一个关键判断暂时无法回溯到 `summary`，应保留“不确定”，或回到 ingest 流程补 summary。
- topic 中的综合判断、方法分层与争议分析可以是作者在多个 `wiki/summaries/` 基础上的高层归纳，但必须能解释其证据来源，不得写成无支撑的个人印象。

### `wiki/concepts/`

模型、方法、数据集、工具、术语等概念页。至少包含：

- 简介
- 关键属性
- 相关主张
- 来源支持
- 关联页面

额外规则：

- `来源支持` 只列 `wiki/summaries/`。
- topic / comparison / timeline / author 链接一律放到 `关联页面`，不进入 `来源支持`。

### `wiki/authors/`

作者或机构页面，用于聚合其在知识库中的来源与主题关系。它是导航层，不是启发式脏统计报表。

### `wiki/comparisons/`

横向比较页，例如模型对比、方法对比、路线对比。适合承接 query 过程中形成的稳定对照。

### `wiki/timelines/`

按时间组织的脉络页，例如模型发布时间线或方法演进线。适合承接成熟家族或方法谱系。

## 4. 命名规则

- 目录名固定使用英文：`summaries`、`topics`、`concepts`、`authors`、`comparisons`、`timelines`。
- 页面标题默认中文，但专有名词保留英文原名。
- `wiki/summaries/` 文件名优先与对应原始来源稳定对应。
- 页面若重命名，必须同步更新 `index.md` 和其他页面中的链接。

## 5. 工作流

### Ingest

目标：

- ingest 不是“把文件放进仓库”这么简单，而是把一个新来源整合进 persistent wiki，使其成为后续 query 与综合的可复用知识单元。
- ingest 的直接产物至少包括：原始来源层、可重建全文层、summary 层，以及必要的 wiki 交叉更新。
- ingest 完成后，相关知识不应只存在于对话里，而应被写进 wiki 并纳入后续可维护结构。

当用户要求接入新来源时，按以下顺序执行：

1. 读取根级 `index.md`。
2. 判断来源类型，并先确认原始文件已落到 `raw/html/` 或 `raw/pdf/`；若原始文件还不存在，先下载并保存原始文件。
3. 若来源存在 arXiv HTML 页面，优先检查 `raw/text/` 中是否已有对应 markdown 全文。
4. 若存在 arXiv HTML 且尚未提取全文，优先从该 HTML 提取结构化内容，并生成同名 `.md` 到 `raw/text/`。
5. 若不存在可用 arXiv HTML，但存在 PDF，则检查 `raw/text/` 中是否已有对应 markdown 全文；若无，则使用 `PyMuPDF` 读取 `raw/pdf/` 中的 PDF，并生成同名 `.md` 到 `raw/text/`。
6. 从原始来源与 `raw/text/` 获取内容。
7. 仅在原始文件与 `raw/text/` 都已存在后，创建或更新 `wiki/summaries/` 页面。
8. 更新相关 `wiki/topics/`、`wiki/concepts/`，必要时也更新 `wiki/authors/`、`wiki/comparisons/`、`wiki/timelines/`。
9. 更新根级 `index.md`。
10. 追加根级 `log.md`。

ingest 约束：

- 严禁跳过原始层直接新增 `wiki/summaries/`。
- 新增 raw 时必须满足 `raw/html 或 raw/pdf -> raw/text -> wiki/summaries` 的顺序。
- 若发现某来源只有 `wiki/summaries/`、却没有对应 `raw/html/` 或 `raw/pdf/` 与 `raw/text/`，应先补齐缺失层，再把该来源视为可用。
- ingest 应默认检查该来源是否会修正、加强、削弱或否定现有 wiki 中的已有说法，并把这种影响写回相关页面。
- 一个来源可以触及多个 wiki 页面；不应只生成一页 summary 就停止，若它明显影响已有 topic / concept / comparison / timeline，应一并更新。
- 自动摘要只是 ingest 起点；若某来源后续成为关键证据，应把对应 `wiki/summaries/` 精修到足以支撑正式 topic。

### Query

目标：

- query 不是临时从原始资料里重新拼答案，而是优先利用现有 wiki 作为已沉淀、已交叉引用、已持续维护的知识中间层。
- query 既是回答问题，也是发现结构缺口、生成新 wiki 页面、推动知识继续复利的机会。

回答问题时，默认顺序是：

1. 先读取根级 `index.md`。
2. 若 `index.md` 不足以定位页面，可使用本地搜索工具在 `index.md`、`wiki/` 与 `raw/text/` 中召回候选页。
3. 优先进入相关 `topics / concepts / comparisons / timelines` 页面。
4. 必要时回溯到相关 `wiki/summaries/` 页面，确认核心判断的来源支持。
5. 回答时明确依据来自哪些 wiki 页面。
6. 若产生稳定知识产物，优先写回 `wiki/topics/`、`wiki/comparisons/` 或 `wiki/timelines/`，而不是只停留在对话里。

query 约束：

- 允许基于现有 `wiki/summaries/` 直接沉淀新的 topic / comparison / timeline。
- 新增 topic 不再以“必须同时新增 raw”为硬前提；关键前提是其核心论断能回溯到现有 `wiki/summaries/`。
- 若现有 topic 仍是“待建设 topic”或弱占位页，应**明确说明其成熟度不足**，不能把占位描述当成稳定结论。
- 若要把某页提升为正式 topic，默认应把正文扩展到 **paper 级综述密度**：有清晰问题意识、方法分层、横向比较、争议分析与证据约束；**不能只做轻量整理**。
- 若 query 形成了有复用价值的比较、分析、框架、脉络梳理，应优先把它沉淀成新页面，而不是让高价值结论消失在聊天记录中。
- 回答可引用 raw 层做补充核对，但默认不应绕开 wiki 重新做一次“从零 RAG”；优先使用 wiki，再在必要时向下回溯。
- 搜索工具的职责是“召回候选页”，不是替代 `index.md`、`wiki/` 或 `wiki/summaries/` 的事实地位。
- 搜索命中后，默认先打开 `wiki/topics/`、`wiki/concepts/`、`wiki/comparisons/`、`wiki/timelines/`；若涉及具体证据，再回到 `wiki/summaries/`。
- `raw/text/` 只作为全文补查与核对层，不直接替代 `wiki/summaries/` 作为稳定依据。
- 若查询暴露出现有 wiki 缺口，例如缺 summary、缺 concept、缺 cross-link、缺对比页，应优先补齐相关页面。

### Lint

目标：

- lint 是对 persistent wiki 的健康检查，不只是找格式问题，而是检查知识结构是否仍然一致、可导航、可追溯、可继续增长。
- lint 的价值在于发现矛盾、过时结论、结构缺口、导航断裂以及下一步值得补充的来源或页面。

健康检查时，重点检查：

- 页面之间是否矛盾
- 旧结论是否已被较新来源修正、削弱或推翻
- 是否存在孤儿页
- 是否缺少关键交叉链接
- 是否存在无来源支持的断言
- 是否有重要主题缺少 `summary / concepts / comparisons / timelines` 支撑
- 是否存在被误当作正式 topic 的弱占位页
- topic 是否缺少足够 `wiki/summaries/` 支撑
- topic 的 `证据基础` 是否混入非 summary 页面
- topic 是否只是论文罗列或摘要拼接，尚未达到 **paper 级综述**应有的综合、比较与判断密度
- concept 的 `来源支持` 是否混入 topic 页面
- authors 页是否出现明显非实体噪声项
- 是否有被多次提及、但仍未拥有独立页面的重要概念 / 模型 / 数据集 / 作者 / 方法
- 是否存在明显数据空白，值得通过新增来源或后续 web search 补齐
- 是否有页面虽存在，但几乎没有被其他页面引用，导致 wiki 复利效应不足

## 6. 本仓库可用工具

当前仓库允许优先使用以下本地工具与脚本来支持 ingest / query / lint：

- `PyMuPDF`（Python 包名 `pymupdf`，代码中通常以 `fitz` 使用）：用于读取 `raw/pdf/` 中的 PDF，并抽取文本到 `raw/text/`。当前对应脚本是 `scripts/extract_pdf_text.py`。
- `requests`：用于下载网页、arXiv PDF、arXiv HTML 等原始来源。当前对应脚本包括 `scripts/download_arxiv.py` 与 `scripts/fetch_web_text.py`。
- `qmd` CLI：本地 markdown 检索工具，用于在 `query` 阶段对 `index.md`、`wiki/` 与 `raw/text/` 做候选页召回；它是检索层，不是事实层。
- `scripts/search_wiki.py`：当前仓库的 `qmd` 包装入口，统一约束索引范围、结果分层与默认排序。
- `scripts/fetch_web_text.py`：用于抓取普通网页，并把正文提取成 markdown；提取逻辑基于标准库 `html.parser` 与页面中的 JSON-LD 信息。
- `scripts/download_arxiv.py`：用于把 arXiv 来源落到 `raw/pdf/`、`raw/html/`、`raw/text/` 三层，适合 arXiv ingest。
- `scripts/extract_pdf_text.py`：用于把已有 PDF 批量或单篇转换为 `raw/text/` markdown。

辅助规则：

- 若来源同时存在 arXiv HTML 与 PDF，优先使用 HTML 提取 `raw/text/`；PDF 抽取作为后备方案。
- 若只是为了读取本地 PDF 内容，优先使用 `PyMuPDF`，不要手工复制 PDF 文本。
- 若只是为了抓取普通网页正文，优先复用 `scripts/fetch_web_text.py` 的现有提取逻辑，而不是每次从零写抓取代码。
- 若需要在现有 wiki 中快速定位候选页，优先通过 `scripts/search_wiki.py` 调用 `qmd`，而不是在对话里手工枚举文件。

历史脚本说明：

- `scripts/import_arxiv_html_for_pdfs.py`、`scripts/reingest_sources.py`、`scripts/rebuild_author_analysis.py` 含有旧目录命名或旧结构假设，默认不应直接作为当前规范下的正式工作流依赖。
- 若后续需要使用这些历史脚本，应先检查其路径常量、输出位置与当前 `AGENTS.md` 结构是否一致，再决定是修复后继续使用，还是重写。

## 7. 根级文件规则

### `index.md`

这是整个知识库的第一入口，不放在 `wiki/` 内。

要求：

- 至少按 `Summary / Topics / Concepts / Authors / Comparisons / Timelines` 组织。
- 每个实质页面都应有一条记录。
- 每条记录包含页面链接和一句话摘要。
- `Topics` 区默认优先登记正式 topic。
- 若确需登记“待建设 topic”，应在摘要中显式标出其待建设状态。

### `log.md`

这是追加式时间线，不回写历史内容，只新增条目。

标题格式统一为：

- `## [YYYY-MM-DD] ingest | 标题`
- `## [YYYY-MM-DD] query | 问题`
- `## [YYYY-MM-DD] lint | 范围`

### Obsidian ignore

若需在 Obsidian 中忽略某个目录或文件，不改变仓库结构本身，而是更新 `.obsidian/app.json` 中的 `userIgnoreFilters`。

规则：

- 目录忽略使用目录前缀形式，例如：`raw/`
- 单文件忽略使用相对根目录路径，例如：`log.md`
- 新增或修改 ignore 规则时，应保留现有条目，不随意覆盖
- 这是 Obsidian 可见性规则，不等于删除、移动或停止维护对应文件

## 8. 谨慎原则

- 不确定时，保留“不确定”而不是补全猜测。
- 跨页面结论必须能回溯到一个或多个 `wiki/summaries/` 页。
- 自动抽取摘要只作为 ingest 起点，不等于最终综述。
- 正式 topic 不应建立在“待精读自动摘要”之上；若必须使用，应明确降级为待建设或先精修 summary。
