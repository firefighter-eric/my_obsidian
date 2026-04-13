# AGENTS.md

本文件定义当前知识库仓库的固定结构和维护规则。后续会话默认遵守本文件，除非用户明确要求偏离。

## 来源与定位

本文件是以下概念文档在当前仓库中的落地版本：

- [LLM Wiki](./LLM%20Wiki.md)
- [LLM Wiki_zh](./LLM%20Wiki_zh.md)

它们负责解释方法；本文件负责把方法收敛成当前仓库可执行的规则。

## 1. 仓库结构

当前仓库固定为：

- `raw/pdfs/`：原始 PDF 来源。只读。
- `raw/html/`：原始 HTML 来源。只读。
- `raw/text/`：来源全文的 markdown 文本层。优先保存 arXiv HTML 提取结果；若只有 PDF，则保存从 PDF 提取的 markdown。可重建，不是事实源头。
- `raw/assets/`：图片、附件等辅助原始资源。
- `raw/summary/`：单篇资料的结构化 summary 页。
- `wiki/topics/`：主题页。
- `wiki/concepts/`：概念 / 模型 / 数据集 / 工具等页面。
- `wiki/authors/`：作者页。
- `wiki/comparisons/`：比较页。
- `wiki/timelines/`：时间线页。
- `index.md`：根级总索引。
- `log.md`：根级操作日志。
- `AGENTS.md`：schema / workflow 规范。

若来源是 HTML（例如 arXiv HTML 或网页正文），默认同时保存原始 HTML 到 `raw/html/`，并提取 markdown 到 `raw/text/`。

默认原则：

- 绝不修改 `raw/pdfs/` 中的原始资料。
- `raw/text/` 只用于辅助读取全文；如需重建，应优先从可用的 arXiv HTML 重新提取 markdown，若无 HTML，则从 `raw/pdfs/` 重新抽取 markdown。
- 所有知识组织优先写入 `wiki/`，而不是停留在对话里。
- 页面正文默认中文；专有名词保留英文原名。

## 2. 页面职责

### `raw/summary/`

单篇资料的结构化 summary 页。它不是原文，也不是全文抽取文本，而是介于两者之间、可被主题页和概念页复用的来源摘要层。至少包含：

- 来源信息
- 自动抽取摘要或人工摘要
- 关键事实
- 争议与不确定点
- 关联页面

### `wiki/topics/`

围绕一个研究方向或问题的研究综述型汇总页。`wiki/topics/` 不是概念罗列页，也不是论文目录页；它的职责是对一个主题给出问题定义、内部结构、主线脉络、关键分歧与证据基础。

#### topic 页面成熟度

`wiki/topics/` 页面分为两种状态：

- 正式 topic：满足完整专业模板，且核心论断有足够 `raw/summary/` 支撑，可作为稳定入口出现在 `index.md` 主导航。
- 待建设 topic：资料不足、主线尚未形成或 summary 支撑不足的页面。可以保留短页，但必须明确标注其待建设性质，例如“待补 summary”“待补主线”“暂不形成正式论断”。

禁止把弱占位页视为正式 topic。若一个主题尚无足够 `raw/summary/` 支撑，应先补 `summary`，再升级 topic。

#### 正式 topic 的固定章节模板

正式 topic 默认使用研究综述型结构，至少包含以下部分：

- 主题定义：说明研究对象、问题边界、与相邻主题的切分关系。
- 核心问题：将主题拆成 2-5 个稳定子问题、任务或分析维度。
- 主线脉络 / 方法分层：按时间、方法族、问题族或系统层次组织主题内部结构。
- 关键争论与分歧：说明不同路线的冲突点、适用边界、尚无定论之处。
- 证据基础：列出支持本页核心论断的 `raw/summary/` 页面，并按主线组织，不得只写“例如若干论文”。
- 代表页面：列出支撑该主题的 `concepts / comparisons / timelines / authors` 页面。
- 未解决问题：只写真实未决问题，不写泛泛的“后续继续补”。
- 关联页面：作为导航层使用，并与证据层明确区分。

在不破坏上述结构的前提下，可以保留“主题摘要”“关键观点”等导读段落，但这些导读不能替代完整结构。

#### topic 写作质量要求

- 页面正文默认应为“综述段落 + 结构化列表”混合，而不是纯占位 bullet。
- 每个 topic 必须明确回答：这个主题解决什么问题、内部如何分层、与相邻主题如何切分。
- 禁止把 topic 写成 concept 列表堆砌、论文名单堆砌、或只有一句话摘要加几条待办事项。
- 正式 topic 不得只有 1 段摘要加少量 bullet；至少应覆盖“问题定义、内部结构、证据基础、争议/开放问题”四类信息。
- 正式 topic 必须有交叉链接，不得成为孤立总页。

#### topic 的证据与可追溯性

- topic 的核心论断必须能回溯到一个或多个 `raw/summary/` 页面。
- `wiki/concepts/`、`wiki/comparisons/`、`wiki/timelines/`、`wiki/authors/` 只能作为组织与导航层，不是一级事实来源。
- 默认以 `raw/summary/` 作为 topic 的事实基座；`raw/text/` 用于辅助核对、补 summary 或复查原文，不作为常规 topic 写作的直接替代。
- topic 中跨论文归纳时，必须显式区分：
  - 来源直接支持的事实
  - 编者归纳出的脉络性结论
  - 尚不稳定的推断或不确定点
- 若一个关键判断暂时无法回溯到 `summary`，应保留“不确定”或回到 ingest 流程补 summary，而不是直接写成稳定结论。

### `wiki/concepts/`

模型、方法、数据集、工具、术语等概念页。至少包含：

- 简介
- 关键属性
- 相关主张
- 来源支持
- 关联页面

### `wiki/authors/`

作者或机构页面，用于聚合其在知识库中的来源与主题关系。

### `wiki/comparisons/`

横向比较页，例如模型对比、方法对比、路线对比。

### `wiki/timelines/`

按时间组织的脉络页，例如模型发布时间线或方法演进线。

## 3. 命名规则

- 目录名固定使用英文：`summary`、`topics`、`concepts`、`authors`、`comparisons`、`timelines`。
- 页面标题默认中文，但专有名词保留英文原名。
- `summary` 页文件名优先与 `raw/pdfs/` 中原始文件稳定对应。
- 页面若重命名，必须同步更新 `index.md` 和其他页面中的链接。

## 4. 工作流

### Ingest

当用户要求接入新来源时，按以下顺序执行：

1. 读取根级 `index.md`。
2. 判断来源类型；若来源存在 arXiv HTML 页面，优先检查 `raw/text/` 中是否已有对应 markdown 全文。
3. 若存在 arXiv HTML 且尚未提取全文，优先从该 HTML 提取结构化内容，并生成同名 `.md` 到 `raw/text/`。
4. 若不存在可用 arXiv HTML，但存在 PDF，则检查 `raw/text/` 中是否已有对应 markdown 全文；若无，则使用 `PyMuPDF` 读取 `raw/pdfs/` 中的 PDF，并生成同名 `.md` 到 `raw/text/`。
5. 从原始来源与 `raw/text/` 获取内容；`raw/text/` 始终保留为 markdown 全文层。
6. 创建或更新 `raw/summary/` 页面。
7. 更新相关 `wiki/topics/`、`wiki/concepts/`，必要时也更新 `authors/`、`comparisons/`、`timelines/`。
8. 更新根级 `index.md`。
9. 追加根级 `log.md`。

ingest / topic / concept 联动规则：

- 若新增 `wiki/topics/` 页面，必须同时引入新的 `raw` 来源材料；不得只基于现有 topic/concept 页面空转生成新 topic。
- 若新增 topic，除整理对应 `raw`、`raw/text/`、`raw/summary/` 外，还应判断是否沉淀出高价值 `wiki/concepts/` 页面；若存在清晰、可复用、可稳定定义的概念，应一并总结。
- 若新增 `wiki/concepts/` 页面，默认应从现有 `raw/summary/`、必要时结合 `raw/text/` 总结；不得在没有现有 raw 素材支撑时凭空创建稳定 concept。
- 若新增 `raw` 素材，不止要保存原始文件，还必须完成整理流程：补 `raw/text/`、补/更 `raw/summary/`，并同步判断和总结由该素材支撑的新 concept。
- 简言之：`topic` 的新增以“新 raw 素材”为前提；`concept` 的新增以“现有 raw 素材可支撑”为前提；`raw` 的新增默认伴随整理与 concept 沉淀。

topic 相关补充规则：

- 若新增来源会改变某主题的主线、方法分层、边界定义或关键争论，必须更新对应 `wiki/topics/` 页面，而不只是补链接。
- 更新 topic 时，优先把新增来源沉淀为“问题定义 / 主线脉络 / 关键分歧 / 证据基础”的结构性变化，而不是在页面末尾机械追加论文。
- 若某主题仍缺少足够 `summary` 支撑，则应先补 `raw/summary/`，必要时将 topic 保持为“待建设 topic”，不得伪装成正式 topic。
- 若新增 topic 使用了新的论文、网页或其他来源材料，必须先检查该来源是否已在 `raw/html/` 或 `raw/pdfs/` 中保留原始文件；若没有原始文件，先下载并保存原始 HTML 或 PDF，再生成 `raw/text/` 与 `raw/summary/`，最后再把它纳入正式 topic。

PDF 特别规则：

- PDF 的全文抽取属于预处理步骤，不等于完成 ingest。
- 若来源有可用 arXiv HTML，默认优先使用 arXiv HTML 提取 markdown 到 `raw/text/`。
- 若没有可用 arXiv HTML、只有 PDF，则 `raw/text/` 中的 markdown 默认由 `PyMuPDF` 生成；批量抽取脚本是 `scripts/extract_pdf_text.py`。
- arXiv 下载脚本是 `scripts/download_arxiv.py`；它可把 PDF 下载到 `raw/pdfs/`，把 arXiv HTML 保存到 `raw/html/`，并把 HTML 提取成 `raw/text/*.md`。
- 若 `raw/text/` 已存在对应文件，优先复用，不重复抽取。
- 无论来源是 arXiv HTML 还是 PDF，`raw/text/` 的最终保存格式都应为 markdown。
- `raw/text/*.md` 必须在文件头明确标注其转换来源：若来自 HTML，则写明对应 `raw/html/*.html`；若来自 PDF，则写明对应 `raw/pdfs/*.pdf`。
- 后续分析、摘要、归类、交叉链接仍然按普通 summary 页流程执行。

### Query

回答问题时，默认顺序是：

1. 先读取根级 `index.md`。
2. 进入相关 `topics/concepts/comparisons/timelines` 页面。
3. 回答时明确依据来自哪些 wiki 页面。
4. 若产生稳定知识产物，优先写成 `comparisons/` 或 `timelines/` 页面，而不是临时聊天内容。

topic 相关补充规则：

- 回答主题性问题时，优先引用 topic 页中的结构化结论，再回到支撑这些结论的 `raw/summary/` 页面。
- 若现有 topic 仍是“待建设 topic”或弱占位页，应明确说明其成熟度不足，不能把其中的占位描述当成稳定结论。
- 若回答过程中形成了稳定的主题结构、方法分层或路线对照，优先更新 `wiki/topics/`、`wiki/comparisons/` 或 `wiki/timelines/`，而不是只保留在对话里。

### Lint

健康检查时，重点检查：

- 页面之间是否矛盾
- 是否存在孤儿页
- 是否缺少关键交叉链接
- 是否存在无来源支持的断言
- 是否有重要主题缺少 `summary` / `concepts` / `comparisons` 支撑
- 是否存在被误当作正式 topic 的弱占位页
- topic 是否缺少足够 `raw/summary/` 支撑
- topic 是否只有概念罗列、没有主线结构
- topic 与相邻主题是否边界重叠、定义冲突或互相矛盾
- topic 中的“未解决问题”是否真实未决，而不是待办事项伪装

## 5. 根级文件规则

### `index.md`

这是整个知识库的第一入口，不放在 `wiki/` 内。

要求：

- 至少按 `Summary / Topics / Concepts / Authors / Comparisons / Timelines` 组织。
- 每个实质页面都应有一条记录。
- 每条记录包含页面链接和一句话摘要。
- `Topics` 区默认优先登记正式 topic。
- 若确需登记“待建设 topic”，应在摘要中显式标出其待建设状态，避免与正式 topic 混淆。

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
- 若某文件仍需参与知识库工作流（例如 `log.md`），即使被 Obsidian 忽略，也仍应按本规范继续维护

## 6. 谨慎原则

- 不确定时，保留“不确定”而不是补全猜测。
- 跨页面结论必须能回溯到一个或多个 `summary` 页。
- 自动抽取摘要只作为 ingest 起点，不等于最终综述。
