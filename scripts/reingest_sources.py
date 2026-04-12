#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "pdfs"
TEXT = ROOT / "raw" / "text"
SUMMARIES = ROOT / "raw" / "summary"
INDEX = ROOT / "index.md"


TOPIC_PATHS = {
    "传统NLP": "../topics/传统NLP.md",
    "传统CV": "../topics/传统CV.md",
    "LLM预训练": "../topics/LLM预训练.md",
    "LLM RL": "../topics/LLM%20RL.md",
    "Slide相关": "../topics/Slide相关.md",
}


@dataclass
class SummaryDoc:
    stem: str
    raw_name: str
    year: str
    author: str
    title: str
    topic: str
    abstract: str


def parse_filename(pdf_name: str) -> tuple[str, str, str]:
    stem = Path(pdf_name).stem
    parts = stem.split(" - ", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "Unknown", parts[1]
    return "Unknown", "Unknown", stem


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_abstract(text: str) -> str:
    body = text.split("## Extracted Text", 1)[-1]
    joined = "\n".join(body.splitlines())

    m = re.search(
        r"(?:^|\n)Abstract\s*\n(.+?)(?:\n\s*(?:1[\.\s]|1\s+Introduction|Introduction)\b)",
        joined,
        flags=re.S,
    )
    if not m:
        m = re.search(
            r"(?:^|\n)Abstract\s+(.+?)(?:\n\s*(?:1[\.\s]|1\s+Introduction|Introduction)\b)",
            joined,
            flags=re.S,
        )
    if m:
        return clean_whitespace(m.group(1))

    excerpt = clean_whitespace(body)
    return excerpt[:900]


def classify_topic(name: str, abstract: str) -> str:
    hay = f"{name} {abstract}".lower()

    if "deepseek-v3" in hay:
        return "LLM预训练"
    if "deepseek-r1" in hay:
        return "LLM RL"

    slide_kw = ["slide", "slides", "presentation", "ppt", "lecture presentations"]
    llm_rl_kw = [
        "rlhf", "reinforcement learning", "reward model", "preference", "dpo",
        "align", "alignment", "human feedback", "instructgpt", "cold start",
        "reasoning capability", "reasoning model", "o1",
    ]
    llm_pre_kw = [
        "gpt", "llama", "qwen", "gemma", "deepseek", "large language model",
        "llm", "mixture-of-experts", "moe", "compute-optimal", "few-shot",
        "palm", "open pre-trained transformer", "open pre-trained",
    ]
    cv_kw = [
        "vision", "image", "video", "object detection", "ocr", "document",
        "layout", "table", "matting", "rendering", "caption", "talking face",
        "vit", "detr", "clip", "florence",
    ]
    nlp_kw = [
        "bert", "roberta", "simcse", "sentence embeddings", "dense retriever",
        "dependency parsing", "coreference", "text classification",
        "summarization", "retrieval", "translation", "multilingual",
        "question answering", "text-to-text", "transformer",
    ]

    if any(k in hay for k in slide_kw):
        return "Slide相关"
    if any(k in hay for k in llm_rl_kw):
        return "LLM RL"
    if any(k in hay for k in cv_kw):
        return "传统CV"
    if any(k in hay for k in llm_pre_kw):
        return "LLM预训练"
    if any(k in hay for k in nlp_kw):
        return "传统NLP"
    return "传统NLP"


def build_summary(pdf_path: Path) -> SummaryDoc:
    author, year, title = parse_filename(pdf_path.name)
    text_path = TEXT / f"{pdf_path.stem}.md"
    text = text_path.read_text(encoding="utf-8")
    abstract = extract_abstract(text)
    topic = classify_topic(pdf_path.stem, abstract)
    return SummaryDoc(pdf_path.stem, pdf_path.name, year, author, title, topic, abstract)


def render_summary(doc: SummaryDoc) -> str:
    topic_link = TOPIC_PATHS[doc.topic]
    return f"""# {doc.stem}

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/{doc.raw_name}
- 全文文本：../../raw/text/{doc.stem}.md
- 作者：{doc.author}
- 年份：{doc.year}
- 状态：已抽取全文，待精读

## 自动抽取摘要

{doc.abstract}

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 已存在可读全文文本，可直接从 `raw/text/{doc.stem}.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。

## 关联页面

- 主题：[{doc.topic}]({topic_link})
- 综合：暂无
"""


def short_summary(doc: SummaryDoc) -> str:
    snippet = doc.abstract[:70].rstrip()
    if len(doc.abstract) > 70:
        snippet += "..."
    return snippet


def replace_index_summary(docs: list[SummaryDoc]) -> None:
    text = INDEX.read_text(encoding="utf-8")
    start = text.index("## Summary")
    end = text.index("## Topics")

    grouped: dict[str, list[SummaryDoc]] = {k: [] for k in TOPIC_PATHS}
    for doc in docs:
        grouped[doc.topic].append(doc)

    lines = ["## Summary", "", f"- 当前已批量 ingest {len(docs)} 篇 summary 页，按候选主题分组如下。", ""]
    for topic in ["LLM预训练", "LLM RL", "传统NLP", "传统CV", "Slide相关"]:
        lines.append(f"### {topic}")
        lines.append("")
        if not grouped[topic]:
            lines.append("- 暂无条目。")
        else:
            for doc in sorted(grouped[topic], key=lambda x: x.stem.lower()):
                lines.append(
                    f"- [{doc.stem}](./raw/summary/{doc.stem.replace(' ', '%20')}.md)：{short_summary(doc)}"
                )
        lines.append("")

    updated = text[:start] + "\n".join(lines) + text[end:]
    INDEX.write_text(updated, encoding="utf-8")


def main() -> int:
    pdfs = sorted(RAW.glob("*.pdf"))
    docs = [build_summary(pdf) for pdf in pdfs]
    SUMMARIES.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        (SUMMARIES / f"{doc.stem}.md").write_text(render_summary(doc), encoding="utf-8")

    replace_index_summary(docs)
    print(f"Rebuilt {len(docs)} summary pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
