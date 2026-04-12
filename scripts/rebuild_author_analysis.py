#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "raw" / "text"
AUTHORS_DIR = ROOT / "wiki" / "authors"

AUTHOR_PAGE = AUTHORS_DIR / "作者分析.md"
INSTITUTION_PAGE = AUTHORS_DIR / "机构分析.md"

INSTITUTION_KEYWORDS = {
    "university",
    "institute",
    "school",
    "college",
    "academy",
    "laboratory",
    "lab",
    "labs",
    "research",
    "group",
    "center",
    "centre",
    "department",
    "faculty",
    "meta ai",
    "deepmind",
    "openai",
    "alibaba",
    "microsoft",
    "google",
    "nvidia",
    "amazon",
    "anthropic",
    "baidu",
    "bytedance",
    "hugging face",
    "toyota technological institute",
    "team",
}

NON_AUTHOR_PHRASES = {
    "abstract",
    "contents",
    "introduction",
    "appendix",
    "figure",
    "table",
    "published as a conference paper",
    "association for computational linguistics",
    "a detailed contributor list can be found in the appendix of this paper",
}

TITLE_WORDS = {
    "survey",
    "report",
    "models",
    "model",
    "language",
    "vision",
    "learning",
    "large",
    "transformer",
    "transformers",
    "generation",
    "recognition",
    "optimization",
    "preferences",
    "tokens",
    "paper",
    "framework",
    "analysis",
}

AUTHOR_BLOCKLIST_WORDS = {
    "direct",
    "preference",
    "optimization",
    "image",
    "recognition",
    "learning",
    "models",
    "model",
    "language",
    "training",
    "generation",
    "report",
    "framework",
    "evaluation",
    "paper",
}

NAME_PATTERN = re.compile(r"[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3}")


@dataclass
class SourceMeta:
    stem: str
    source_link: str
    authors: list[str]
    institutions: list[str]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slug_link(stem: str) -> str:
    return f"../../raw/summary/{stem.replace(' ', '%20')}.md"


def clean_affiliation_markers(text: str) -> str:
    text = re.sub(r"[\*†‡§¶⋆∗]+", " ", text)
    text = re.sub(r"(?<=\D)\d+(?=\D|$)", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    return normalize_space(text)


def front_lines(text: str) -> list[str]:
    body = text.split("## Extracted Text", 1)[-1]
    lines = [normalize_space(line) for line in body.splitlines()]
    results: list[str] = []
    for line in lines:
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(("abstract", "contents", "1 introduction", "introduction")):
            break
        results.append(line)
        if len(results) >= 24:
            break
    return results


def looks_like_metadata(line: str) -> bool:
    lower = line.lower()
    if lower in NON_AUTHOR_PHRASES:
        return True
    if lower.startswith(("published as", "proceedings of", "arxiv:", "preprint", "february", "june", "july", "august", "september", "october", "november", "december")):
        return True
    if "conference paper" in lower or "association for computational linguistics" in lower:
        return True
    if re.fullmatch(r"\d{1,4}", line):
        return True
    return False


def looks_like_institution(line: str) -> bool:
    lower = line.lower()
    if any(keyword in lower for keyword in INSTITUTION_KEYWORDS):
        return True
    if re.match(r"^\d+\s*[A-Z]", line):
        return True
    return False


def looks_like_title(line: str) -> bool:
    lower = line.lower()
    words = re.findall(r"[A-Za-z][A-Za-z-]*", line)
    if len(words) >= 5 and sum(word.lower() in TITLE_WORDS for word in words) >= 2:
        return True
    if line.isupper() and len(words) >= 3:
        return True
    if ":" in line and len(words) >= 4:
        return True
    return False


def author_lines(lines: list[str]) -> list[str]:
    block: list[str] = []
    started = False
    for line in lines:
        if looks_like_metadata(line):
            continue
        if looks_like_institution(line):
            if started:
                break
            continue
        if looks_like_title(line):
            if not started:
                continue
        started = True
        block.append(line)
        if len(block) >= 8:
            break
    return block


def extract_authors(lines: list[str]) -> list[str]:
    joined = clean_affiliation_markers(" ".join(author_lines(lines)))
    if not joined:
        return []

    if "," in joined:
        raw_parts = re.split(r",| and ", joined)
        candidates = [normalize_space(part) for part in raw_parts]
    else:
        candidates = [normalize_space(match.group(0)) for match in NAME_PATTERN.finditer(joined)]

    results: list[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        words = candidate.split()
        lower = candidate.lower()
        if any(keyword in lower for keyword in INSTITUTION_KEYWORDS):
            continue
        if lower in NON_AUTHOR_PHRASES:
            continue
        if len(words) < 2:
            continue
        if sum(word.lower() in AUTHOR_BLOCKLIST_WORDS for word in words) >= 2:
            continue
        if len(words) == 4 and all(word[:1].isupper() for word in words):
            pair1 = " ".join(words[:2])
            pair2 = " ".join(words[2:])
            for pair in (pair1, pair2):
                if pair not in seen:
                    seen.add(pair)
                    results.append(pair)
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        results.append(candidate)
    return results


def normalize_institution(name: str) -> str:
    name = clean_affiliation_markers(name)
    name = re.sub(r"^\d+\s*", "", name)
    return name.strip(" ,.;")


def extract_institutions(lines: list[str]) -> list[str]:
    results: list[str] = []
    seen = set()
    for line in lines:
        if not looks_like_institution(line):
            continue
        cleaned = normalize_institution(line)
        if not cleaned:
            continue
        lower = cleaned.lower()
        if "@" in cleaned:
            continue
        if cleaned.lower() in NON_AUTHOR_PHRASES:
            continue
        if sum(word.lower() in TITLE_WORDS for word in cleaned.split()) >= 3:
            continue
        if len(cleaned.split()) > 14 and not any(keyword in lower for keyword in INSTITUTION_KEYWORDS):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        results.append(cleaned)
    return results


def build_source_meta(text_path: Path) -> SourceMeta:
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    lines = front_lines(text)
    stem = text_path.stem
    return SourceMeta(
        stem=stem,
        source_link=slug_link(stem),
        authors=extract_authors(lines),
        institutions=extract_institutions(lines),
    )


def top_items(mapping: dict[str, list[SourceMeta]], limit: int = 25) -> list[tuple[str, list[SourceMeta]]]:
    return sorted(
        mapping.items(),
        key=lambda item: (-len(item[1]), item[0].lower()),
    )[:limit]


def render_item_list(items: list[tuple[str, list[SourceMeta]]], label: str) -> list[str]:
    lines: list[str] = []
    for name, metas in items:
        examples = ", ".join(
            f"[{meta.stem}]({meta.source_link})" for meta in metas[:3]
        )
        lines.append(f"- {name}：{len(metas)} 篇。代表来源：{examples}")
    if not lines:
        lines.append("- 暂无条目。")
    return lines


def render_author_page(metas: list[SourceMeta]) -> str:
    author_map: dict[str, list[SourceMeta]] = defaultdict(list)
    inst_map: dict[str, list[SourceMeta]] = defaultdict(list)
    sources_with_authors = 0
    for meta in metas:
        if meta.authors:
            sources_with_authors += 1
        for author in meta.authors:
            author_map[author].append(meta)
        for inst in meta.institutions:
            inst_map[inst].append(meta)

    lines = [
        "# 作者分析",
        "",
        "## 范围",
        "",
        f"- 扫描来源：{len(metas)} 篇 `raw/text/*.md`",
        f"- 成功识别到作者的来源：{sources_with_authors} 篇",
        f"- 去重后作者条目：{len(author_map)} 个",
        f"- 去重后机构条目：{len(inst_map)} 个",
        "",
        "## 方法说明",
        "",
        "- 当前分析基于 `raw/text/` 中 PDF 首页附近的文本块做启发式抽取。",
        "- 作者名单优先从标题下方、`Abstract` 之前的名字行抽取。",
        "- PDF 排版差异较大，结果适合做知识库导航和粗分析，不应视为最终准确的人名库。",
        "",
        "## 高频作者",
        "",
        *render_item_list(top_items(author_map, 30), "作者"),
        "",
        "## 高频作者对应机构样本",
        "",
    ]

    for author, author_metas in top_items(author_map, 15):
        related_insts = sorted(
            {
                inst
                for meta in author_metas
                for inst in meta.institutions
                if "@" not in inst
            }
        )[:5]
        inst_text = "、".join(related_insts) if related_insts else "未稳定抽取到机构"
        lines.append(f"- {author}：机构样本 {inst_text}")

    lines.extend(
        [
            "",
            "## 相关页面",
            "",
            "- [机构分析](./机构分析.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def render_institution_page(metas: list[SourceMeta]) -> str:
    inst_map: dict[str, list[SourceMeta]] = defaultdict(list)
    sources_with_institutions = 0
    for meta in metas:
        if meta.institutions:
            sources_with_institutions += 1
        for inst in meta.institutions:
            inst_map[inst].append(meta)

    lines = [
        "# 机构分析",
        "",
        "## 范围",
        "",
        f"- 扫描来源：{len(metas)} 篇 `raw/text/*.md`",
        f"- 成功识别到机构的来源：{sources_with_institutions} 篇",
        f"- 去重后机构条目：{len(inst_map)} 个",
        "",
        "## 方法说明",
        "",
        "- 当前分析基于 `raw/text/` 中首页附近的机构行、团队名和邮箱行附近文本做启发式抽取。",
        "- 机构名会受到 PDF 断行、缩写和团队命名方式影响，后续应逐步人工归并。",
        "",
        "## 高频机构",
        "",
        *render_item_list(top_items(inst_map, 30), "机构"),
        "",
        "## 机构样本观察",
        "",
    ]

    for inst, inst_metas in top_items(inst_map, 15):
        author_samples = sorted(
            {
                author
                for meta in inst_metas
                for author in meta.authors[:4]
            }
        )[:5]
        author_text = "、".join(author_samples) if author_samples else "未稳定抽取到作者"
        lines.append(f"- {inst}：作者样本 {author_text}")

    lines.extend(
        [
            "",
            "## 相关页面",
            "",
            "- [作者分析](./作者分析.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    metas = [build_source_meta(path) for path in sorted(TEXT.glob("*.md"))]
    AUTHORS_DIR.mkdir(parents=True, exist_ok=True)
    AUTHOR_PAGE.write_text(render_author_page(metas), encoding="utf-8")
    INSTITUTION_PAGE.write_text(render_institution_page(metas), encoding="utf-8")
    print(f"Analyzed {len(metas)} sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
