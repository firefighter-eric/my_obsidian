#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse

import requests


SCHOLAR_BASE = "https://scholar.google.com/scholar"
USER_AGENT = "Mozilla/5.0 (compatible; wiki-ingest/1.0)"


@dataclass
class ScholarResult:
    title: str
    result_url: str | None
    pdf_url: str | None
    meta: str | None
    snippet: str | None
    cited_by: int | None
    cited_by_url: str | None
    versions: int | None
    versions_url: str | None


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[-\s]+", "-", value)
    return value.strip("-") or "google-scholar-query"


def strip_tracking_href(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("&amp;", "&")
    return value


def extract_anchor(block: str) -> tuple[str | None, str | None]:
    match = re.search(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.DOTALL)
    if not match:
        return None, None
    return strip_tracking_href(match.group(1)), clean_text(match.group(2))


def extract_first_count(label: str) -> int | None:
    match = re.search(r"([\d,]+)", label)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def split_result_blocks(html_text: str) -> list[str]:
    marker = '<div class="gs_r gs_or gs_scl"'
    starts = [m.start() for m in re.finditer(re.escape(marker), html_text)]
    blocks: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(html_text)
        blocks.append(html_text[start:end])
    return blocks


def parse_result_block(block: str) -> ScholarResult | None:
    title_match = re.search(r'<h3 class="gs_rt".*?>(.*?)</h3>', block, flags=re.DOTALL)
    if not title_match:
        return None

    title_html = title_match.group(1)
    result_url, title = extract_anchor(title_html)
    if title is None:
        title = clean_text(title_html)
    title = re.sub(r"^\[PDF\]\s*", "", title).strip()
    if not title:
        return None

    pdf_url = None
    pdf_match = re.search(
        r'<div class="gs_or_ggsm".*?<a [^>]*href="([^"]+)"',
        block,
        flags=re.DOTALL,
    )
    if pdf_match:
        pdf_url = strip_tracking_href(pdf_match.group(1))

    meta = None
    meta_match = re.search(r'<div class="gs_a">(.*?)</div>', block, flags=re.DOTALL)
    if meta_match:
        meta = clean_text(meta_match.group(1))

    snippet = None
    snippet_match = re.search(r'<div class="gs_rs">(.*?)</div>', block, flags=re.DOTALL)
    if snippet_match:
        snippet = clean_text(snippet_match.group(1))

    cited_by = None
    cited_by_url = None
    versions = None
    versions_url = None
    for href, label in re.findall(r'<a href="([^"]+)".*?>(.*?)</a>', block, flags=re.DOTALL):
        clean_label = clean_text(label)
        abs_href = urljoin("https://scholar.google.com", strip_tracking_href(href))
        if clean_label.startswith("被引用") or clean_label.lower().startswith("cited by"):
            cited_by = extract_first_count(clean_label)
            cited_by_url = abs_href
        if "版本" in clean_label or "versions" in clean_label.lower():
            versions = extract_first_count(clean_label)
            versions_url = abs_href

    return ScholarResult(
        title=title,
        result_url=result_url,
        pdf_url=pdf_url,
        meta=meta,
        snippet=snippet,
        cited_by=cited_by,
        cited_by_url=cited_by_url,
        versions=versions,
        versions_url=versions_url,
    )


def parse_scholar_results(html_text: str) -> list[ScholarResult]:
    results: list[ScholarResult] = []
    for block in split_result_blocks(html_text):
        result = parse_result_block(block)
        if result is not None:
            results.append(result)
    return results


def build_query_url(
    query: str,
    start: int,
    year_from: int | None,
    year_to: int | None,
    language: str,
    include_patents: bool,
    include_citations: bool,
) -> str:
    params = {
        "q": query,
        "hl": language,
        "start": start,
        "as_vis": "0" if include_citations else "1",
        "as_sdt": "0,5",
    }
    if year_from:
        params["as_ylo"] = str(year_from)
    if year_to:
        params["as_yhi"] = str(year_to)
    if include_patents:
        params["as_sdt"] = "0,33"
    return f"{SCHOLAR_BASE}?{urlencode(params)}"


def fetch_html(url: str) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def detect_block_page(html_text: str) -> str | None:
    text = clean_text(html_text[:4000]).lower()
    for marker in ("unusual traffic", "not a robot", "sorry", "captcha"):
        if marker in text:
            return marker
    return None


def render_markdown(query: str, url: str, html_path: Path, results: list[ScholarResult]) -> str:
    fetched_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        f"# Google Scholar Search - {query}",
        "",
        f"- Query: `{query}`",
        f"- Source URL: {url}",
        f"- Source HTML: `{html_path.as_posix()}`",
        "- Generated from: `scripts/fetch_google_scholar.py`",
        f"- Fetched at: `{fetched_at}`",
        f"- Result count: `{len(results)}`",
        "",
        "## Results",
        "",
    ]
    if not results:
        lines.append("- No parsable results.")
        lines.append("")
        return "\n".join(lines)

    for idx, result in enumerate(results, start=1):
        lines.append(f"### {idx}. {result.title}")
        lines.append("")
        if result.result_url:
            lines.append(f"- Result URL: {result.result_url}")
        if result.pdf_url:
            lines.append(f"- PDF URL: {result.pdf_url}")
        if result.meta:
            lines.append(f"- Meta: {result.meta}")
        if result.cited_by is not None:
            lines.append(f"- Cited By: `{result.cited_by}`")
        if result.cited_by_url:
            lines.append(f"- Cited By URL: {result.cited_by_url}")
        if result.versions is not None:
            lines.append(f"- Versions: `{result.versions}`")
        if result.versions_url:
            lines.append(f"- Versions URL: {result.versions_url}")
        if result.snippet:
            lines.append(f"- Snippet: {result.snippet}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a Google Scholar search result page into raw/html and raw/text."
    )
    parser.add_argument("query", help="Scholar query string.")
    parser.add_argument("--start", type=int, default=0, help="Result offset, typically multiples of 10.")
    parser.add_argument("--year-from", type=int, help="Lower publication year filter.")
    parser.add_argument("--year-to", type=int, help="Upper publication year filter.")
    parser.add_argument("--hl", default="en", help="Scholar UI language, e.g. en or zh-CN.")
    parser.add_argument(
        "--include-patents",
        action="store_true",
        help="Include patent results. Default is to exclude them.",
    )
    parser.add_argument(
        "--include-citations",
        action="store_true",
        help="Include citation-only records. Default is to exclude them.",
    )
    parser.add_argument(
        "--html-root",
        default="raw/html/google_scholar",
        help="Output root for saved Scholar HTML.",
    )
    parser.add_argument(
        "--text-root",
        default="raw/text/google_scholar",
        help="Output root for parsed Scholar markdown.",
    )
    parser.add_argument(
        "--stem",
        help="Optional output stem. Defaults to a slugified query plus result offset.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing targets.")
    args = parser.parse_args()

    if args.start < 0:
        print("--start must be >= 0", file=sys.stderr)
        return 1

    url = build_query_url(
        query=args.query,
        start=args.start,
        year_from=args.year_from,
        year_to=args.year_to,
        language=args.hl,
        include_patents=args.include_patents,
        include_citations=args.include_citations,
    )
    stem = args.stem or f"{slugify(args.query)}-start-{args.start}"
    html_path = Path(args.html_root) / f"{stem}.html"
    text_path = Path(args.text_root) / f"{stem}.md"

    if html_path.exists() and text_path.exists() and not args.force:
        print(f"Skip existing {html_path}")
        print(f"Skip existing {text_path}")
        return 0

    html_text = fetch_html(url)
    blocked = detect_block_page(html_text)
    if blocked:
        print(
            f"Google Scholar returned a block/check page ({blocked}); retry later or from a browser session.",
            file=sys.stderr,
        )
        return 2

    results = parse_scholar_results(html_text)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    text_path.write_text(render_markdown(args.query, url, html_path, results), encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Wrote {text_path}")
    print(f"Parsed {len(results)} results from {urlparse(url).netloc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
