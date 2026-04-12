#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests


BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "ul",
    "ol",
    "pre",
    "blockquote",
    "table",
    "tr",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_TAGS = {"script", "style", "noscript", "svg"}


def normalize_whitespace(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ArticleExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.capture_stack: list[str] = []
        self.current_tag: str | None = None
        self.current_attrs: dict[str, str] = {}
        self.current_text: list[str] = []
        self.blocks: list[str] = []
        self.title: str | None = None
        self._title_text: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: (v or "") for k, v in attrs}
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return

        if tag == "title":
            self._title_text = []

        if self.skip_depth:
            return

        if tag in HEADING_TAGS or tag in BLOCK_TAGS:
            self._flush_current()
            self.current_tag = tag
            self.current_attrs = attrs_dict
            self.current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return

        if tag == "title":
            title = normalize_whitespace("".join(self._title_text or []))
            if title:
                self.title = title
            self._title_text = None
            return

        if self.skip_depth:
            return

        if self.current_tag == tag:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self._title_text is not None:
            self._title_text.append(data)
        if self.current_tag:
            self.current_text.append(data)

    def _flush_current(self) -> None:
        if not self.current_tag:
            return

        text = normalize_whitespace("".join(self.current_text))
        tag = self.current_tag
        attrs = self.current_attrs

        if text:
            if tag in HEADING_TAGS:
                level = min(max(int(tag[1]), 1), 6)
                self.blocks.append(f'{"#" * level} {text}')
            elif tag == "li":
                self.blocks.append(f"- {text}")
            elif tag == "pre":
                self.blocks.append(f"```\n{text}\n```")
            else:
                classes = f"{attrs.get('class', '')} {attrs.get('id', '')}".lower()
                if not any(
                    bad in classes
                    for bad in ("nav", "menu", "footer", "header", "sidebar", "toc", "breadcrumb")
                ):
                    self.blocks.append(text)

        self.current_tag = None
        self.current_attrs = {}
        self.current_text = []

    def render(self) -> str:
        self._flush_current()

        deduped: list[str] = []
        prev = None
        for block in self.blocks:
            block = block.strip()
            if not block:
                continue
            if block == prev:
                continue
            deduped.append(block)
            prev = block

        return "\n\n".join(deduped).strip()


def render_markdown(
    title: str,
    url: str,
    body: str,
    html_path: str | None = None,
) -> str:
    lines = [f"# {title}", ""]
    if html_path:
        lines.append(f"- Source HTML: `{html_path}`")
    lines.append(f"- Source URL: {url}")
    lines.append("- Generated from: `scripts/fetch_web_text.py`")
    lines.extend(["", "## Extracted Text", "", body, ""])
    return "\n".join(lines)


def extract_jsonld_article(html_text: str) -> tuple[str | None, str | None]:
    field_title = None
    headline_match = re.search(r'"headline":"(.*?)"', html_text, flags=re.DOTALL)
    if headline_match:
        try:
            field_title = json.loads(f'"{headline_match.group(1)}"')
        except json.JSONDecodeError:
            field_title = normalize_whitespace(headline_match.group(1))

    article_match = re.search(r'"articleBody":"(.*?)","wordCount"', html_text, flags=re.DOTALL)
    if article_match:
        try:
            article_body = json.loads(f'"{article_match.group(1)}"')
        except json.JSONDecodeError:
            article_body = article_match.group(1)
        article_body = normalize_whitespace(article_body)
        if article_body:
            return field_title, article_body

    matches = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for raw in matches:
        try:
            data = json.loads(html.unescape(raw.strip()))
        except json.JSONDecodeError:
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            article_body = item.get("articleBody")
            title = item.get("headline") or item.get("name")
            if isinstance(article_body, str) and article_body.strip():
                return title if isinstance(title, str) else None, normalize_whitespace(article_body)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a webpage and save extracted text as markdown.")
    parser.add_argument("url", help="Source URL to fetch.")
    parser.add_argument("output", help="Output markdown path.")
    parser.add_argument("--title", help="Override output title.")
    parser.add_argument("--html-out", help="Optional path to save the raw HTML response.")
    args = parser.parse_args()

    response = requests.get(
        args.url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; wiki-ingest/1.0)"},
    )
    response.raise_for_status()

    jsonld_title, jsonld_body = extract_jsonld_article(response.text)

    parser_ = ArticleExtractor()
    parser_.feed(response.text)
    body = jsonld_body or parser_.render()
    if not body:
        print(f"Failed to extract body from {args.url}", file=sys.stderr)
        return 1

    title = args.title or jsonld_title or parser_.title or Path(urlparse(args.url).path).name or "Untitled"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.html_out:
        html_path = Path(args.html_out)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(response.text, encoding="utf-8")
        print(f"Wrote {html_path}")
    out_path.write_text(
        render_markdown(title, args.url, body, args.html_out),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
