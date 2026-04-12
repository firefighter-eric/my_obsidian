#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote_plus

import fitz
import requests

from download_arxiv import fetch_html_markdown


ROOT = Path(__file__).resolve().parent.parent
PDF_ROOT = ROOT / "raw" / "pdfs"
HTML_ROOT = ROOT / "raw" / "html"
TEXT_ROOT = ROOT / "raw" / "text"

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "Mozilla/5.0 (compatible; wiki-ingest/1.0)"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
GENERIC_TITLES = {"intro", "introduction", "images"}
TITLE_OVERRIDES = {
    "deepseek r1 incentivizing reasoning capability in llms via reinforcement learning": "2501.12948",
    "deepseek v3 technical report": "2412.19437",
    "detrs with collaborative hybrid assignments training 2": "2211.12860",
}

FILENAME_ID_PATTERNS = [
    re.compile(r"\barxiv[\s._-]*(\d{4}\s*\.\s*\d{4,5})(?:v\d+)?\b", re.I),
    re.compile(r"\b(\d{4}\s*\.\s*\d{4,5})(?:v\d+)?\b"),
]
TEXT_ID_PATTERNS = [
    re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5})(?:v\d+)?", re.I),
    re.compile(r"arXiv[:\s]+([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I),
]
TITLE_SPLIT = re.compile(r"^[^-]+ - (?:\d{4}|Unknown) - (.+)$")


@dataclass
class ScanResult:
    stem: str
    arxiv_id: str | None
    method: str
    status: str
    note: str = ""


def normalize_arxiv_id(raw: str) -> str:
    return re.sub(r"\s+", "", raw).rstrip(".")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def title_from_stem(stem: str) -> str:
    match = TITLE_SPLIT.match(stem)
    return match.group(1).strip() if match else stem


def title_is_searchable(title: str) -> bool:
    normalized = normalize_text(title)
    if normalized in TITLE_OVERRIDES:
        return True
    if not normalized or normalized in GENERIC_TITLES:
        return False
    tokens = normalized.split()
    return len(tokens) >= 3 and len(normalized) >= 15


def text_has_html_header(text_path: Path) -> bool:
    if not text_path.exists():
        return False
    head = text_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:6]
    return any(line.startswith("- Source HTML:") for line in head)


def extract_id_from_filename(stem: str) -> str | None:
    for pattern in FILENAME_ID_PATTERNS:
        match = pattern.search(stem)
        if match:
            return normalize_arxiv_id(match.group(1))
    return None


def extract_id_from_pdf(pdf_path: Path, pages: int = 2) -> str | None:
    doc = fitz.open(pdf_path)
    try:
        text_parts = []
        for page_no in range(min(pages, doc.page_count)):
            text_parts.append(doc.load_page(page_no).get_text())
    finally:
        doc.close()

    text = "\n".join(text_parts)
    for pattern in TEXT_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_arxiv_id(match.group(1))
    return None


def choose_title_match(local_title: str, candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    normalized_local = normalize_text(local_title)
    if not normalized_local:
        return None

    best: tuple[float, tuple[str, str]] | None = None
    for arxiv_id, remote_title in candidates:
        normalized_remote = normalize_text(remote_title)
        if not normalized_remote:
            continue
        ratio = SequenceMatcher(None, normalized_local, normalized_remote).ratio()
        if normalized_local == normalized_remote:
            ratio = 1.0
        elif normalized_local in normalized_remote or normalized_remote in normalized_local:
            ratio = max(ratio, 0.95)

        if best is None or ratio > best[0]:
            best = (ratio, (arxiv_id, remote_title))

    if best is None:
        return None

    ratio, choice = best
    if ratio < 0.88:
        return None
    return choice


def search_arxiv_by_title(title: str) -> tuple[str, str] | None:
    query = quote_plus(f'ti:"{title}"')
    url = f"{ARXIV_API}?search_query={query}&start=0&max_results=5"

    root = None
    for delay in (0, 2, 5):
        if delay:
            time.sleep(delay)
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            root = ET.fromstring(response.text)
            break
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                return None
        except ET.ParseError:
            continue
        except Exception as exc:  # noqa: BLE001
            return None

    if root is None:
        return None

    candidates: list[tuple[str, str]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        id_text = entry.findtext("atom:id", default="", namespaces=ARXIV_NS).strip()
        title_text = entry.findtext("atom:title", default="", namespaces=ARXIV_NS).strip()
        match = re.search(r"/abs/([^/?#]+)$", id_text)
        if not match or not title_text:
            continue
        candidates.append((match.group(1), title_text))

    return choose_title_match(title, candidates)


def scan_pdf(pdf_path: Path, force: bool) -> ScanResult:
    stem = pdf_path.stem
    html_path = HTML_ROOT / f"{stem}.html"
    text_path = TEXT_ROOT / f"{stem}.md"

    if not force and html_path.exists() and text_has_html_header(text_path):
        return ScanResult(stem=stem, arxiv_id=None, method="skip", status="already_html")

    arxiv_id = extract_id_from_filename(stem)
    method = "filename"

    if arxiv_id is None:
        arxiv_id = extract_id_from_pdf(pdf_path)
        method = "pdf"

    if arxiv_id is None:
        title = title_from_stem(stem)
        normalized_title = normalize_text(title)
        if normalized_title in TITLE_OVERRIDES:
            arxiv_id = TITLE_OVERRIDES[normalized_title]
            return ScanResult(
                stem=stem,
                arxiv_id=arxiv_id,
                method="override",
                status="matched",
            )
        if not title_is_searchable(title):
            return ScanResult(stem=stem, arxiv_id=None, method="title", status="no_match")
        match = search_arxiv_by_title(title)
        if match is None:
            return ScanResult(stem=stem, arxiv_id=None, method="title", status="no_match")
        arxiv_id, matched_title = match
        return ScanResult(
            stem=stem,
            arxiv_id=arxiv_id,
            method="title",
            status="matched",
            note=f"title match: {matched_title}",
        )

    return ScanResult(stem=stem, arxiv_id=arxiv_id, method=method, status="matched")


def import_html(result: ScanResult, force: bool) -> None:
    assert result.arxiv_id
    html_path = HTML_ROOT / f"{result.stem}.html"
    text_path = TEXT_ROOT / f"{result.stem}.md"
    fetch_html_markdown(
        arxiv_id=result.arxiv_id,
        html_path=html_path,
        text_path=text_path,
        title=result.stem,
        force=force,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan raw/pdfs for arXiv papers and import corresponding arXiv HTML into raw/html and raw/text."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="Optional list of PDF paths. Defaults to all PDFs under raw/pdfs/.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing HTML/text outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Only print matches without importing.")
    args = parser.parse_args()

    pdf_paths = [Path(path).resolve() for path in args.pdfs] if args.pdfs else sorted(PDF_ROOT.glob("*.pdf"))
    if not pdf_paths:
        print("No PDF files found.")
        return 1

    matched: list[ScanResult] = []
    skipped = 0
    no_match = 0
    imported = 0
    failed = 0

    for pdf_path in pdf_paths:
        result = scan_pdf(pdf_path, force=args.force)
        if result.status == "already_html":
            skipped += 1
            print(f"SKIP {result.stem}: already backed by HTML")
            continue
        if result.status == "no_match":
            no_match += 1
            print(f"MISS {result.stem}: no reliable arXiv match")
            continue

        matched.append(result)
        suffix = f" ({result.note})" if result.note else ""
        print(f"MATCH {result.stem}: {result.arxiv_id} via {result.method}{suffix}")
        if args.dry_run:
            continue

        try:
            import_html(result, force=args.force)
            imported += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {result.stem}: {exc}")

    print(
        f"Summary: total={len(pdf_paths)} matched={len(matched)} imported={imported} "
        f"skipped={skipped} no_match={no_match} failed={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
