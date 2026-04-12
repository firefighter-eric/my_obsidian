#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import requests


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    patterns = [
        r"arxiv\.org/(?:abs|pdf|html)/([^/?#]+)",
        r"ar5iv\.labs\.arxiv\.org/html/([^/?#]+)",
        r"^arXiv:([^ ]+)$",
        r"^([^ ]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)
            if arxiv_id.endswith(".pdf"):
                arxiv_id = arxiv_id[:-4]
            return arxiv_id
    raise ValueError(f"Unsupported arXiv identifier: {value}")


def load_fetch_web_text_module():
    module_path = Path(__file__).with_name("fetch_web_text.py")
    spec = importlib.util.spec_from_file_location("fetch_web_text", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_file(url: str, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"Skip existing {out_path}")
        return

    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; wiki-ingest/1.0)"},
    )
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    print(f"Wrote {out_path}")


def fetch_html_markdown(
    arxiv_id: str,
    html_path: Path,
    text_path: Path,
    title: str | None,
    force: bool,
) -> None:
    if html_path.exists() and text_path.exists() and not force:
        print(f"Skip existing {html_path}")
        print(f"Skip existing {text_path}")
        return

    module = load_fetch_web_text_module()
    html_urls = [
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}",
    ]

    last_error: Exception | None = None
    for url in html_urls:
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (compatible; wiki-ingest/1.0)"},
            )
            response.raise_for_status()
            jsonld_title, jsonld_body = module.extract_jsonld_article(response.text)
            parser = module.ArticleExtractor()
            parser.feed(response.text)
            body = jsonld_body or parser.render()
            if not body:
                raise RuntimeError(f"Empty extracted body from {url}")

            final_title = title or jsonld_title or parser.title or arxiv_id
            html_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(response.text, encoding="utf-8")
            text_path.write_text(
                module.render_markdown(final_title, url, body, str(html_path)),
                encoding="utf-8",
            )
            print(f"Wrote {html_path}")
            print(f"Wrote {text_path}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Failed to fetch arXiv HTML for {arxiv_id}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download arXiv sources into raw/pdfs/, raw/html/, and raw/text/."
    )
    parser.add_argument("arxiv_id", help="arXiv id, abs/pdf/html URL, or arXiv:ID.")
    parser.add_argument("--stem", help="Output file stem. Defaults to the arXiv id.")
    parser.add_argument("--title", help="Markdown title for raw/text output.")
    parser.add_argument("--pdf-root", default="raw/pdfs", help="Output root for downloaded PDFs.")
    parser.add_argument("--html-root", default="raw/html", help="Output root for saved HTML files.")
    parser.add_argument("--text-root", default="raw/text", help="Output root for generated markdown.")
    parser.add_argument("--skip-pdf", action="store_true", help="Do not download the PDF.")
    parser.add_argument("--skip-text", action="store_true", help="Do not fetch arXiv HTML into markdown.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing targets.")
    args = parser.parse_args()

    arxiv_id = normalize_arxiv_id(args.arxiv_id)
    stem = args.stem or arxiv_id

    if args.skip_pdf and args.skip_text:
        print("Nothing to do: both --skip-pdf and --skip-text are set.", file=sys.stderr)
        return 1

    if not args.skip_pdf:
        pdf_path = Path(args.pdf_root) / f"{stem}.pdf"
        download_file(f"https://arxiv.org/pdf/{arxiv_id}.pdf", pdf_path, args.force)

    if not args.skip_text:
        html_path = Path(args.html_root) / f"{stem}.html"
        text_path = Path(args.text_root) / f"{stem}.md"
        fetch_html_markdown(arxiv_id, html_path, text_path, args.title, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
