#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import fitz


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts).strip()


def build_output_path(pdf_path: Path, raw_root: Path, out_root: Path) -> Path:
    rel = pdf_path.relative_to(raw_root)
    return out_root / rel.with_suffix(".md")


def render_markdown(pdf_path: Path, raw_root: Path, extracted: str) -> str:
    rel = pdf_path.relative_to(raw_root)
    title = pdf_path.stem
    return (
        f"# {title}\n\n"
        f"- Source PDF: `raw/pdfs/{rel.as_posix()}`\n"
        f"- Generated from: `scripts/extract_pdf_text.py`\n\n"
        "## Extracted Text\n\n"
        f"{extracted}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs in raw/pdfs/ into markdown files in raw/text/."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="Optional list of PDF paths. Defaults to all PDFs under raw/pdfs/.",
    )
    parser.add_argument(
        "--raw-root",
        default="raw/pdfs",
        help="Root directory containing source PDFs.",
    )
    parser.add_argument(
        "--out-root",
        default="raw/text",
        help="Root directory for extracted markdown files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing markdown files in raw/text/.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    if args.pdfs:
        pdf_paths = [Path(p).resolve() for p in args.pdfs]
    else:
        pdf_paths = sorted(raw_root.rglob("*.pdf"))

    if not pdf_paths:
        print("No PDF files found.", file=sys.stderr)
        return 1

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"Missing PDF: {pdf_path}", file=sys.stderr)
            return 1

        out_path = build_output_path(pdf_path, raw_root, out_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.force:
            print(f"Skip existing {out_path}")
            continue

        extracted = extract_pdf_text(pdf_path)
        markdown = render_markdown(pdf_path, raw_root, extracted)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
