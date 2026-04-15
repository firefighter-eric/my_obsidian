#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CollectionSpec:
    slug: str
    path: Path
    mask: str
    layer: str
    label: str
    priority: int


COLLECTION_SPECS = (
    CollectionSpec("index", ROOT, "index.md", "index", "索引层", 0),
    CollectionSpec("wiki", ROOT / "wiki", "**/*.md", "wiki", "知识层", 1),
    CollectionSpec("raw-text", ROOT / "raw" / "text", "**/*.md", "raw-text", "全文补查层", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search the local Obsidian wiki through qmd. "
            "Default mode prioritizes index.md and wiki/ over raw/text/."
        )
    )
    parser.add_argument("query", help="Query text.")
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of merged results to print. Default: 8.",
    )
    parser.add_argument(
        "--mode",
        choices=("wiki-first", "fulltext"),
        default="wiki-first",
        help="wiki-first prioritizes wiki pages; fulltext allows raw/text to surface more aggressively.",
    )
    parser.add_argument(
        "--per-collection-limit",
        type=int,
        default=None,
        help="Override qmd result count for each collection search.",
    )
    return parser.parse_args()


def repo_prefix() -> str:
    digest = hashlib.sha1(str(ROOT).encode("utf-8")).hexdigest()[:8]
    return f"my-obsidian-{digest}"


def collection_name(spec: CollectionSpec) -> str:
    return f"{repo_prefix()}-{spec.slug}"


def normalize_path(value: str) -> str:
    text = value.strip()
    if text.startswith("qmd://"):
        text = text[len("qmd://") :]
        parts = text.split("/", 1)
        text = parts[1] if len(parts) == 2 else ""
    elif text.startswith(str(ROOT)):
        try:
            text = str(Path(text).resolve().relative_to(ROOT))
        except Exception:  # noqa: BLE001
            pass
    return text.lstrip("/")


def short_text(value: str, limit: int = 140) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def query_terms(query: str) -> list[str]:
    text = query.strip().lower()
    if not text:
        return []
    if re.search(r"[\u4e00-\u9fff]", text):
        return [text]
    terms = [part for part in re.split(r"[^a-z0-9]+", text) if len(part) >= 3]
    return terms or [text]


def matches_query_text(query: str, result: dict[str, Any]) -> bool:
    terms = query_terms(query)
    if not terms:
        return True
    haystack = " ".join(
        [
            result.get("path", ""),
            result.get("title", ""),
            result.get("snippet", ""),
        ]
    ).lower()
    matched = sum(1 for term in terms if term in haystack)
    required = len(terms) if len(terms) <= 2 else 2
    return matched >= required


def installation_error() -> str:
    return "\n".join(
        [
            "qmd CLI not found.",
            "Install it with one of the official commands:",
            "  npm install -g @tobilu/qmd",
            "  bun install -g @tobilu/qmd",
            "",
            "If qmd later reports SQLite extension issues on macOS, install Homebrew sqlite:",
            "  brew install sqlite",
        ]
    )


def subtype_priority(path: str) -> int:
    if path == "index.md":
        return 0
    order = (
        "wiki/topics/",
        "wiki/concepts/",
        "wiki/comparisons/",
        "wiki/timelines/",
        "wiki/summaries/",
        "wiki/authors/",
        "raw/text/",
    )
    for index, prefix in enumerate(order, start=1):
        if path.startswith(prefix):
            return index
    return len(order) + 1


def run_qmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    qmd = shutil.which("qmd")
    if not qmd:
        raise FileNotFoundError(installation_error())
    return subprocess.run(
        [qmd, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def ensure_collection(spec: CollectionSpec) -> None:
    name = collection_name(spec)
    probe = run_qmd(["ls", name], check=False)
    if probe.returncode == 0:
        return

    add = run_qmd(
        [
            "collection",
            "add",
            str(spec.path),
            "--name",
            name,
            "--mask",
            spec.mask,
        ],
        check=False,
    )
    if add.returncode != 0:
        reprobe = run_qmd(["ls", name], check=False)
        if reprobe.returncode == 0:
            return
        stderr = add.stderr.strip()
        stdout = add.stdout.strip()
        message = stderr or stdout or "unknown qmd error"
        raise RuntimeError(f"Failed to initialize qmd collection '{name}': {message}")


def load_title_index() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for base in (ROOT / "wiki", ROOT / "raw" / "text"):
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:  # noqa: BLE001
                continue
            title = ""
            for line in lines[:8]:
                stripped = line.strip()
                if stripped.startswith("# "):
                    title = stripped[2:].strip()
                    break
            if title:
                mapping.setdefault(title, []).append(path.relative_to(ROOT).as_posix())

    mapping.setdefault("Wiki Index", []).append("index.md")
    return mapping


def resolve_real_path(
    item: dict[str, Any],
    layer: str,
    title_index: dict[str, list[str]],
) -> str:
    title = pick_first(item, ("title", "name"))
    if title:
        candidates = title_index.get(title, [])
        if layer == "wiki":
            preferred = [path for path in candidates if path.startswith("wiki/")]
            if preferred:
                return preferred[0]
        if layer == "raw-text":
            preferred = [path for path in candidates if path.startswith("raw/text/")]
            if preferred:
                return preferred[0]
        if layer == "index" and "index.md" in candidates:
            return "index.md"
        if candidates:
            return candidates[0]

    return normalize_path(pick_first(item, ("path", "file", "filepath", "id", "docid")))


def extract_result_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "matches", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise RuntimeError("Unsupported qmd JSON output shape.")


def pick_first(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pick_score(item: dict[str, Any]) -> float:
    for key in ("score", "finalScore", "rerankScore"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def search_collection(
    spec: CollectionSpec,
    query: str,
    limit: int,
    title_index: dict[str, list[str]],
) -> list[dict[str, Any]]:
    name = collection_name(spec)
    completed = run_qmd(
        [
            "search",
            query,
            "-c",
            name,
            "--json",
            "-n",
            str(limit),
        ]
    )
    payload = json.loads(completed.stdout)
    results = []
    for item in extract_result_list(payload):
        title_value = pick_first(item, ("title", "name"))
        snippet_value = pick_first(item, ("snippet", "excerpt", "text", "preview"))
        docid_value = pick_first(item, ("docid", "id"))
        results.append(
            {
                "path": resolve_real_path(item, spec.layer, title_index),
                "title": title_value,
                "snippet": snippet_value,
                "docid": docid_value,
                "score": pick_score(item),
                "layer": spec.layer,
                "label": spec.label,
                "priority": spec.priority,
            }
        )
    return results


def merged_results(mode: str, query: str, per_collection_limit: int) -> list[dict[str, Any]]:
    title_index = load_title_index()
    all_results: list[dict[str, Any]] = []
    for spec in COLLECTION_SPECS:
        all_results.extend(search_collection(spec, query, per_collection_limit, title_index))

    deduped: dict[str, dict[str, Any]] = {}
    for result in all_results:
        if not matches_query_text(query, result):
            continue
        key = result["path"] or f"{result['layer']}::{result['title']}"
        previous = deduped.get(key)
        if previous is None or result["score"] > previous["score"]:
            deduped[key] = result

    results = list(deduped.values())
    if mode == "wiki-first":
        results.sort(
            key=lambda item: (
                item["priority"],
                subtype_priority(item["path"]),
                -item["score"],
                item["path"],
            )
        )
    else:
        results.sort(
            key=lambda item: (
                -item["score"],
                item["priority"],
                subtype_priority(item["path"]),
                item["path"],
            )
        )
    return results


def print_results(query: str, mode: str, results: list[dict[str, Any]], limit: int) -> None:
    print(f"# Search: {query}")
    print(f"- Mode: {mode}")
    print(f"- Root: {ROOT}")
    if not results:
        print("- Results: 0")
        return

    print(f"- Results: {min(limit, len(results))}")
    print("")
    for index, item in enumerate(results[:limit], start=1):
        path_text = item["path"] or "(unknown path)"
        title_text = item["title"] or Path(path_text).stem
        print(f"{index}. [{item['layer']}] {path_text}")
        print(f"   标题: {short_text(title_text, 100)}")
        if item["snippet"]:
            print(f"   摘要: {short_text(item['snippet'])}")
        print(f"   层级: {item['label']} | 分数: {item['score']:.3f}")
        if item["docid"]:
            print(f"   qmd: {item['docid']}")


def main() -> int:
    args = parse_args()
    try:
        for spec in COLLECTION_SPECS:
            ensure_collection(spec)

        per_collection_limit = args.per_collection_limit or max(args.limit * 3, 12)
        results = merged_results(args.mode, args.query, per_collection_limit)
        print_results(args.query, args.mode, results, args.limit)
        return 0
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        message = stderr or stdout or str(exc)
        print(f"qmd command failed: {message}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Failed to parse qmd JSON output: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
