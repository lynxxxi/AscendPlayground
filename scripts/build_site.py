#!/usr/bin/env python3
"""Build the static AscendPlayground site into _site/."""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_site"
CATEGORIES = (
    ("features", "特性研究", 1),
    ("frameworks", "框架分析", 2),
    ("models", "模型研究", 3),
    ("operators", "算子分析", 4),
    ("work", "工作记录", 5),
)


def compact_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def markdown_metadata(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title_match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    description = ""
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith(("#", "|", ">", "- ", "* ", "![")):
            continue
        description = compact_text(re.sub(r"[`*_\[\]]", "", line))
        if description:
            break
    return title, description


def html_metadata(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if not title_match:
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.IGNORECASE | re.DOTALL)
    title = compact_text(title_match.group(1)) if title_match else path.stem
    description_match = re.search(
        r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not description_match:
        description_match = re.search(r"<p[^>]*>(.*?)</p>", text, re.IGNORECASE | re.DOTALL)
    description = compact_text(description_match.group(1)) if description_match else ""
    return title, description


def git_date(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def scan_documents() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for category, label, order in CATEGORIES:
        directory = ROOT / category
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in {".html", ".md"}:
                continue
            title, description = markdown_metadata(path) if suffix == ".md" else html_metadata(path)
            entries.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "title": title,
                    "description": description or "技术文档与研究记录。",
                    "category": category,
                    "categoryLabel": label,
                    "categoryOrder": order,
                    "type": "markdown" if suffix == ".md" else "html",
                    "date": git_date(path),
                }
            )
    return sorted(entries, key=lambda item: (item["categoryOrder"], item["title"]))


def build() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    for filename in ("index.html", "LICENSE"):
        source = ROOT / filename
        if source.exists():
            shutil.copy2(source, OUTPUT / filename)
    for directory, _, _ in CATEGORIES:
        source = ROOT / directory
        if source.is_dir():
            shutil.copytree(source, OUTPUT / directory)
    entries = scan_documents()
    manifest = {"site": "AscendPlayground", "contentCount": len(entries), "entries": entries}
    (OUTPUT / "site-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / ".nojekyll").touch()
    print(f"Built {len(entries)} reports into {OUTPUT}")


if __name__ == "__main__":
    build()
