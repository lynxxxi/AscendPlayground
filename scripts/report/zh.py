#!/usr/bin/env python3
"""中文说明层：为条目提供人工撰写的中文标题与说明。

数据文件：`config/curated_zh.json`
    按 stableId 精确匹配 `{title, digest}`；未命中的条目保留自动抽取的原文说明。

这样报告主体是中文，且同一份快照重跑结果完全一致（纯本地映射，无外部服务）。
新条目首次出现时若还没有中文说明，会以原文占位，便于发现待补写的条目。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .lib.util import title_key

DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "curated_zh.json"
DEFAULT_RELEASES_PATH = Path(__file__).resolve().parent / "config" / "curated_releases_zh.json"


class CuratedReleases:
    """按 `repo -> tag -> 中文特性说明` 匹配的发版说明层。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_RELEASES_PATH
        self.repos: dict[str, dict[str, str]] = {}
        self.hits = 0
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        for repo, tags in payload.items():
            if repo == "meta" or not isinstance(tags, dict):
                continue
            self.repos[repo] = {str(tag): str(text).strip() for tag, text in tags.items()}

    def lookup(self, repo: str, tag: str) -> str:
        value = (self.repos.get(repo) or {}).get(str(tag))
        if value:
            self.hits += 1
            return value
        return ""


class CuratedDescriptions:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.entries: dict[str, dict[str, str]] = {}
        self.meta: dict[str, Any] = {}
        self.used: set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        self.meta = payload.get("meta") or {}
        for section, mapping in payload.items():
            if section == "meta" or not isinstance(mapping, dict):
                continue
            for stable_id, value in mapping.items():
                if isinstance(value, dict):
                    self.entries[stable_id] = {
                        "title": str(value.get("title") or "").strip(),
                        "digest": str(value.get("digest") or "").strip(),
                    }

    def apply(self, item: dict[str, Any]) -> bool:
        """就地写入中文标题/说明；返回是否命中。

        只有「中文标题 + 中文说明」都齐备时才标记 `zhCurated=True`——
        报告正文只渲染中文条目（config 的 report.chineseOnly），
        只补了一半的条目仍算待补，避免英文混进正文。
        """
        stable_id = str(item.get("stableId") or "")
        curated = self.entries.get(stable_id)
        if not curated:
            item.setdefault("zhCurated", False)
            return False
        self.used.add(stable_id)
        original_title = str(item.get("title") or "")
        if curated.get("title"):
            item["title"] = curated["title"]
            # 保留原文标题：既便于对照，也用于跨源去重
            item.setdefault("originalTitle", original_title)
            item["originalTitleKey"] = title_key(original_title)
        if curated.get("digest"):
            item["digest"] = curated["digest"]
            item["digestSource"] = "curated"
        item["zhCurated"] = bool(curated.get("title")) and bool(curated.get("digest"))
        return True

    def stats(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        covered = sum(1 for item in items if str(item.get("stableId") or "") in self.entries)
        return {
            "curatedTotal": len(self.entries),
            "covered": covered,
            "total": total,
            "coverage": round(covered / total, 3) if total else 0.0,
            "unused": sorted(set(self.entries) - self.used)[:20],
        }
