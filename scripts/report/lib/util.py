#!/usr/bin/env python3
"""AscendPlayground 多模态 Infra 情报系统 · 公共工具。

仅依赖 Python 标准库，保证在任意 3.8+ 环境可复现。
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

UTC = timezone.utc

# --------------------------------------------------------------------------
# 时间
# --------------------------------------------------------------------------

_DATE_PATTERNS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a, %d %b %Y %H:%M:%S",
    "%a, %d %b %Y %H:%M %z",
    "%d %b %Y %H:%M:%S %z",
    "%Y%m%dT%H%M%SZ",
    "%Y%m%d",
)


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: Any) -> Optional[datetime]:
    """把各种格式的时间字符串解析为带时区的 UTC datetime；失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    # 规范化时区写法：+08:00 -> +0800
    text = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", text)
    for pattern in _DATE_PATTERNS:
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    # ISO 8601 兜底（Python 3.11+ 支持 Z 与部分非标准写法）
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def iso(value: Optional[datetime]) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if value else ""


def day(value: Optional[datetime]) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d") if value else "未标注"


def within_window(value: Optional[datetime], days: int, reference: Optional[datetime] = None) -> bool:
    if value is None:
        return True  # 无时间信息的条目保留，交由打分淘汰
    reference = reference or now_utc()
    return value >= reference - timedelta(days=days)


# --------------------------------------------------------------------------
# 文本
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"(?i)</p>", " ", text)
    text = _TAG_RE.sub(" ", text)
    # 去掉常见实体并保留可读性
    text = html.unescape(text)
    text = text.replace("\u200b", "").replace("\xa0", " ")
    return _WS_RE.sub(" ", text).strip()


def clean_title(value: Any, limit: int = 220) -> str:
    text = strip_html(value)
    text = re.sub(r"\s*[-|·]\s*(机器之心|量子位|新智元|Hugging Face|NVIDIA|OpenAI)\s*$", "", text)
    text = _WS_RE.sub(" ", text).strip(" -·|")
    return text[:limit].strip()


def truncate(value: str, limit: int) -> str:
    value = _WS_RE.sub(" ", (value or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text, flags=re.UNICODE).strip("-").lower()
    return text[:100] or "item"


def title_key(value: str) -> str:
    """跨源去重用的标题指纹：去掉标点/空白/大小写差异后取哈希。"""
    text = unicodedata.normalize("NFKC", value or "").lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else ""


def short_id(prefix: str, *parts: Any) -> str:
    payload = "\u241f".join(str(part) for part in parts if part not in (None, ""))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def text_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def markdown_escape(value: str) -> str:
    return re.sub(r"([|\\])", r"\\\1", (value or "").replace("\n", " ")).strip()


def linkify(title: str, url: str) -> str:
    label = markdown_escape(truncate(title, 150))
    return f"[{label}]({url})" if url else label


def keywords_in(text: str, terms: Iterable[str]) -> list[str]:
    haystack = (text or "").lower()
    hits: list[str] = []
    for term in terms:
        needle = term.lower().strip()
        if not needle:
            continue
        if needle in haystack:
            hits.append(term)
    return hits


# --------------------------------------------------------------------------
# 杂项
# --------------------------------------------------------------------------


def dedupe_preserve(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def week_key(value: Optional[datetime] = None) -> str:
    """ISO 周编号，例如 2026-W38。"""
    value = value or now_utc()
    year, week, _ = value.isocalendar()
    return f"{year}-W{week:02d}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def dump_json(path: Path, payload: Any) -> None:
    import json

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    import json

    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
