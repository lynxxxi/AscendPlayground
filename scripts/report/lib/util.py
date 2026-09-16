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
# 说明文字抽取（让报告每条自解释，不需要点进去）
# --------------------------------------------------------------------------

_SENTENCE_END = "。！？!?；;"
_CJK_CHAR = re.compile(r"[\u3400-\u9fff]")
_BULLET_LINE = re.compile(r"^[-*•·]\s+")
_HEADING_LINE = re.compile(r"^#{1,6}\s+")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_HTML_BLOCK = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_COMMIT_PREFIX = re.compile(
    r"^\s*(\[[^\]]{2,30}\]\s*)+"          # [Perf][CosyVoice3]
    r"|^\s*(feat|fix|perf|refactor|chore|docs|test|ci|build|style|revert)"
    r"(\([^)]*\))?!?:\s*",
    re.IGNORECASE,
)
_NOISE_LINE = re.compile(
    r"^(full changelog|what'?s changed|new contributors|contributors|"
    r"full changelog|signature|installation|usage|requirements|"
    r"generated by|this release|compare|assets?\s*\d*)$",
    re.IGNORECASE,
)


def _cut_at_sentence(text: str, limit: int) -> str:
    """在句子边界处截断，避免把句子从中间切开。"""
    if len(text) <= limit:
        return text
    head = text[:limit]
    best = max(head.rfind(char) for char in _SENTENCE_END)
    if best >= limit * 0.5:
        return head[: best + 1]
    return head.rstrip() + "…"


def digest(value: Any, limit: int = 260, *, prefer_bullets: bool = False) -> str:
    """把正文压成一句到两句的说明。

    * 去掉 HTML、代码块、URL、changelog 噪声行；
    * prefer_bullets=True 时优先返回列举式要点（适合 release notes）。
    """
    if value is None:
        return ""
    text = _HTML_BLOCK.sub(" ", str(value))
    text = _CODE_FENCE.sub(" ", text)
    if "<" in text and ">" in text:
        text = strip_html(text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("\r", "\n")

    lines: list[str] = []
    bullets: list[str] = []
    for raw in text.split("\n"):
        line = _WS_RE.sub(" ", raw).strip()
        if not line:
            continue
        if _HEADING_LINE.match(line):
            continue
        is_bullet = bool(_BULLET_LINE.match(line)) or bool(re.match(r"^\d+[、.)]\s*", line))
        stripped = _BULLET_LINE.sub("", line).strip()
        stripped = re.sub(r"^\d+[、.)]\s*", "", stripped).strip()
        # 归一化后再判断噪声行（去掉 markdown 强调符与尾部标点）
        normalized = re.sub(r"[\s*_`:\-]+$", "", stripped).strip("*_` \t").lower()
        raw_normalized = re.sub(r"[\s*_`:\-]+$", "", line).strip("*_` \t").lower()
        if _NOISE_LINE.match(normalized) or _NOISE_LINE.match(raw_normalized):
            continue
        if re.fullmatch(r"[\W_*]+", line):
            continue
        if len(stripped) < 6:
            continue
        lines.append(stripped)
        if is_bullet:
            bullets.append(stripped)

    if not lines:
        return ""
    if prefer_bullets and bullets:
        # 用 · 连接，避免与条目自身的。；混淆
        return _cut_at_sentence(" · ".join(bullets[:3]), limit)

    body = " ".join(lines)
    body = _WS_RE.sub(" ", body).strip()
    if len(body) <= limit:
        return body
    head = body[:limit]
    if _CJK_CHAR.search(body):
        best = max(head.rfind(char) for char in _SENTENCE_END)
        if best >= limit * 0.4:
            return head[: best + 1]
        return head.rstrip() + "…"
    matches = list(re.finditer(r"[.!?](?:\s|$)", head))
    if matches and matches[-1].end() >= limit * 0.4:
        return head[: matches[-1].end()].strip()
    return head.rsplit(" ", 1)[0].rstrip() + "…"


def summarise_abstract(value: Any, limit: int = 320) -> str:
    """论文摘要：优先取第一句结论性描述，否则取开头。"""
    text = digest(value, limit=10_000, prefer_bullets=False)
    if not text:
        return ""
    for marker in ("We present", "We propose", "We introduce", "This paper", "In this paper", "本文", "我们提出"):
        position = text.find(marker)
        if 0 <= position <= 200:
            return _cut_at_sentence(text[position:], limit)
    return _cut_at_sentence(text, limit)


_TRAILER_LINE = re.compile(
    r"^(co-authored-by|co-committed-by|signed-off-by|reviewed-by|tested-by|"
    r"created-by|commit-by|merged-by|approved-by|change-id|see merge request|"
    r"merge (dev|master|main)|cherry[- ]picked from)\b.*$",
    re.IGNORECASE,
)
_PR_REF = re.compile(r"\(#\d+\)|!\d+\s|#\d+\s*$")
_TEMPLATE_HEADER = re.compile(
    r"^(what this pr does.*|why is this pr needed.*|does this pr introduce.*|"
    r"how was this tested.*|description|motivation|summary|背景|方案|描述|修改点)\s*[:：/]?\s*$",
    re.IGNORECASE,
)


def _clean_commit_text(text: str) -> str:
    """清理 commit/PR 正文中的样板：trailer、模板小标题、PR 编号引用。"""
    kept: list[str] = []
    for raw in (text or "").split("\n"):
        line = _WS_RE.sub(" ", raw).strip().lstrip("#").strip()
        if not line:
            continue
        if _TRAILER_LINE.match(line):
            continue
        if _TEMPLATE_HEADER.match(line):
            continue
        if re.fullmatch(r"[-=_*`~\s]{3,}", line):
            continue
        kept.append(line)
    joined = " ".join(kept)
    joined = _PR_REF.sub("", joined)
    return _WS_RE.sub(" ", joined).strip()


def commit_digest(subject: str, body: str = "", limit: int = 240) -> str:
    """commit 说明：去掉 [Tag]/conventional 前缀与 PR 样板，正文优先。"""
    cleaned_subject = _clean_commit_text(_COMMIT_PREFIX.sub("", (subject or "").strip()))
    body_text = _clean_commit_text(body)
    # 正文常常重复 subject（GitHub 把标题作为正文首行），去重后再比较
    if body_text and cleaned_subject:
        head = body_text[: len(cleaned_subject) + 12].lower()
        if cleaned_subject[:40].lower() in head:
            body_text = _clean_commit_text(body_text.replace(cleaned_subject, "", 1))
    if body_text and len(body_text) > len(cleaned_subject):
        return _cut_at_sentence(body_text, limit)
    return _cut_at_sentence(cleaned_subject or (subject or "").strip(), limit)


def first_paragraph(page: str, *, minimum: int = 80, limit: int = 320) -> str:
    """从文章页抽首个有信息量的段落（跳过导航/页脚样板）。"""
    if not page:
        return ""
    cleaned = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form)[^>]*>.*?</\1>", " ", page)
    for match in re.finditer(r"(?is)<p[^>]*>(.*?)</p>", cleaned):
        text = strip_html(match.group(1))
        lowered = text.lower()
        if len(text) < minimum:
            continue
        if any(
            token in lowered
            for token in (
                "we're on a journey",
                "a blog post by",
                "cookie",
                "subscribe",
                "sign in",
                "all rights reserved",
                "privacy policy",
                "skip to",
                "models datasets spaces",
            )
        ):
            continue
        if len(re.findall(r"\w", text)) < 40:
            continue
        return _cut_at_sentence(text, limit)
    return ""


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
