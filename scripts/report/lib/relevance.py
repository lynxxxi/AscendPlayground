#!/usr/bin/env python3
"""相关性关键词抽取 + 打分 + 跨源去重。

打分模型（可在 config 的 scoring 段调参）：
    score = (1 + 关键词加权命中，上限 maxKeywordBoost)
            × 源权重 weight
            × 时效因子 recency
            × 新条目加成 newItemBoost

其中 recency 在 lookbackDays 内为 1.0，之后线性衰减到 minRecencyFactor。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from .util import dedupe_preserve, now_utc, parse_datetime, title_key

# 中文/日文/韩文表意文字：直接子串匹配
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _is_ascii_token(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.\-+/#:]*", term or ""))


def _ascii_pattern(term: str) -> str:
    escaped = re.escape(term.strip())
    # 空格与连字符允许彼此互换，命中 "kv cache" / "kv-cache"
    escaped = escaped.replace(r"\ ", r"[\s\-]+").replace(r"\-", r"[\s\-]*")
    return escaped


class TagExtractor:
    """基于配置的关键词表做标签抽取，全部命中即可作为打分依据。

    匹配策略（按优先级）：
      1. 中文/日韩词：直接子串匹配。
      2. 英文多词短语：先试精确短语；不中则退化为"所有词都在同一邻域窗口内出现"，
         这样 `multimodal serving` 也能命中 "multimodal inference serving"。
      3. 单词：词边界 + 允许空格/连字符互换（`kv cache` ↔ `kv-cache`）。
    """

    NEIGHBORHOOD = 80

    def __init__(self, keyword_groups: dict[str, list[str]], weights: Optional[dict[str, float]] = None) -> None:
        self.groups: list[tuple[str, list[str]]] = [
            (group, [term for term in terms if term and term.strip()])
            for group, terms in (keyword_groups or {}).items()
        ]
        self.term_weights: dict[str, float] = dict(weights or {})
        self._patterns: dict[str, Optional[re.Pattern[str]]] = {}
        self._tokens: dict[str, list[str]] = {}
        for _, terms in self.groups:
            for term in terms:
                self._patterns[term] = self._compile(term)
                if not _CJK_RE.search(term) and _is_ascii_token(term) and len(term.split()) > 1:
                    self._tokens[term] = [token for token in re.split(r"[\s\-]+", term.strip().lower()) if token]

    @staticmethod
    def _compile(term: str) -> Optional[re.Pattern[str]]:
        if _CJK_RE.search(term):
            return None  # 中日韩词直接 in 判断
        if _is_ascii_token(term):
            return re.compile(_ascii_pattern(term), re.IGNORECASE)
        return re.compile(re.escape(term), re.IGNORECASE)

    def matches(self, term: str, text: str) -> bool:
        if not text:
            return False
        pattern = self._patterns.get(term)
        if pattern is None:
            return term.lower() in text.lower()
        if pattern.search(text):
            return True
        tokens = self._tokens.get(term)
        if tokens:
            return self._tokens_nearby(tokens, text)
        return False

    def _tokens_nearby(self, tokens: list[str], text: str) -> bool:
        """所有词元是否在 NEIGHBORHOOD 字符窗口内共现。"""
        lowered = text.lower()
        for token in tokens:
            if token not in lowered:
                return False
        first = tokens[0]
        for match in re.finditer(re.escape(first), lowered):
            window = lowered[match.start() : match.start() + self.NEIGHBORHOOD]
            if all(token in window for token in tokens[1:]):
                return True
        return False

    def extract(self, text: str) -> list[str]:
        return [term for _, terms in self.groups for term in terms if self.matches(term, text)]

    def extract_with_groups(self, text: str) -> list[tuple[str, str]]:
        hits: list[tuple[str, str]] = []
        for group, terms in self.groups:
            for term in terms:
                if self.matches(term, text):
                    hits.append((group, term))
        return hits


def merge_keywords(hits: Iterable[tuple[str, str]]) -> list[str]:
    return dedupe_preserve(f"{group}/{term}" for group, term in hits)


def _weights_for(hits: Iterable[tuple[str, str]], table: dict[str, float]) -> float:
    total = 0.0
    for _, term in hits:
        total += table.get(term.lower(), table.get(term, 1.0))
    return total


def _as_datetime(value: Any) -> Optional[datetime]:
    """兼容 datetime 与 ISO 字符串。"""
    if value is None or isinstance(value, datetime):
        return value
    return parse_datetime(value)


class Scorer:
    def __init__(self, config: dict[str, Any]) -> None:
        scoring = dict(config.get("scoring") or {})
        relevance = dict(config.get("relevance") or {})
        defaults = dict(config.get("defaults") or {})
        self.title_weight = float(scoring.get("titleWeight", 3.0))
        self.summary_weight = float(scoring.get("summaryWeight", 1.0))
        self.max_boost = float(scoring.get("maxKeywordBoost", 12.0))
        self.lookback_days = float(scoring.get("lookbackDays", 7))
        self.min_recency = float(scoring.get("minRecencyFactor", 0.35))
        self.new_boost = float(scoring.get("newItemBoost", 1.0))
        self.min_score = float(defaults.get("minScore", 0.0))
        table: dict[str, float] = {}
        for term in relevance.get("heavy", []):
            table[str(term).lower()] = 3.0
        for term in relevance.get("medium", []):
            table[str(term).lower()] = 1.5
        for term in relevance.get("requireAnyOf", []):
            table.setdefault(str(term).lower(), 2.0)
        self.weights = table
        self.require_any = [str(term) for term in relevance.get("requireAnyOf", [])]

    def passes_gate(self, text: str) -> bool:
        if not self.require_any:
            return True
        lowered = (text or "").lower()
        return any(term.lower() in lowered for term in self.require_any)

    def score(self, item: dict[str, Any], reference: Optional[datetime] = None) -> float:
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        weight = float(item.get("weight") or 1.0)

        boost = 0.0
        text = f"{title} {summary}".lower()
        for term, value in self.weights.items():
            if term in text:
                boost += value * weight
        boost = min(boost, self.max_boost)

        recency = self.recency_factor(_as_datetime(item.get("published")), reference)
        raw = (1.0 + boost) * weight * recency
        if item.get("isNew", True):
            raw *= self.new_boost
        return round(raw, 4)

    def recency_factor(self, published: Optional[datetime], reference: Optional[datetime] = None) -> float:
        if published is None:
            return self.min_recency
        reference = reference or now_utc()
        if published.tzinfo is None:
            published = published.replace(tzinfo=reference.tzinfo)
        age_days = max(0.0, (reference - published).total_seconds() / 86400.0)
        if age_days <= self.lookback_days:
            return 1.0
        span = max(1.0, self.lookback_days)
        decay = 1.0 - ((age_days - self.lookback_days) / (span * 4.0))
        return max(self.min_recency, min(1.0, decay))


def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """先按 stable_id 合并同源重复，再按标题指纹合并跨源同一事件。"""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in items:
        key = item.get("stableId") or item.get("url") or item.get("title", "")
        if key in by_id:
            _merge_into(by_id[key], item)
            continue
        by_id[key] = item
        order.append(key)

    by_title: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for key in order:
        item = by_id[key]
        # 中文标题会改变标题指纹，因此同时用原文标题（若存在）参与跨源去重
        fingerprints = [fp for fp in (item.get("originalTitleKey"), title_key(item.get("title", ""))) if fp]
        matched = next((by_title[fp] for fp in fingerprints if fp in by_title), None)
        if matched is not None:
            _merge_into(matched, item)
            continue
        for fp in fingerprints:
            by_title[fp] = item
        result.append(item)
    return result


def _merge_into(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    target["sources"] = dedupe_preserve(
        list(target.get("sources") or []) + list(incoming.get("sources") or [])
    )
    target["sourceLabels"] = dedupe_preserve(
        list(target.get("sourceLabels") or []) + list(incoming.get("sourceLabels") or [])
    )
    target["keywords"] = dedupe_preserve(
        list(target.get("keywords") or []) + list(incoming.get("keywords") or [])
    )
    target["repoRefs"] = dedupe_preserve(
        list(target.get("repoRefs") or []) + list(incoming.get("repoRefs") or [])
    )
    signals = dict(target.get("signals") or {})
    for key, value in (incoming.get("signals") or {}).items():
        if isinstance(value, (int, float)) and isinstance(signals.get(key), (int, float)):
            signals[key] = max(signals[key], value)
        else:
            signals.setdefault(key, value)
    target["signals"] = signals
    if not target.get("summary") and incoming.get("summary"):
        target["summary"] = incoming["summary"]
    if not target.get("digest") and incoming.get("digest"):
        target["digest"] = incoming["digest"]
    if incoming.get("weight", 0) > target.get("weight", 0):
        target["weight"] = incoming["weight"]
    target_published = parse_datetime(target.get("published"))
    incoming_published = parse_datetime(incoming.get("published"))
    if incoming_published and (not target_published or incoming_published < target_published):
        target["published"] = incoming["published"]
    target["isNew"] = bool(target.get("isNew", True) or incoming.get("isNew", True))
    target["duplicateCount"] = int(target.get("duplicateCount", 1)) + 1


def cluster_repo_activity(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把仓库动态按 repo 聚合，输出每仓的 release/commit/PR 计数与时间跨度。"""
    buckets: dict[str, dict[str, Any]] = {}
    for item in items:
        repo = item.get("repo")
        if not repo:
            continue
        bucket = buckets.setdefault(
            repo,
            {
                "repo": repo,
                "label": repo,
                "group": item.get("repoGroup") or "",
                "note": item.get("repoNote") or "",
                "stars": item.get("stars") or 0,
                "combinedScore": 0.0,
                "counts": {"release": 0, "commit": 0, "pr": 0, "issue": 0},
                "latest": None,
                "items": [],
            },
        )
        counts = bucket["counts"]
        kind = item.get("repoActivity")
        if kind in counts:
            counts[kind] += 1
        bucket["combinedScore"] = round(bucket["combinedScore"] + float(item.get("score") or 0), 3)
        bucket["stars"] = max(bucket["stars"], int(item.get("stars") or 0))
        published = parse_datetime(item.get("published"))
        if published and (bucket["latest"] is None or published > bucket["latest"]):
            bucket["latest"] = published
        bucket["items"].append(item)

    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        bucket["items"].sort(key=lambda entry: entry.get("score", 0), reverse=True)
        bucket["total"] = sum(bucket["counts"].values())
        bucket["latestIso"] = bucket["latest"].strftime("%Y-%m-%d %H:%MZ") if bucket["latest"] else ""
        out.append(bucket)
    out.sort(key=lambda entry: (entry["total"], entry["combinedScore"]), reverse=True)
    return out
