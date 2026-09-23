#!/usr/bin/env python3
"""关键词抽取 + 入库闸门 + 跨源去重。

本模块**不打分**：报告按发布时间排序，是否收录由 `ScopeGate` 的三道闸门决定
（排除法 → infra 证据 → 主题），词表都在 config/sources.json 里。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable, Optional

from .util import dedupe_preserve, parse_datetime, title_key

# 中文/日文/韩文表意文字：直接子串匹配
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _is_ascii_token(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.\-+/#:]*", term or ""))


def _ascii_pattern(term: str) -> str:
    """把配置里的英文关键词翻成正则。

    - 词内空格 → `[\\s\\-]+`（`kv cache` 也命中 `kv-cache`）
    - 词内连字符 → `[\\s\\-]*`（`multi-gpu` 也命中 `multi gpu` / `multigpu`）
    - 词首锚定 → 避免 `npu` 命中 `Input`、`dit` 命中 `audit`

    注意：这里必须一次拼装完成。早先的写法是「先 replace 空格、再 replace 连字符」，
    第二次替换会把第一次插入的 `[\\s\\-]+` 再改写一遍，产出
    `[\\s[\\s\\-]*]+` 这种要求字面 `]` 的必失配模式（实测 `kv cache` / `sparse attention`
    / `world action model` 全部静默失配，多词短语退化成邻域兜底匹配）。
    """
    parts = re.split(r"(\s+|-)", term.strip())
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part == "-":
            out.append(r"[\s\-]*")
        elif part.isspace():
            out.append(r"[\s\-]+")
        else:
            out.append(re.escape(part))
    return r"\b" + "".join(out)


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
        """所有词元是否在 NEIGHBORHOOD 字符窗口内按整词共现（"mode" 不应命中 "Models"）。"""
        lowered = text.lower()
        patterns = {token: re.compile(r"\b" + re.escape(token) + r"\b") for token in tokens}
        if any(not patterns[token].search(lowered) for token in tokens):
            return False
        first = tokens[0]
        for match in patterns[first].finditer(lowered):
            window = lowered[match.start() : match.start() + self.NEIGHBORHOOD]
            if all(patterns[token].search(window) for token in tokens[1:]):
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


class ScopeGate:
    """入库闸门：只判断「收 / 不收」，不做任何打分与排序（排序一律按发布时间）。

    两道范围闸门 + 一道主题闸门，词表全部在 config/sources.json：
      ① 排除法 scope.exclude        → 明确非 infra 议题直接丢
      ② infra 证据 scope.infraEvidence → 必须出现系统工程/降本增效证据词（默认只看标题）
      ③ 主题闸门 relevance.requireAnyOf → 必须属于多模态/生成式方向
    """

    def __init__(self, config: dict[str, Any]) -> None:
        relevance = dict(config.get("relevance") or {})
        self.require_any = [str(term) for term in relevance.get("requireAnyOf", [])]

        scope = dict(config.get("scope") or {})
        self.scope = scope
        self.infra_terms = [str(term) for term in scope.get("infraEvidence", []) if str(term).strip()]
        self.exclude_terms = [str(term) for term in scope.get("exclude", []) if str(term).strip()]
        self._infra_matcher = TagExtractor({"infra": self.infra_terms}) if self.infra_terms else None
        self._exclude_matcher = TagExtractor({"exclude": self.exclude_terms}) if self.exclude_terms else None
        self.require_infra_groups = {str(group) for group in scope.get("requireInfraEvidenceGroups", [])}
        self.require_infra_kinds = {str(kind) for kind in scope.get("requireInfraEvidenceKinds", [])}
        self.exclude_groups = {str(group) for group in scope.get("excludeGroups", [])}
        # 证据词只看标题：摘要里 throughput / latency / kernel 之类的词人人都写，不足以判定 infra
        self.evidence_in_title = bool(scope.get("evidenceInTitle", False))

    def passes_gate(self, text: str, *, require_infra: bool = False, title: str = "") -> bool:
        """主题闸门：先要求命中多模态主题词，可选再要求 infra 证据词。

        `title` 非空且 `scope.evidenceInTitle=true` 时，infra 证据只在标题里找。
        """
        if self.require_any:
            lowered = (text or "").lower()
            if not any(term.lower() in lowered for term in self.require_any):
                return False
        if require_infra:
            haystack = title if (self.evidence_in_title and title) else text
            if not self.has_infra_evidence(haystack):
                return False
        return True

    def has_infra_evidence(self, text: str) -> bool:
        """正文里是否出现「系统工程 / 降本增效」类证据词（scope.infraEvidence）。"""
        if self._infra_matcher is None:
            return True
        return any(self._infra_matcher.matches(term, text) for term in self.infra_terms)

    def out_of_scope_reason(self, text: str) -> str:
        """命中排除词则返回该词（用于日志/统计），否则返回空串。"""
        if self._exclude_matcher is None:
            return ""
        for term in self.exclude_terms:
            if self._exclude_matcher.matches(term, text):
                return term
        return ""

    def requires_infra_evidence(
        self, group: str, source: Optional[dict[str, Any]] = None, kind: str = ""
    ) -> bool:
        """分组或条目类型任一要求 infra 证据即可（博客/媒体文章也要，避免混进纯资讯）。"""
        source = source or {}
        if "requireInfraEvidence" in source:
            return bool(source.get("requireInfraEvidence"))
        if str(kind or "") in self.require_infra_kinds:
            return True
        return str(group or "") in self.require_infra_groups

    def should_check_exclude(self, group: str, source: Optional[dict[str, Any]] = None) -> bool:
        source = source or {}
        if "skipScopeExclude" in source:
            return not bool(source.get("skipScopeExclude"))
        return str(group or "") in self.exclude_groups


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
    """把仓库动态按 repo 聚合，输出每仓的 release/commit/PR 计数与时间跨度。

    无打分：活跃度按「条目数 + 最近动态时间」排序。
    """
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
                "counts": {"release": 0, "commit": 0, "pr": 0, "issue": 0},
                "latest": None,
                "items": [],
            },
        )
        counts = bucket["counts"]
        kind = item.get("repoActivity")
        if kind in counts:
            counts[kind] += 1
        bucket["stars"] = max(bucket["stars"], int(item.get("stars") or 0))
        published = parse_datetime(item.get("published"))
        if published and (bucket["latest"] is None or published > bucket["latest"]):
            bucket["latest"] = published
        bucket["items"].append(item)

    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        bucket["items"].sort(key=lambda entry: str(entry.get("published") or ""), reverse=True)
        bucket["total"] = sum(bucket["counts"].values())
        bucket["latestIso"] = bucket["latest"].strftime("%Y-%m-%d %H:%MZ") if bucket["latest"] else ""
        out.append(bucket)
    out.sort(key=lambda entry: (entry["total"], entry["latestIso"]), reverse=True)
    return out
