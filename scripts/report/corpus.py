#!/usr/bin/env python3
"""L2 语料层：去重、中文过滤、排序、主题聚类、周边团队矩阵。

设计要点：
  · **不打分**：条目顺序一律按发布时间，收录与否只由 ScopeGate 的闸门决定。
  · **只出中文**：报告正文只保留已补中文说明（config/curated_zh.json）的条目；
    未补中文的条目不进正文，计入 stats["pendingZh"]，并写到采集缓存的 pending_zh.json
    作为待补清单（周报流程 = 先采集 → 补中文 → 再离线渲染）。
  · “本周新增”不依赖任何持久化文件：发布时间落在窗口内即视为新增，
    因此同一份快照无论何时重跑，结果都完全一致（可复现）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from report.collect import repo_index  # noqa: F401 - 对外统一出口
from report.lib.relevance import TagExtractor, cluster_repo_activity, dedupe
from report.lib.util import (
    day,
    dedupe_preserve,
    iso,
    keywords_in,
    now_utc,
    parse_datetime,
)

GROUP_LABELS = {
    "papers": "论文与研究",
    "teams": "技术团队与媒体",
    "repos": "多模态仓库更新",
    "wechat": "公众号 / 中文媒体",
    "peers": "周边团队工作",
}

KIND_LABELS = {
    "paper": "论文",
    "blog": "博客/资讯",
    "model-release": "模型发布",
    "repo-release": "仓发布",
    "repo-commit": "代码提交",
    "repo-pr": "Pull Request",
    "wechat-article": "公众号文章",
    "media-article": "媒体文章",
    "peer-release": "周边团队发版",
}


@dataclass
class Corpus:
    run_id: str
    week: str
    generated_at: str
    freshness_days: int
    items: list[dict[str, Any]]
    stats: dict[str, Any]
    repo_clusters: list[dict[str, Any]]
    peer_matrix: dict[str, Any]
    themes: list[dict[str, Any]]
    pending_zh: list[dict[str, Any]] = field(default_factory=list)


def build_corpus(
    raw_items: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    run_id: str,
    week: str,
    reference: Optional[datetime] = None,
    freshness_days: Optional[int] = None,
    release_history: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> Corpus:
    reference = reference or now_utc()
    scoring = config.get("scoring") or {}
    if freshness_days is None:
        freshness_days = int(scoring.get("lookbackDays", 7))
    fresh_cutoff = reference - timedelta(days=freshness_days)

    merged = dedupe(raw_items)
    for item in merged:
        item["sources"] = dedupe_preserve(list(item.get("sources") or []) + [item.get("sourceId", "")])
        item["sourceLabels"] = dedupe_preserve(
            list(item.get("sourceLabels") or []) + [item.get("sourceLabel", "")]
        )
        published = parse_datetime(item.get("published"))
        # 本周新增：发布时间落在 lookbackDays 窗口内（无时间信息者保守视为非新增）
        item["isNew"] = bool(published and published >= fresh_cutoff)

    chinese_only = bool((config.get("report") or {}).get("chineseOnly", False))
    if chinese_only:
        kept = [item for item in merged if bool(item.get("zhCurated"))]
        pending = [item for item in merged if not bool(item.get("zhCurated"))]
    else:
        kept = list(merged)
        pending = []
    # 排序：新增优先 → 发布时间倒序（无分数）
    kept.sort(key=lambda entry: (1 if entry.get("isNew") else 0, str(entry.get("published") or "")), reverse=True)

    stats = corpus_stats(kept, raw_items)
    stats["pendingZh"] = len(pending)
    stats["chineseOnly"] = chinese_only
    clusters = cluster_repo_activity([item for item in kept if item.get("repo")])
    for cluster in clusters:
        cluster["label"] = cluster["repo"]

    return Corpus(
        run_id=run_id,
        week=week,
        generated_at=iso(reference),
        freshness_days=freshness_days,
        items=kept,
        stats=stats,
        repo_clusters=clusters,
        peer_matrix=build_peer_matrix(kept, config, reference, release_history or {}),
        themes=build_themes(kept, config),
        pending_zh=[
            {
                "stableId": item.get("stableId"),
                "group": item.get("group"),
                "kind": item.get("kind"),
                "title": item.get("title"),
                "day": item.get("day"),
                "url": item.get("url"),
            }
            for item in sorted(pending, key=lambda entry: (str(entry.get("group")), str(entry.get("published"))))
        ],
    )


def corpus_stats(items: list[dict[str, Any]], raw_items: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_new_group: dict[str, int] = {}
    by_day: dict[str, int] = {}
    new_count = 0
    for item in items:
        group = str(item.get("group") or "other")
        by_group[group] = by_group.get(group, 0) + 1
        kind = str(item.get("kind") or "other")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if item.get("isNew"):
            new_count += 1
            by_new_group[group] = by_new_group.get(group, 0) + 1
        item_day = str(item.get("day") or "")
        if item_day:
            by_day[item_day] = by_day.get(item_day, 0) + 1
    return {
        "raw": len(raw_items),
        "total": len(items),
        "new": new_count,
        "byGroup": dict(sorted(by_group.items())),
        "byGroupLabel": {GROUP_LABELS.get(key, key): value for key, value in sorted(by_group.items())},
        "byKind": dict(sorted(by_kind.items(), key=lambda pair: pair[1], reverse=True)),
        "newByGroup": dict(sorted(by_new_group.items())),
        "byDay": dict(sorted(by_day.items())),
        "multiSource": sum(1 for item in items if len(item.get("sources") or []) > 1),
    }


def _items_for_repo(items: Iterable[dict[str, Any]], repo: str) -> list[dict[str, Any]]:
    target = repo.lower()
    return [item for item in items if str(item.get("repo") or "").lower() == target]


def _by_published_desc(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda entry: str(entry.get("published") or ""), reverse=True)


# --------------------------------------------------------------------------
# 维度 E · 周边团队工作
# --------------------------------------------------------------------------


def build_peer_matrix(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    reference: datetime,
    release_history: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """周边团队的版本特性（按引擎）+ 团队动态雷达（按团队名录反查语料）。"""
    peers = config.get("peers") or {}
    engines = peers.get("engines") or []
    chinese_only = bool((config.get("report") or {}).get("chineseOnly", False))
    history = {str(repo).lower(): list(releases) for repo, releases in (release_history or {}).items()}
    # 条目级中文说明（config/curated_zh.json）优先：repo+tag → 中文说明
    curated_digest: dict[tuple[str, str], str] = {}
    for item in items:
        if not item.get("zhCurated"):
            continue
        repo = str(item.get("repo") or "").lower()
        tag = str((item.get("signals") or {}).get("tag") or "")
        if repo and tag and item.get("digest"):
            curated_digest.setdefault((repo, tag), str(item["digest"]))
    highlights = int(peers.get("releaseHighlightsPerEngine", 3))
    rows: list[dict[str, Any]] = []
    for engine in engines:
        repo = str(engine.get("repo") or "")
        engine_items = _items_for_repo(items, repo)
        releases = [item for item in engine_items if item.get("repoActivity") == "release"]
        history_sorted = _by_published_desc(history.get(repo.lower(), []))
        recent_releases: list[dict[str, Any]] = []
        for entry in history_sorted:
            tag = str(entry.get("tag") or "")
            digest = curated_digest.get((repo.lower(), tag)) or str(entry.get("digest") or "")
            has_zh = (repo.lower(), tag) in curated_digest
            # 报告只出中文：没有中文说明的版本卡片不进正文（避免英文混排）
            if chinese_only and not has_zh:
                continue
            recent_releases.append(
                {
                    "tag": tag,
                    "published": entry.get("published", ""),
                    "day": (entry.get("published", "") or "")[:10],
                    "digest": digest,
                    "url": entry.get("url", ""),
                    "isNew": bool(entry.get("published"))
                    and (parse_datetime(entry.get("published")) or reference) >= reference - timedelta(days=14),
                }
            )
            if len(recent_releases) >= highlights:
                break
        rows.append(
            {
                "id": engine.get("id"),
                "label": engine.get("label"),
                "org": engine.get("org") or "",
                "category": engine.get("category") or "",
                "repo": repo,
                "stack": engine.get("stack") or "",
                "releaseCount": len(history.get(repo.lower(), [])) or len(releases),
                "windowCount": len(engine_items),
                "latestTag": (history_sorted[0].get("tag") if history_sorted else ""),
                "latestDay": (history_sorted[0].get("published", "")[:10] if history_sorted else ""),
                "recentReleases": recent_releases,
            }
        )
    rows.sort(key=lambda row: (str(row.get("category") or ""), str(row.get("label") or "")))
    return {
        "generatedAt": iso(reference),
        "rows": rows,
        "radar": build_peer_radar(items, config),
        "categories": sorted({str(row.get("category") or "") for row in rows if row.get("category")}),
    }


def build_peer_radar(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """按 teams(peers.roster) 名录反查本周语料：谁这周有动静、动静是什么。"""
    roster = (config.get("peers") or {}).get("roster") or []
    alias_owner: dict[str, dict[str, Any]] = {}
    aliases: list[str] = []
    for team in roster:
        for alias in team.get("aliases") or []:
            alias = str(alias).strip()
            if alias and alias not in alias_owner:
                alias_owner[alias] = team
                aliases.append(alias)
    if not aliases:
        return []
    extractor = TagExtractor({"peers": aliases})
    matched: dict[str, list[tuple[dict[str, Any], list[str]]]] = {}
    for item in items:
        haystack = f"{item.get('title','')} {item.get('digest','')} {item.get('summary','')}"
        hits = [alias for alias in aliases if extractor.matches(alias, haystack)]
        for alias in hits:
            team = alias_owner[alias]
            matched.setdefault(str(team.get("id")), []).append((item, hits))
    rows: list[dict[str, Any]] = []
    for team in roster:
        team_id = str(team.get("id"))
        pairs = matched.get(team_id, [])
        seen: set[str] = set()
        unique: list[tuple[dict[str, Any], list[str]]] = []
        for item, hits in pairs:
            key = str(item.get("stableId") or item.get("url") or item.get("title"))
            if key in seen:
                continue
            seen.add(key)
            unique.append((item, hits))
        unique.sort(key=lambda pair: str(pair[0].get("published") or ""), reverse=True)
        rows.append(
            {
                "id": team_id,
                "label": team.get("label"),
                "category": team.get("category") or "",
                "aliases": list(team.get("aliases") or []),
                "matches": len(unique),
                "newMatches": sum(1 for item, _ in unique if item.get("isNew")),
                "evidence": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "day": item.get("day"),
                        "group": item.get("group"),
                        "kind": item.get("kind"),
                        "isNew": item.get("isNew"),
                        "matched": hits[:4],
                    }
                    for item, hits in unique[:4]
                ],
            }
        )
    rows.sort(key=lambda row: (row["newMatches"], row["matches"]), reverse=True)
    return rows


def build_themes(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """主题聚类：按 config.keywords 的关键词组统计命中条目（不打分，按时间取代表条目）。"""
    themes: list[dict[str, Any]] = []
    for theme, terms in (config.get("keywords") or {}).items():
        matched: list[tuple[dict[str, Any], list[str]]] = []
        for item in items:
            haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            hits = keywords_in(haystack, [str(term) for term in terms])
            if hits:
                matched.append((item, hits))
        if not matched:
            continue
        matched.sort(key=lambda pair: str(pair[0].get("published") or ""), reverse=True)
        themes.append(
            {
                "theme": theme,
                "count": len(matched),
                "newCount": sum(1 for item, _ in matched if item.get("isNew")),
                "top": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "day": item.get("day"),
                        "group": item.get("group"),
                        "isNew": item.get("isNew"),
                    }
                    for item, _ in matched[:6]
                ],
            }
        )
    themes.sort(key=lambda entry: (entry["newCount"], entry["count"]), reverse=True)
    return themes
