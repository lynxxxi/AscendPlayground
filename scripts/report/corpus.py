#!/usr/bin/env python3
"""L2 语料层：去重、打分排序、主题聚类、竞品与能力矩阵。

“本周新增”不依赖任何持久化文件：条目的发布时间落在本周窗口内即视为新增，
因此同一份快照无论何时重跑，结果都完全一致（可复现）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from report.collect import repo_index  # noqa: F401 - 对外统一出口
from report.lib.relevance import Scorer, cluster_repo_activity, dedupe
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
    "repos": "仓库更新细则",
    "wechat": "公众号 / 中文媒体",
    "competitors": "竞品发版观测",
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
    "competitor-release": "竞品发版",
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
    competitor_matrix: dict[str, Any]
    capability_matrix: dict[str, Any]
    themes: list[dict[str, Any]]
    mindie: dict[str, Any] = field(default_factory=dict)


def build_corpus(
    raw_items: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    run_id: str,
    week: str,
    reference: Optional[datetime] = None,
    min_score: Optional[float] = None,
    freshness_days: Optional[int] = None,
    release_history: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> Corpus:
    reference = reference or now_utc()
    defaults = config.get("defaults") or {}
    scoring = config.get("scoring") or {}
    if min_score is None:
        min_score = float(defaults.get("minScore", 0.0))
    if freshness_days is None:
        freshness_days = int(scoring.get("lookbackDays", 7))
    fresh_cutoff = reference - timedelta(days=freshness_days)

    merged = dedupe(raw_items)
    for item in merged:
        item["sources"] = dedupe_preserve(list(item.get("sources") or []) + [item.get("sourceId", "")])
        item["sourceLabels"] = dedupe_preserve(
            list(item.get("sourceLabels") or []) + [item.get("sourceLabel", "")]
        )

    scorer = Scorer(config)
    for item in merged:
        item["score"] = scorer.score(item, reference)
        published = parse_datetime(item.get("published"))
        # 本周新增：发布时间落在 lookbackDays 窗口内（无时间信息者保守视为非新增）
        item["isNew"] = bool(published and published >= fresh_cutoff)

    kept = [item for item in merged if float(item.get("score") or 0) >= min_score]
    kept.sort(key=lambda entry: (entry.get("score", 0), entry.get("published", "")), reverse=True)

    stats = corpus_stats(kept, raw_items)
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
        competitor_matrix=build_competitor_matrix(kept, config, reference, release_history or {}),
        capability_matrix=build_capability_matrix(kept, config),
        themes=build_themes(kept, config),
        mindie=build_mindie_section(kept, config, reference, release_history or {}),
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


def build_competitor_matrix(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    reference: datetime,
    release_history: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    competitors = config.get("competitors") or {}
    engines = competitors.get("engines") or []
    focus = set(competitors.get("focus") or [])
    history = {str(repo).lower(): list(releases) for repo, releases in (release_history or {}).items()}
    rows: list[dict[str, Any]] = []
    for engine in engines:
        repo = str(engine.get("repo") or "")
        engine_items = _items_for_repo(items, repo)
        releases = [item for item in engine_items if item.get("repoActivity") == "release"]
        commits = [item for item in engine_items if item.get("repoActivity") == "commit"]
        pulls = [item for item in engine_items if item.get("repoActivity") == "pr"]
        releases_sorted = sorted(releases, key=lambda entry: entry.get("published", ""), reverse=True)
        history_sorted = sorted(
            history.get(repo.lower(), []), key=lambda entry: entry.get("published", ""), reverse=True
        )
        latest = releases_sorted[0] if releases_sorted else (history_sorted[0] if history_sorted else None)
        window_1w = _count_since(engine_items, reference, days=7)
        window_2w = _count_since(engine_items, reference, days=14)
        # 展示用：最近几次发版的特性说明（而非发版次数）
        recent_releases = [
            {
                "tag": entry.get("tag", ""),
                "published": entry.get("published", ""),
                "day": (entry.get("published", "") or "")[:10],
                "digest": entry.get("digest", ""),
                "url": entry.get("url", ""),
                "isNew": (parse_datetime(entry.get("published")) or reference) >= reference - timedelta(days=14)
                if entry.get("published")
                else False,
            }
            for entry in history_sorted[: int(competitors.get("releaseHighlightsPerEngine", 3))]
        ]
        rows.append(
            {
                "id": engine.get("id"),
                "label": engine.get("label"),
                "repo": repo,
                "stack": engine.get("stack") or "",
                "isFocus": str(engine.get("id")) in focus,
                "total": len(engine_items),
                "releases": len(releases) or len(history.get(repo.lower(), [])),
                "commits": len(commits),
                "pulls": len(pulls),
                "window1w": window_1w,
                "window2w": window_2w,
                "velocity": round(window_1w / 7.0, 2),
                "latestTag": (latest or {}).get("tag") or (latest or {}).get("signals", {}).get("tag") or "",
                "latestRelease": (latest or {}).get("published", ""),
                "latestDigest": (latest or {}).get("digest", ""),
                "latestTitle": (latest or {}).get("title", ""),
                "recentReleases": recent_releases,
                "topItems": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "kind": item.get("kind"),
                        "digest": item.get("digest"),
                    }
                    for item in sorted(engine_items, key=lambda entry: entry.get("score", 0), reverse=True)[:5]
                ],
            }
        )
    rows.sort(key=lambda row: (row["isFocus"], row["window1w"], row["total"]), reverse=True)
    return {"generatedAt": iso(reference), "rows": rows, "focusIds": sorted(focus), "summary": _competitor_summary(rows)}


def _count_since(items: Iterable[dict[str, Any]], reference: datetime, days: int) -> int:
    cutoff = reference - timedelta(days=days)
    count = 0
    for item in items:
        published = parse_datetime(item.get("published"))
        if published and published >= cutoff:
            count += 1
    return count


def _competitor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def total(prefix: str) -> int:
        return sum(row["window1w"] for row in rows if str(row["id"]).startswith(prefix))

    ascend = sum(row["window1w"] for row in rows if row.get("isFocus"))
    others = sum(row["window1w"] for row in rows if not row.get("isFocus"))
    return {
        "ascendWeeklySignals": ascend,
        "nonAscendWeeklySignals": others,
        "ratio": round(ascend / others, 3) if others else None,
        "vllmWeekly": total("vllm"),
        "sglangWeekly": total("sglang"),
        "nvidiaWeekly": total("tensorrt") + total("dynamo"),
    }


def build_capability_matrix(items: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    probes = (config.get("mindie") or {}).get("capabilityProbes") or []
    rows: list[dict[str, Any]] = []
    for probe in probes:
        terms = [str(term) for term in probe.get("terms") or []]
        matched: list[tuple[dict[str, Any], list[str]]] = []
        for item in items:
            haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            hit = keywords_in(haystack, terms)
            if hit:
                matched.append((item, hit))
        matched.sort(key=lambda pair: pair[0].get("score", 0), reverse=True)
        evidence = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "day": item.get("day"),
                "isNew": item.get("isNew"),
                "group": item.get("group"),
                "matched": hit[:4],
            }
            for item, hit in matched[:6]
        ]
        rows.append(
            {
                "id": probe.get("id"),
                "label": probe.get("label"),
                "terms": terms,
                "matches": len(matched),
                "newMatches": sum(1 for item, _ in matched if item.get("isNew")),
                "evidence": evidence,
                "coverage": "有活跃信号" if len(matched) >= 3 else ("信号稀少" if matched else "本周无信号"),
            }
        )
    gap = [row["label"] for row in rows if row["newMatches"] == 0]
    active = [row["label"] for row in rows if row["newMatches"] >= 3]
    return {"rows": rows, "gap": gap, "active": active}


def build_themes(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
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
        matched.sort(key=lambda pair: pair[0].get("score", 0), reverse=True)
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
                        "score": item.get("score"),
                        "isNew": item.get("isNew"),
                    }
                    for item, _ in matched[:6]
                ],
            }
        )
    themes.sort(key=lambda entry: (entry["newCount"], entry["count"]), reverse=True)
    return themes


def build_mindie_section(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    reference: datetime,
    release_history: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    owned = (config.get("mindie") or {}).get("ownedRepos") or []
    labels = {str(entry.get("repo", "")).lower(): entry.get("label") for entry in owned}
    history = {str(repo).lower(): list(rows) for repo, rows in (release_history or {}).items()}
    sections: list[dict[str, Any]] = []
    for repo, label in labels.items():
        repo_items = _items_for_repo(items, repo)
        repo_items.sort(key=lambda entry: entry.get("published", ""), reverse=True)
        # 发版特性：窗口内 may 无发版，回退到不受窗口限制的发版历史
        releases = [item for item in repo_items if item.get("kind") == "repo-release"]
        latest_release: dict[str, Any] = {}
        if releases:
            latest_release = {
                "tag": (releases[0].get("signals") or {}).get("tag", ""),
                "published": releases[0].get("published", ""),
                "day": releases[0].get("day", ""),
                "digest": releases[0].get("digest", ""),
                "url": releases[0].get("url", ""),
                "inWindow": True,
            }
        else:
            rows = sorted(history.get(repo, []), key=lambda entry: entry.get("published", ""), reverse=True)
            if rows:
                latest_release = {
                    "tag": rows[0].get("tag", ""),
                    "published": rows[0].get("published", ""),
                    "day": (rows[0].get("published", "") or "")[:10],
                    "digest": rows[0].get("digest", ""),
                    "url": rows[0].get("url", ""),
                    "inWindow": False,
                }
        sections.append(
            {
                "repo": repo,
                "label": label,
                "count": len(repo_items),
                "newCount": sum(1 for item in repo_items if item.get("isNew")),
                "latest": repo_items[0].get("published", "") if repo_items else "",
                "latestRelease": latest_release,
                "releaseCount": len(history.get(repo, [])),
                "items": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "day": item.get("day"),
                        "kind": item.get("kind"),
                        "isNew": item.get("isNew"),
                        "score": item.get("score"),
                        "digest": item.get("digest"),
                        "tag": (item.get("signals") or {}).get("tag", ""),
                    }
                    for item in repo_items[:8]
                ],
            }
        )
    topic_hits = []
    for item in items:
        haystack = f"{item.get('title','')} {item.get('summary','')}".lower()
        if "ascend" in haystack or "mindie" in haystack or "昇腾" in haystack:
            topic_hits.append(item)
    topic_hits.sort(key=lambda entry: entry.get("score", 0), reverse=True)
    return {
        "repos": sections,
        "ascendTopicCount": len(topic_hits),
        "ascendTopicNew": sum(1 for item in topic_hits if item.get("isNew")),
        "topicTop": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "day": item.get("day"),
                "group": item.get("group"),
                "isNew": item.get("isNew"),
            }
            for item in topic_hits[:12]
        ],
    }
