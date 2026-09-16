#!/usr/bin/env python3
"""L1 采集层：把配置里的信息源统一抓成归一化条目。

所有信息源均为公开、免鉴权通道。任一源失败只记录错误并降级，不影响其它源。
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from report.lib.feed import Feed, FeedEntry, parse_feed
from report.lib.httpclient import DEFAULT_USER_AGENT, FetchError, HttpClient, SnapshotCache
from report.lib.relevance import Scorer, TagExtractor
from report.lib.util import (
    UTC,
    clean_title,
    commit_digest,
    day,
    digest,
    first_paragraph,
    iso,
    now_utc,
    parse_datetime,
    short_id,
    strip_html,
    summarise_abstract,
    truncate,
    within_window,
)


@dataclass
class SourceReport:
    id: str
    label: str
    group: str
    kind: str
    enabled: bool = True
    fetched: int = 0
    kept: int = 0
    newItems: int = 0
    filtered: int = 0
    droppedNoDate: int = 0
    droppedByDate: int = 0
    errors: list[str] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "kind": self.kind,
            "enabled": self.enabled,
            "fetched": self.fetched,
            "kept": self.kept,
            "newItems": self.newItems,
            "filtered": self.filtered,
            "droppedNoDate": self.droppedNoDate,
            "droppedByDate": self.droppedByDate,
            "errors": self.errors,
            "snapshots": self.snapshots,
        }


class Reporter:
    """极简日志器，静默模式用于测试。"""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)


def load_config(config_path: Path, logger: Callable[[str], None] = print) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"缺少配置文件：{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, logger, config_path)
    return config


def _validate_config(config: dict[str, Any], logger: Callable[[str], None], path: Path) -> None:
    problems: list[str] = []
    for section in ("papers", "teams"):
        for key, source in (config.get(section) or {}).items():
            if not isinstance(source, dict):
                problems.append(f"{section}.{key} 不是对象")
                continue
            if source.get("enabled", True) and not source.get("endpoint"):
                problems.append(f"{section}.{key} 缺少 endpoint")
            if source.get("kind") not in {
                "arxiv", "rss", "openalex", "hf-daily-papers", "hf-models", "github-org", "web-search",
            }:                problems.append(f"{section}.{key} 未知 kind={source.get('kind')!r}")
    for entry in (config.get("repos") or {}).get("watch", []):
        if not entry.get("repo") or "/" not in str(entry.get("repo")):
            problems.append(f"repos.watch 非法仓库名：{entry.get('repo')!r}")
    if problems:
        logger("配置校验发现问题：")
        for problem in problems:
            logger(f"  - {problem}")


def repo_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for entry in (config.get("repos") or {}).get("watch", []):
        repo = str(entry.get("repo", "")).strip()
        if repo:
            index[repo.lower()] = entry
    return index


DEFAULT_REPO_LIMITS = {
    "releasesPerRepo": 5,
    "commitsPerRepo": 1,
    "pullsPerRepo": 0,
    "releaseWatchReleases": 8,
}


def config_repo_limits(config: dict[str, Any]) -> dict[str, int]:
    """仓库采集配额；用于控制在匿名 GitHub 限流下的请求量。"""
    limits = dict(DEFAULT_REPO_LIMITS)
    limits.update({key: int(value) for key, value in ((config.get("repos") or {}).get("limits") or {}).items()})
    return limits


class Collector:
    def __init__(
        self,
        config: dict[str, Any],
        client: HttpClient,
        *,
        logger: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.client = client
        self.log = logger
        defaults = config.get("defaults") or {}
        self.max_age_days = int(defaults.get("maxAgeDays", 14))
        self.require_date = bool(defaults.get("requireDate", True))
        self.github_budget = int((config.get("repos") or {}).get("requestBudget", 45))
        self.user_agent = str(defaults.get("userAgent", "")) or None  # type: ignore[assignment]
        self.extractor = TagExtractor(config.get("keywords") or {})
        self.scorer = Scorer(config)
        self.reference = now_utc()
        self.reports: list[SourceReport] = []
        self.items: list[dict[str, Any]] = []
        self.repos = repo_index(config)
        self.github_requests = 0
        self.github_rate_limit_hits = 0
        self.github_rate_limited = False
        self.github_attempted: set[str] = set()
        # 竞品发版历史（不经过条目窗口过滤），用于构建版本节奏矩阵
        self.release_history: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # 通用包装：跑一个采集函数并统一处理错误与统计
    # ------------------------------------------------------------------
    def _run(
        self,
        source: dict[str, Any],
        handler: Callable[[dict[str, Any]], list[dict[str, Any]]],
        *,
        allow_empty: bool = True,
    ) -> None:
        source_id = str(source.get("id") or source.get("label") or "unknown")
        report = SourceReport(
            id=source_id,
            label=str(source.get("label") or source_id),
            group=str(source.get("group") or ""),
            kind=str(source.get("kind") or ""),
            enabled=bool(source.get("enabled", True)),
        )
        self.reports.append(report)
        if not report.enabled:
            self.log(f"[skip] {report.label}（已禁用）")
            return
        self.log(f"[pull] {report.label} ({report.kind})")
        try:
            raw_items = handler(source)
        except FetchError as error:
            report.errors.append(str(error))
            self.log(f"  x 抓取失败：{error}")
            return
        except Exception as error:  # noqa: BLE001 - 单源异常不应中断整轮采集
            report.errors.append(f"{type(error).__name__}: {error}")
            self.log(f"  x 解析异常：{type(error).__name__}: {error}")
            return

        report.fetched = len(raw_items)
        kept = self._finalize(raw_items, source, report)
        report.kept = len(kept)
        self.items.extend(kept)
        self.log(f"  -> 抓取 {report.fetched} 条，保留 {report.kept} 条（过滤 {report.filtered}）")

    def _finalize(
        self,
        raw_items: Iterable[dict[str, Any]],
        source: dict[str, Any],
        report: SourceReport,
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_items:
            title = clean_title(item.get("title"))
            if not title:
                report.filtered += 1
                continue
            published = parse_datetime(item.get("published"))
            if published is None:
                if self.require_date:
                    report.droppedNoDate += 1
                    report.filtered += 1
                    continue
            elif not within_window(published, self.max_age_days, self.reference):
                report.droppedByDate += 1
                report.filtered += 1
                continue
            item["title"] = title
            item["published"] = iso(published) if published else ""
            # 每条都生成一句到两句的说明，报告里可直接阅读，不必点进原文
            item["digest"] = item.get("digest") or _build_digest(item)
            text = f"{title} {item.get('summary') or ''}"
            if not self.scorer.passes_gate(text):
                report.filtered += 1
                continue
            item.setdefault("sourceId", report.id)
            item.setdefault("sourceLabel", report.label)
            item.setdefault("group", report.group)
            item.setdefault("kind", report.kind)
            item.setdefault("weight", float(source.get("weight", 1.0)))
            item["keywords"] = self.extractor.extract(text)
            item["stableId"] = item.get("stableId") or short_id(
                report.id, item.get("nativeId") or item.get("url") or title
            )
            if item["stableId"] in seen:
                report.filtered += 1
                continue
            seen.add(item["stableId"])
            item["score"] = self.scorer.score(item, self.reference)
            item["day"] = day(published)
            item["isNew"] = True
            kept.append(item)
        return kept

    # ------------------------------------------------------------------
    # arXiv
    # ------------------------------------------------------------------
    @staticmethod
    def _arxiv_query(terms: list[str]) -> str:
        parts = []
        for term in terms:
            escaped = term.replace('"', "")
            parts.append(f'all:"{escaped}"' if " " in escaped else f"all:{escaped}")
        return " AND ".join(parts)

    def _collect_arxiv(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = source["endpoint"]
        page_size = int(source.get("pageSize", 100))
        query_count = max(1, len(source.get("queries") or [1]))
        per_query = max(10, min(40, page_size // query_count))
        entries: list[dict[str, Any]] = []
        for query in source.get("queries") or []:
            terms = [str(term) for term in (query.get("terms") or []) if str(term).strip()]
            if not terms:
                continue
            params = {
                "search_query": self._arxiv_query(terms),
                "start": 0,
                "max_results": per_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            url = endpoint + "?" + urllib.parse.urlencode(params)
            self.log(f"  · {query.get('id')}: {' AND '.join(terms)}")
            try:
                result = self.client.get(url, accept="application/atom+xml")
            except FetchError as error:
                self.log(f"    x {error}")
                continue
            feed = parse_feed(result.body)
            for entry in feed.entries:
                arxiv_id = self._arxiv_id(entry)
                if not arxiv_id:
                    continue
                entries.append(
                    self._item(
                        title=entry.title,
                        summary=entry.summary,
                        url=entry.link or f"https://arxiv.org/abs/{arxiv_id}",
                        published=entry.published,
                        native_id=arxiv_id,
                        kind="paper",
                        signals={
                            "authors": (entry.author or "")[:400],
                            "query": str(query.get("id") or ""),
                            "tags": entry.tags[:8],
                        },
                    )
                )
        return entries

    @staticmethod
    def _arxiv_id(entry: FeedEntry) -> str:
        for candidate in (entry.guid, entry.link):
            if not candidate:
                continue
            match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", candidate)
            if match:
                return match.group(1) + (match.group(2) or "")
        return ""

    # ------------------------------------------------------------------
    # RSS / Atom
    # ------------------------------------------------------------------
    def _collect_rss(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        result = self.client.get(
            source["endpoint"],
            accept="application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        )
        feed: Feed = parse_feed(result.body)
        if not feed.entries:
            # 部分站点返回 HTML 索引页，退化为抽取 <a> 标题
            return self._fallback_html_links(source, result.body)
        entries: list[dict[str, Any]] = []
        for entry in feed.entries:
            entries.append(
                self._item(
                    title=entry.title,
                    summary=entry.summary,
                    url=entry.link,
                    published=entry.published,
                    native_id=entry.guid or entry.link or entry.title,
                    kind="blog",
                    signals={"author": entry.author, "tags": entry.tags[:8], "feed": feed.title},
                )
            )
        return entries

    def _fallback_html_links(self, source: dict[str, Any], body: str) -> list[dict[str, Any]]:
        base = str(source.get("baseUrl") or source["endpoint"])
        entries: list[dict[str, Any]] = []
        for match in re.finditer(r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>', body, re.I | re.S):
            href, text = match.group(1), strip_html(match.group(2))
            if len(text) < 12:
                continue
            url = urllib.parse.urljoin(base, href)
            entries.append(
                self._item(
                    title=text,
                    summary="",
                    url=url,
                    published=None,
                    native_id=url,
                    kind="blog",
                    signals={"fallback": "html-anchor"},
                )
            )
        return entries[:80]

    # ------------------------------------------------------------------
    # HuggingFace Daily Papers
    # ------------------------------------------------------------------
    def _collect_hf_daily_papers(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        url = source["endpoint"] + "?" + urllib.parse.urlencode({"limit": int(source.get("pageSize", 100))})
        payload = self.client.get_json(url, default=[])
        if not isinstance(payload, list):
            return []
        entries: list[dict[str, Any]] = []
        for row in payload:
            paper = row.get("paper") or {}
            arxiv_id = str(paper.get("id") or "")
            if not arxiv_id:
                continue
            authors = paper.get("authors") or []
            entries.append(
                self._item(
                    title=paper.get("title") or row.get("title") or "",
                    summary=paper.get("summary") or "",
                    url=f"https://huggingface.co/papers/{arxiv_id}",
                    published=paper.get("publishedAt") or row.get("publishedAt") or row.get("date"),
                    native_id=arxiv_id,
                    kind="paper",
                    signals={
                        "upvotes": row.get("upvotes") or paper.get("upvotes") or 0,
                        "authors": ", ".join(
                            str(author.get("name")) for author in authors[:8] if isinstance(author, dict)
                        ),
                        "hfPaper": True,
                    },
                )
            )
        return entries

    # ------------------------------------------------------------------
    # OpenAlex
    # ------------------------------------------------------------------
    def _collect_openalex(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        per_page = int(source.get("pageSize", 25))
        for search in source.get("searches") or []:
            params = {
                "search": str(search),
                "per-page": per_page,
                "sort": "publication_date:desc",
                "filter": "from_publication_date:"
                + (self.reference - timedelta(days=self.max_age_days)).strftime("%Y-%m-%d"),
                "mailto": "ascend-playground-report@example.com",
            }
            url = source["endpoint"] + "?" + urllib.parse.urlencode(params)
            self.log(f"  · {search}")
            try:
                payload = self.client.get_json(url, default={})
            except FetchError as error:
                self.log(f"    x {error}")
                continue
            for work in (payload or {}).get("results", []) or []:
                abstract = self._openalex_abstract(work.get("abstract_inverted_index"))
                entries.append(
                    self._item(
                        title=work.get("title") or work.get("display_name") or "",
                        summary=abstract,
                        url=work.get("doi") or work.get("id") or "",
                        published=work.get("publication_date") or work.get("publication_year"),
                        native_id=work.get("id") or work.get("doi") or "",
                        kind="paper",
                        signals={
                            "citedBy": work.get("cited_by_count") or 0,
                            "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                            "openAlex": True,
                        },
                    )
                )
        return entries

    @staticmethod
    def _openalex_abstract(inverted: Any) -> str:
        if not isinstance(inverted, dict):
            return ""
        positions: list[tuple[int, str]] = []
        for word, indexes in inverted.items():
            for index in indexes or []:
                positions.append((int(index), str(word)))
        positions.sort()
        return truncate(" ".join(word for _, word in positions), 1200)

    # ------------------------------------------------------------------
    # HuggingFace 模型发布雷达
    # ------------------------------------------------------------------
    def _collect_hf_models(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        limit = int(source.get("pageSize", 20))
        cutoff = self.reference.timestamp() - self.max_age_days * 86400

        def harvest(rows: Any) -> None:
            for row in rows or []:
                model_id = str(row.get("modelId") or row.get("id") or "")
                if not model_id or model_id in seen:
                    continue
                created = parse_datetime(row.get("createdAt"))
                if created and created.timestamp() < cutoff:
                    continue
                seen.add(model_id)
                entries.append(
                    self._item(
                        title=f"新模型发布：{model_id}",
                        summary=" / ".join(str(tag) for tag in (row.get("tags") or [])[:12]),
                        url=f"https://huggingface.co/{model_id}",
                        published=row.get("createdAt"),
                        native_id=model_id,
                        kind="model-release",
                        signals={
                            "downloads": row.get("downloads") or 0,
                            "likes": row.get("likes") or 0,
                            "pipeline": row.get("pipeline_tag") or "",
                            "org": model_id.split("/")[0],
                        },
                    )
                )

        for org in source.get("orgs") or []:
            params = {"author": org, "sort": "createdAt", "direction": "-1", "limit": limit}
            url = source["endpoint"] + "?" + urllib.parse.urlencode(params)
            try:
                harvest(self.client.get_json(url, default=[]))
            except FetchError as error:
                self.log(f"  x 组织 {org}: {error}")

        for term in source.get("searchTerms") or []:
            params = {
                "search": str(term),
                "sort": "trendingScore",
                "direction": "-1",
                "limit": limit,
            }
            url = source["endpoint"] + "?" + urllib.parse.urlencode(params)
            try:
                harvest(self.client.get_json(url, default=[]))
            except FetchError as error:
                self.log(f"  x 搜索 {term}: {error}")
        return entries

    # ------------------------------------------------------------------
    # GitHub 仓库动态
    #
    # 默认走 Atom feed（github.com/<repo>/releases.atom 与 commits.atom）：
    # 无 API 限流、无需鉴权、含完整 release 说明与 commit 信息，是最稳的通道。
    # 若显式配置 repos.channel = "api"，则回退到 REST API（受匿名 60 次/小时限制）。
    # ------------------------------------------------------------------
    def _github_get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        url = f"https://api.github.com{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        cached = self.client.cache.load(url)
        # 命中本地快照不消耗配额；同一个 URL 的重试也不重复计数。
        if cached is None and url not in self.github_attempted:
            if self.github_rate_limited:
                raise FetchError("本轮已判定匿名限流生效，停止后续 GitHub 请求", 403)
            if self.github_requests >= self.github_budget:
                raise FetchError(f"已达本轮 GitHub 请求预算（{self.github_budget}），跳过 {path}", 429)
            self.github_attempted.add(url)
            self.github_requests += 1
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        try:
            return self.client.get_json(url, headers=headers, cache_key=url, default=None)
        except FetchError as error:
            if error.status == 403 and "rate limit" in str(error).lower():
                self.github_rate_limit_hits += 1
                if self.github_rate_limit_hits >= 2:
                    self.github_rate_limited = True
            raise

    def _github_atom(self, repo: str, feed_name: str) -> list[FeedEntry]:
        url = f"https://github.com/{repo}/{feed_name}.atom"
        result = self.client.get(url, accept="application/atom+xml, application/xml, */*")
        return parse_feed(result.body).entries

    def _collect_github(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        channel = str((self.config.get("repos") or {}).get("channel") or "atom").lower()
        if channel == "api":
            return self._collect_github_api(source)
        return self._collect_github_atom(source)

    def _collect_github_atom(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        failures: list[str] = []
        watch = [entry for entry in (source.get("watch") or []) if entry.get("enabled") is not False]
        limits = config_repo_limits(self.config)
        max_commits = int(limits.get("commitsPerRepo", 1)) * 20
        self.log(f"  · Atom 通道（无限流）/ {len(watch)} 个仓库，每仓最多 {max_commits} 条提交")

        for entry in watch:
            repo = str(entry.get("repo") or "")
            if "/" not in repo:
                continue
            if entry.get("enabled") is False:
                continue
            meta = self.repos.get(repo.lower(), entry)
            group = str(meta.get("group") or entry.get("group") or "")
            note = str(meta.get("note") or entry.get("note") or "")
            label = group_label(self.config, group)

            try:
                releases = self._github_atom(repo, "releases")
            except FetchError as error:
                failures.append(f"{repo} releases.atom: {error}")
                releases = []
            for release in releases[: int(limits.get("releasesPerRepo", 5))]:
                tag, _, subject = release.title.partition(":")
                entries.append(
                    self._item(
                        title=f"[{repo}] 发布 {tag.strip() or release.title}：{clean_title(subject)}",
                        summary=truncate(release.summary or subject, 4000),
                        url=release.link,
                        published=release.published,
                        native_id=release.guid or release.link,
                        kind="repo-release",
                        signals={
                            "repo": repo,
                            "tag": tag.strip(),
                            "author": release.author,
                        },
                        extra={
                            "repo": repo,
                            "repoGroup": group,
                            "repoNote": note,
                            "repoGroupLabel": label,
                            "repoActivity": "release",
                        },
                    )
                )

            try:
                commits = self._github_atom(repo, "commits")
            except FetchError as error:
                failures.append(f"{repo} commits.atom: {error}")
                commits = []
            kept_commits = 0
            for commit in commits:
                message = clean_title(commit.title)
                if not message or not is_interesting_commit(message):
                    continue
                entries.append(
                    self._item(
                        title=f"[{repo}] {message}",
                        summary=truncate(commit.summary or "", 900),
                        url=commit.link,
                        published=commit.published,
                        native_id=commit.guid or commit.link,
                        kind="repo-commit",
                        signals={"repo": repo, "author": commit.author, "subject": message},
                        extra={
                            "repo": repo,
                            "repoGroup": group,
                            "repoNote": note,
                            "repoGroupLabel": label,
                            "repoActivity": "commit",
                        },
                    )
                )
                kept_commits += 1
                if kept_commits >= max_commits:
                    break

        if failures:
            self.log(f"  ! {len(failures)} 个 Atom feed 拉取失败（已降级，保留成功条目）")
            for failure in failures[:4]:
                self.log(f"    - {failure[:160]}")
            if self.reports:
                self.reports[-1].errors.extend(failures[:10])
        return entries

    def _collect_github_api(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        failures: list[str] = []
        since = (self.reference - timedelta(days=self.max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        watch = list(source.get("watch") or [])
        repos_cfg = config_repo_limits(self.config)
        skipped: list[str] = []
        self.log(f"  · REST 通道，预算 {self.github_budget} 次请求 / {len(watch)} 个仓库（release 优先）")

        for entry in watch:
            repo = str(entry.get("repo") or "")
            if "/" not in repo:
                continue
            if self.github_rate_limited:
                skipped.append(repo)
                continue
            remaining = self.github_budget - self.github_requests
            if remaining <= 0:
                skipped.append(repo)
                continue
            # 按剩余预算裁剪每个仓库的请求数（release 永远优先）
            want_releases = 1 if remaining >= 1 else 0
            want_commits = 1 if (remaining >= 2 and int(repos_cfg.get("commitsPerRepo", 1)) > 0) else 0
            want_pulls = 1 if (remaining >= 3 and int(repos_cfg.get("pullsPerRepo", 0)) > 0) else 0
            if not want_releases and not want_commits:
                skipped.append(repo)
                continue
            meta = self.repos.get(repo.lower(), entry)
            group = str(meta.get("group") or entry.get("group") or "")
            note = str(meta.get("note") or entry.get("note") or "")
            label = group_label(self.config, group)

            if want_releases:
                try:
                    releases = self._github_get(
                        f"/repos/{repo}/releases", {"per_page": int(repos_cfg.get("releasesPerRepo", 5))}
                    )
                except FetchError as error:
                    failures.append(f"{repo} releases: {error}")
                    releases = None
                for release in releases or []:
                    entries.append(
                        self._item(
                            title=f"[{repo}] 发布 {release.get('tag_name') or release.get('name') or ''}"
                            f"：{clean_title(release.get('name') or '')}",
                            summary=truncate(strip_html(release.get("body") or ""), 900),
                            url=release.get("html_url") or f"https://github.com/{repo}/releases",
                            published=release.get("published_at") or release.get("created_at"),
                            native_id=f"{repo}@{release.get('id')}",
                            kind="repo-release",
                            signals={"repo": repo, "prerelease": bool(release.get("prerelease"))},
                            extra={
                                "repo": repo,
                                "repoGroup": group,
                                "repoNote": note,
                                "repoGroupLabel": label,
                                "repoActivity": "release",
                            },
                        )
                    )

            if want_commits:
                try:
                    commits = self._github_get(
                        f"/repos/{repo}/commits",
                        {"since": since, "per_page": int(repos_cfg.get("commitsPerRepo", 1)) * 8},
                    )
                except FetchError as error:
                    failures.append(f"{repo} commits: {error}")
                    commits = None
                for commit in commits or []:
                    info = commit.get("commit") or {}
                    message = str(info.get("message") or "").splitlines()[0]
                    if not is_interesting_commit(message):
                        continue
                    entries.append(
                        self._item(
                            title=f"[{repo}] {clean_title(message)}",
                            summary=truncate(str(info.get("message") or ""), 400),
                            url=commit.get("html_url") or f"https://github.com/{repo}/commits",
                            published=((info.get("committer") or {}).get("date"))
                            or ((info.get("author") or {}).get("date")),
                            native_id=f"{repo}#{commit.get('sha')}",
                            kind="repo-commit",
                            signals={"repo": repo, "author": (commit.get("author") or {}).get("login") or ""},
                            extra={
                                "repo": repo,
                                "repoGroup": group,
                                "repoNote": note,
                                "repoGroupLabel": label,
                                "repoActivity": "commit",
                            },
                        )
                    )

            if want_pulls:
                try:
                    pulls = self._github_get(
                        f"/repos/{repo}/pulls",
                        {
                            "state": "all",
                            "sort": "updated",
                            "direction": "desc",
                            "per_page": int(repos_cfg.get("pullsPerRepo", 0)) * 4,
                        },
                    )
                except FetchError as error:
                    failures.append(f"{repo} pulls: {error}")
                    pulls = None
                recent_cutoff = self.reference.timestamp() - self.max_age_days * 86400
                for pull in pulls or []:
                    updated = parse_datetime(pull.get("updated_at"))
                    if updated and updated.timestamp() < recent_cutoff:
                        continue
                    state = "已合并" if pull.get("merged_at") else ("开启" if pull.get("state") == "open" else "关闭")
                    entries.append(
                        self._item(
                            title=f"[{repo}] PR {state}：{clean_title(pull.get('title') or '')}",
                            summary=truncate(strip_html(pull.get("body") or ""), 400),
                            url=pull.get("html_url") or "",
                            published=pull.get("merged_at") or pull.get("updated_at") or pull.get("created_at"),
                            native_id=f"{repo}!{pull.get('number')}",
                            kind="repo-pr",
                            signals={
                                "repo": repo,
                                "user": (pull.get("user") or {}).get("login") or "",
                                "state": state,
                            },
                            extra={
                                "repo": repo,
                                "repoGroup": group,
                                "repoNote": note,
                                "repoGroupLabel": label,
                                "repoActivity": "pr",
                            },
                        )
                    )

        if skipped:
            self.log(
                f"  ! GitHub 预算耗尽，本轮未覆盖 {len(skipped)} 个仓库："
                f"{', '.join(skipped[:6])}{' …' if len(skipped) > 6 else ''}"
            )
            self.log("    （缓存有效期内重跑会复用快照，不会重复消耗配额）")
        if failures:
            degraded = sum(1 for failure in failures if "rate limit" in failure.lower())
            self.log(f"  ! {len(failures)} 个 GitHub 子请求失败（已降级，保留成功条目）")
            if degraded:
                self.log(f"    （其中 {degraded} 个为匿名限流；可调小 repos.limits 或稍后重跑）")
            for failure in failures[:4]:
                self.log(f"    - {failure[:160]}")
            if self.reports:
                self.reports[-1].errors.extend(failures[:10])
        return entries

    # ------------------------------------------------------------------
    # 微信公众号（搜狗微信检索）
    # ------------------------------------------------------------------
    def _collect_sogou_wechat(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = source["endpoint"]
        search_type = str(source.get("type", 2))
        delay = float(source.get("politeDelaySeconds", 0) or 0)
        resolve = bool(source.get("resolveLinks", True))
        max_resolve = int(source.get("maxResolvePerQuery", 20))
        # 搜狗必须先建立 cookie 会话，否则跳转请求会被反爬页拦截
        self.client.session_headers(BROWSER_USER_AGENT)
        entries: list[dict[str, Any]] = []
        for index, search in enumerate(self.config.get("wechat", {}).get("searches", []) or []):
            query = str(search.get("query") or "").strip()
            if not query:
                continue
            params = {"type": search_type, "query": query, "ie": "utf8"}
            url = endpoint + "?" + urllib.parse.urlencode(params)
            self.log(f"  · [{search.get('id')}] {query}")
            if index and delay:
                time.sleep(delay)
            try:
                result = self.client.get(
                    url,
                    headers={"Referer": "https://weixin.sogou.com/"},
                    accept="text/html,application/xhtml+xml",
                    # 搜索结果里的跳转链接与会话 cookie 绑定，复用旧快照会导致解析失败
                    use_cache=False,
                )
            except FetchError as error:
                self.log(f"    x {error}")
                continue
            hits = parse_sogou_wechat(result.body)
            if not hits:
                self.log("    ! 未解析到结果（可能触发人机校验）")
            resolved = 0
            for hit_index, hit in enumerate(hits):
                article_url = hit["url"]
                if resolve and hit["url"].startswith("https://weixin.sogou.com/link") and hit_index < max_resolve:
                    if delay:
                        time.sleep(delay)
                    direct = self._resolve_sogou_link(hit["url"], referer=url)
                    if direct:
                        article_url = direct
                        resolved += 1
                entries.append(
                    self._item(
                        title=hit["title"],
                        summary=hit.get("summary", ""),
                        url=article_url,
                        published=hit.get("published"),
                        native_id=hit["url"],
                        kind="wechat-article",
                        signals={
                            "query": query,
                            "engine": "sogou-wechat",
                            "account": hit.get("account", ""),
                            "searchUrl": url,
                            "linkResolved": article_url != hit["url"],
                        },
                    )
                )
            if resolve and hits:
                self.log(f"    ↳ 原文直链解析 {resolved}/{len(hits)}")
        return entries

    def _resolve_sogou_link(self, link: str, *, referer: str) -> str:
        """把搜狗 /link?url=... 跳转解析为 mp.weixin.qq.com 原文直链。

        关键点：搜狗返回的 href 内嵌换行等控制字符，必须先清理，否则请求非法；
        跳转页用 `url += '...'` 分片拼接真实地址。
        """
        target = re.sub(r"[\s\x00-\x1f]+", "", link)
        try:
            page = self.client.get(
                target,
                headers={"Referer": referer},
                accept="text/html,application/xhtml+xml",
                use_cache=False,
            ).body
        except FetchError:
            return ""
        fragments = re.findall(r"url\s*\+=\s*'([^']*)'", page)
        if not fragments:
            return ""
        raw = "".join(fragments)
        # 注意顺序：查询串里的 "&timestamp=" 会被 html.unescape 当成 "&times;" 实体
        # 变成 "×tamp="，因此先把被误转义的形式还原，再做实体解码。
        raw = raw.replace("&times;tamp", "&timestamp").replace("\u00d7tamp", "&timestamp")
        candidate = html.unescape(raw).replace("@", "")
        candidate = candidate.replace("\u00d7tamp", "&timestamp")
        match = re.search(r"https?://mp\.weixin\.qq\.com/s[^\s\"'<>\\]*", candidate)
        return re.sub(r"[\s\x00-\x1f]+", "", match.group(0)) if match else ""

    # ------------------------------------------------------------------
    # 通用搜索引擎（默认关闭；适配器保留供有鉴权通道时启用）
    # ------------------------------------------------------------------
    def _collect_web_search(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        site_filter = source.get("siteFilter")
        engine_id = str(source.get("id") or "")
        parser = self._parse_bing if "bing" in engine_id else self._parse_duckduckgo
        for search in self.config.get("wechat", {}).get("searches", []) or []:
            query = str(search.get("query") or "").strip()
            if not query:
                continue
            scoped = f"site:{site_filter} {query}" if site_filter else query
            params: dict[str, Any] = {"q": scoped}
            if source.get("market"):
                params["mkt"] = source["market"]
            params["setlang"] = "zh-hans" if source.get("market") else "en"
            url = source["base"] + "?" + urllib.parse.urlencode(params)
            self.log(f"  · [{search.get('id')}] {scoped}")
            try:
                result = self.client.get(url, accept="text/html,application/xhtml+xml")
            except FetchError as error:
                self.log(f"    x {error}")
                continue
            for hit in parser(result.body):
                entries.append(
                    self._item(
                        title=hit["title"],
                        summary=hit.get("summary", ""),
                        url=hit["url"],
                        published=hit.get("published"),
                        native_id=hit["url"],
                        kind="wechat-article" if site_filter else "media-article",
                        signals={"query": query, "engine": engine_id, "site": site_filter or ""},
                    )
                )
        return entries

    @staticmethod
    def _parse_bing(body: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for block in re.split(r'<li class="b_algo"', body)[1:]:
            link_match = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.I | re.S)
            if not link_match:
                continue
            url = html.unescape(link_match.group(1))
            title = strip_html(link_match.group(2))
            if not title or not url.startswith("http"):
                continue
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.I | re.S)
            hits.append(
                {
                    "title": title,
                    "url": url,
                    "summary": strip_html(snippet_match.group(1)) if snippet_match else "",
                }
            )
        return hits

    @staticmethod
    def _parse_duckduckgo(body: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
            r'(?:.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?',
            re.I | re.S,
        )
        for match in pattern.finditer(body):
            url = html.unescape(match.group(1))
            if url.startswith("//duckduckgo.com/l/"):
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + url).query)
                url = (parsed.get("uddg") or [url])[0]
            title = strip_html(match.group(2))
            if not title:
                continue
            hits.append({"title": title, "url": url, "summary": strip_html(match.group(3) or "")})
        return hits

    # ------------------------------------------------------------------
    # 竞品发版观测（只取 release tag 与时间，用于版本节奏矩阵）
    # ------------------------------------------------------------------
    def _collect_competitor_releases(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        channel = str((self.config.get("repos") or {}).get("channel") or "atom").lower()
        entries: list[dict[str, Any]] = []
        failures: list[str] = []
        limits = config_repo_limits(self.config)
        watch = list(source.get("releaseWatch") or [])
        self.log(f"  · {'Atom' if channel != 'api' else 'REST'} 通道 / {len(watch)} 个引擎")

        for repo in watch:
            if channel != "api":
                try:
                    releases = self._github_atom(repo, "releases")
                except FetchError as error:
                    failures.append(f"{repo}: {error}")
                    continue
                for release in releases[: limits["releaseWatchReleases"]]:
                    tag, _, subject = release.title.partition(":")
                    tag = tag.strip() or release.title
                    published = parse_datetime(release.published)
                    feature = digest(release.summary, limit=420, prefer_bullets=True) or digest(subject, limit=200)
                    # 版本历史用于竞品矩阵的"最新版本 + 特性说明"；发版往往早于采集窗口，
                    # 因此这里独立记录，不依赖条目是否被窗口过滤器保留。
                    self.release_history.setdefault(repo, []).append(
                        {
                            "tag": tag,
                            "published": iso(published) if published else "",
                            "url": release.link,
                            "digest": feature,
                        }
                    )
                    entries.append(
                        self._item(
                            title=f"[{repo}] release {tag}",
                            summary=truncate(release.summary or subject, 4000),
                            url=release.link,
                            published=release.published,
                            native_id=release.guid or release.link,
                            kind="competitor-release",
                            signals={"repo": repo, "tag": tag, "author": release.author},
                            extra={"repo": repo, "repoActivity": "release"},
                        )
                    )
                continue

            if self.github_rate_limited:
                failures.append(f"（限流中，跳过 {repo}）")
                continue
            try:
                payload = self._github_get(
                    f"/repos/{repo}/releases", {"per_page": limits["releaseWatchReleases"]}
                )
            except FetchError as error:
                failures.append(f"{repo}: {error}")
                continue
            for release in payload or []:
                tag = release.get("tag_name") or release.get("name") or ""
                if not tag:
                    continue
                published = parse_datetime(release.get("published_at") or release.get("created_at"))
                body_text = strip_html(release.get("body") or "")
                self.release_history.setdefault(repo, []).append(
                    {
                        "tag": tag,
                        "published": iso(published) if published else "",
                        "url": release.get("html_url") or f"https://github.com/{repo}/releases",
                        "digest": digest(body_text, limit=420, prefer_bullets=True),
                    }
                )
                entries.append(
                    self._item(
                        title=f"[{repo}] release {tag}",
                        summary=truncate(body_text, 4000),
                        url=release.get("html_url") or f"https://github.com/{repo}/releases",
                        published=release.get("published_at") or release.get("created_at"),
                        native_id=f"{repo}@{release.get('id')}",
                        kind="competitor-release",
                        signals={"repo": repo, "tag": tag, "prerelease": bool(release.get("prerelease"))},
                        extra={"repo": repo, "repoActivity": "release"},
                    )
                )
        if failures:
            self.log(f"  ! {len(failures)} 个竞品仓库发版查询失败（已降级）")
            for failure in failures[:3]:
                self.log(f"    - {failure[:160]}")
            if self.reports:
                self.reports[-1].errors.extend(failures[:10])
        return entries

    # ------------------------------------------------------------------
    # 条目工厂
    # ------------------------------------------------------------------
    @staticmethod
    def _item(
        *,
        title: str,
        summary: str,
        url: str,
        published: Any,
        native_id: str,
        kind: str,
        signals: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "title": strip_html(title or ""),
            "summary": truncate(strip_html(summary or ""), 1200),
            "url": str(url or "").strip(),
            "published": published,
            "nativeId": str(native_id or ""),
            "kind": kind,
            "signals": signals or {},
        }
        if extra:
            item.update(extra)
        return item


def group_label(config: dict[str, Any], group: str) -> str:
    return str(((config.get("repos") or {}).get("groups") or {}).get(group) or group)


def _build_digest(item: dict[str, Any]) -> str:
    """按条目类型抽取可读说明。"""
    kind = str(item.get("kind") or "")
    summary = str(item.get("summary") or "")
    title = str(item.get("title") or "")
    if kind == "paper":
        return summarise_abstract(summary)
    if kind in {"repo-release", "competitor-release"}:
        return digest(summary, limit=360, prefer_bullets=True)
    if kind == "repo-commit":
        return commit_digest(item.get("signals", {}).get("subject") or title, summary)
    if kind == "model-release":
        return digest(summary, limit=200)
    return digest(summary, limit=260) or digest(title, limit=160)


INTERESTING_COMMIT_HINTS = (
    "feat",
    "fix",
    "perf",
    "opt",
    "support",
    "add",
    "bump",
    "update",
    "refactor",
    "revert",
    "quant",
    "kernel",
    "attention",
    "cache",
    "multimodal",
    "vlm",
    "video",
    "image",
    "parallel",
    "moe",
    "ascend",
    "npu",
    "mindie",
    "release",
)

# 这些前缀表示纯工程杂务，不构成技术信号
COMMIT_NOISE_PREFIXES = (
    "[docs]",
    "[doc]",
    "[ci]",
    "[build]",
    "[test]",
    "[style]",
    "[chore]",
    "[branding]",
    "[misc]",
    "[skip ci]",
    "docs:",
    "doc:",
    "style:",
    "test:",
    "ci:",
    "build:",
    "chore:",
    "typo",
)
COMMIT_NOISE_PHRASES = (
    "message auto-generated",
    "no-merge-commit",
    "update readme",
    "update logo",
    "update license",
    "codecheck",
    "cleancode",
    "fix typo",
    "bump version",
)


def is_interesting_commit(message: str) -> bool:
    """过滤掉纯杂务提交，只保留具备技术信号的变更。"""
    lowered = (message or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith(COMMIT_NOISE_PREFIXES):
        return False
    if any(phrase in lowered for phrase in COMMIT_NOISE_PHRASES):
        return False
    return any(hint in lowered for hint in INTERESTING_COMMIT_HINTS)


SOGOU_BLOCK_RE = re.compile(r'<li[^>]+id="sogou_vr_11002601_box_\d+"', re.IGNORECASE)

# 需要 cookie 会话 + 浏览器指纹的站点（搜狗微信）使用真实浏览器 UA
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SOGOU_TIME_RE = re.compile(r"timeConvert\('(\d{9,12})'\)")
_SOGOU_NOISE = re.compile(r"<!--.*?-->|<!--red_beg-->|<!--red_end-->", re.S)


def parse_sogou_wechat(body: str) -> list[dict[str, Any]]:
    """解析搜狗微信搜索结果页，抽取标题 / 摘要 / 公众号 / 发布时间。"""
    if not body:
        return []
    hits: list[dict[str, Any]] = []
    blocks = SOGOU_BLOCK_RE.split(body)[1:]
    for block in blocks:
        title_match = re.search(r"<h3>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.S)
        if not title_match:
            continue
        href = html.unescape(title_match.group(1)).strip()
        title = clean_sogou_text(title_match.group(2))
        if not title:
            continue
        snippet_match = re.search(r'<p class="txt-info"[^>]*>(.*?)</p>', block, re.S)
        summary = clean_sogou_text(snippet_match.group(1)) if snippet_match else ""
        account_match = re.search(r'class="account"[^>]*>(.*?)</a>', block, re.S)
        account = clean_sogou_text(account_match.group(1)) if account_match else ""
        published = None
        time_match = SOGOU_TIME_RE.search(block)
        if time_match:
            try:
                published = datetime.fromtimestamp(int(time_match.group(1)), UTC)
            except (OverflowError, OSError, ValueError):
                published = None
        hits.append(
            {
                "title": title,
                "summary": summary,
                "account": account,
                "published": published,
                "url": urllib.parse.urljoin("https://weixin.sogou.com/", href) if href.startswith("/") else href,
            }
        )
    return hits


def clean_sogou_text(raw: str) -> str:
    """搜狗结果里带 <em> 高亮与注释噪声，统一清洗为纯文本。"""
    text = _SOGOU_NOISE.sub(" ", raw or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\u200b", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------

SOURCE_SECTIONS = ("papers", "teams")


def build_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    """把配置展开成扁平的待采集源列表（含动态生成的 GitHub 与搜索源）。"""
    sources: list[dict[str, Any]] = []
    for section in SOURCE_SECTIONS:
        for key, source in (config.get(section) or {}).items():
            entry = dict(source)
            entry.setdefault("id", key)
            entry.setdefault("group", section)
            sources.append(entry)

    repos = config.get("repos") or {}
    competitors_cfg = config.get("competitors") or {}
    # 竞品发版观测只需要少量请求，先跑以保证维度 E 的数据完整性；
    # 仓库细则随后使用剩余预算。
    if competitors_cfg.get("releaseWatch"):
        sources.append(
            {
                "id": "competitor-releases",
                "kind": "competitor-releases",
                "label": f"竞品发版观测（{len(competitors_cfg['releaseWatch'])} 个引擎）",
                "group": "competitors",
                "weight": 1.3,
                "enabled": bool(competitors_cfg.get("enabled", True)),
                "releaseWatch": competitors_cfg["releaseWatch"],
            }
        )
    if repos.get("watch"):
        sources.append(
            {
                "id": "github-repos",
                "kind": "github",
                "label": f"GitHub 仓库动态（{len(repos['watch'])} 个）",
                "group": "repos",
                "weight": 1.25,
                "enabled": bool(repos.get("enabled", True)),
                "watch": repos["watch"],
            }
        )

    wechat = config.get("wechat") or {}
    primary = wechat.get("primary")
    if primary:
        entry = dict(primary)
        entry["kind"] = "sogou-wechat"
        entry["group"] = "wechat"
        entry["enabled"] = bool(wechat.get("enabled", True))
        sources.append(entry)
    for mirror in wechat.get("mediaMirrors", []) or []:
        entry = dict(mirror)
        entry.setdefault("kind", "rss")
        entry["group"] = "wechat"
        entry.setdefault("enabled", True)
        sources.append(entry)
    for engine in wechat.get("disabledEngines", []) or []:
        entry = dict(engine)
        entry["kind"] = entry.get("kind") or "web-search"
        entry["group"] = "wechat"
        entry["enabled"] = bool(entry.get("enabled", False))
        sources.append(entry)
    return sources


def collect_all(
    config: dict[str, Any],
    client: HttpClient,
    *,
    logger: Callable[[str], None] = print,
    only: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], list[SourceReport], dict[str, list[dict[str, Any]]]]:
    collector = Collector(config, client, logger=logger)
    handlers: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
        "arxiv": collector._collect_arxiv,
        "rss": collector._collect_rss,
        "hf-daily-papers": collector._collect_hf_daily_papers,
        "openalex": collector._collect_openalex,
        "hf-models": collector._collect_hf_models,
        "github": collector._collect_github,
        "github-org": collector._collect_github,
        "competitor-releases": collector._collect_competitor_releases,
        "sogou-wechat": collector._collect_sogou_wechat,
        "web-search": collector._collect_web_search,
    }
    for source in build_sources(config):
        if only and str(source.get("id")) not in only:
            continue
        handler = handlers.get(str(source.get("kind")))
        if handler is None:
            logger(f"[skip] {source.get('id')}：未实现的 kind={source.get('kind')!r}")
            continue
        collector._run(source, handler)
    return collector.items, collector.reports, collector.release_history


def snapshot_cache(root: Path) -> SnapshotCache:
    return SnapshotCache(root)
