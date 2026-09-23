#!/usr/bin/env python3
"""HTTP 客户端 + 原始响应快照缓存。

设计目标：
* 只依赖标准库（urllib），无第三方包。
* 每次请求都把原始响应写入 snapshots/，保证离线可复现。
* --offline 模式下只读快照，缺快照即报错，绝不偷偷联网。
* 任一源失败都不抛到顶层，交由 collect 层降级处理。
"""

from __future__ import annotations

import gzip
import http.cookiejar
import io
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .util import UTC, dump_json, file_hash, iso, now_utc, text_hash

DEFAULT_USER_AGENT = (
    "AscendPlayground-ReportBot/1.0 "
"(+https://github.com/lynxxxi/AscendPlayground; research digest; contact: repo issues)"
)


class FetchError(RuntimeError):
    """一次 HTTP 抓取最终失败。"""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class FetchResult:
    url: str
    status: int
    body: str
    from_cache: bool
    fetched_at: str
    snapshot: Optional[str] = None
    content_type: str = ""
    error: Optional[str] = None
    attempts: int = 0
    elapsed_ms: int = 0


@dataclass
class ClientStats:
    requests: int = 0
    cache_hits: int = 0
    errors: int = 0
    by_host: dict[str, int] = field(default_factory=dict)

    def record(self, url: str) -> None:
        self.requests += 1
        host = urllib.parse.urlparse(url).netloc
        self.by_host[host] = self.by_host.get(host, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cacheHits": self.cache_hits,
            "errors": self.errors,
            "byHost": dict(sorted(self.by_host.items())),
        }


class SnapshotCache:
    """按 URL 哈希落盘的原始响应快照。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "_index.json"
        self.index: dict[str, Any] = {}
        if self.index_path.exists():
            try:
                self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.index = {}

    def key_for(self, url: str) -> str:
        return text_hash(url)[:16]

    def paths(self, url: str) -> tuple[Path, Path]:
        key = self.key_for(url)
        return self.root / f"{key}.body", self.root / f"{key}.json"

    def load(self, url: str) -> Optional[FetchResult]:
        body_path, meta_path = self.paths(url)
        if not body_path.exists() or not meta_path.exists():
            return None
        try:
            raw = body_path.read_bytes()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        # 自愈：历史版本可能落盘了未解压的 gzip 快照，视为未命中并重新抓取
        if raw[:2] == b"\x1f\x8b":
            return None
        # 失败响应（限流/超时/5xx）不作为回复源：否则一旦缓存了 403，后续运行会
        # 把它当作"成功但空结果"，导致源静默失效且永不重试。
        status = int(meta.get("status", 200))
        if status >= 400 or meta.get("error"):
            return None
        body = raw.decode("utf-8", errors="replace")
        return FetchResult(
            url=url,
            status=status,
            body=body,
            from_cache=True,
            fetched_at=str(meta.get("fetchedAt", "")),
            snapshot=meta_path.name,
            content_type=str(meta.get("contentType", "")),
            error=meta.get("error"),
        )

    def store(self, url: str, status: int, body: str, content_type: str, error: Optional[str] = None) -> str:
        body_path, meta_path = self.paths(url)
        body_path.write_text(body, encoding="utf-8")
        meta = {
            "url": url,
            "status": status,
            "contentType": content_type,
            "fetchedAt": iso(now_utc()),
            "bytes": len(body.encode("utf-8")),
            "bodyHash": file_hash(body_path),
            "error": error,
        }
        dump_json(meta_path, meta)
        self.index[self.key_for(url)] = {"url": url, "fetchedAt": meta["fetchedAt"], "status": status}
        dump_json(self.index_path, self.index)
        return meta_path.name


class HttpClient:
    def __init__(
        self,
        cache: SnapshotCache,
        *,
        offline: bool = False,
        refresh: bool = False,
        timeout: int = 30,
        retries: int = 3,
        backoff: float = 1.6,
        user_agent: str = DEFAULT_USER_AGENT,
        logger: Any = None,
    ) -> None:
        self.cache = cache
        self.offline = offline
        self.refresh = refresh
        self.timeout = timeout
        self.retries = max(1, retries)
        self.backoff = backoff
        self.user_agent = user_agent
        self.stats = ClientStats()
        self.log = logger or (lambda message: None)
        self._context = ssl.create_default_context()

    # ------------------------------------------------------------------
    # 会话 / cookie
    # ------------------------------------------------------------------
    def session_headers(self, user_agent: str, **extra: str) -> dict[str, str]:
        """为某个站点建立一个持久会话（带 cookie jar）。

        部分站点（如搜狗微信）必须先获取搜索页拿到 cookie，后续跳转请求才不被
        判定为爬虫；只带 header 而没有 cookie 会被反爬页拦截。
        """
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar),
            urllib.request.HTTPSHandler(context=self._context),
        )
        self._opener = opener
        return {"User-Agent": user_agent, **extra}

    def _request_opener(self) -> urllib.request.OpenerDirector:
        opener = getattr(self, "_opener", None)
        if opener is not None:
            return opener
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=self._context))

    # ------------------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        cache_key: Optional[str] = None,
        accept: str = "*/*",
        use_cache: bool = True,
        cache_when: Optional[Callable[[str], bool]] = None,
    ) -> FetchResult:
        """抓取 URL。

        cache_key 允许把多个不同 header 的同一 URL 区分缓存；
        use_cache=False 用于一次性/带签名的跳转链接（缓存会导致过期结果被复用）。
        cache_when 是「这个响应值不值得落快照」的判定：反爬/人机校验页虽然返回 200，
        但缓存下来会让后续 --offline 复现永远拿到空结果，因此要显式拒绝落盘。
        """
        effective_url = cache_key or url
        if use_cache and not self.refresh:
            cached = self.cache.load(effective_url)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached
        if self.offline:
            raise FetchError(
                f"offline 模式下缺少快照：{url}（请先联网执行一次 collect，或去掉 --offline）"
            )

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None
        status = 0
        started = time.time()
        for attempt in range(1, self.retries + 1):
            self.stats.record(url)
            try:
                request = urllib.request.Request(url, headers=request_headers, method="GET")
                opener = self._request_opener()
                with opener.open(request, timeout=self.timeout) as response:
                    raw = response.read()
                    body = self._decode(raw, dict(response.headers))
                    status = int(getattr(response, "status", 200) or 200)
                    content_type = response.headers.get("Content-Type", "")
                    if use_cache and (cache_when is None or cache_when(body)):
                        snapshot = self.cache.store(effective_url, status, body, content_type)
                    else:
                        snapshot = None
                    return FetchResult(
                        url=url,
                        status=status,
                        body=body,
                        from_cache=False,
                        fetched_at=iso(now_utc()),
                        snapshot=snapshot,
                        content_type=content_type,
                        attempts=attempt,
                        elapsed_ms=int((time.time() - started) * 1000),
                    )
            except urllib.error.HTTPError as error:
                status = int(error.code)
                detail = ""
                try:
                    detail = error.read(512).decode("utf-8", errors="replace")
                except Exception:  # pragma: no cover - 读取错误体失败不影响主流程
                    detail = ""
                last_error = FetchError(f"HTTP {status} {error.reason} {detail}".strip(), status)
                if status in (400, 401, 403, 404, 410, 422):
                    break
                if status == 429:
                    self.log(f"  ! 限流 429，{url[:90]}")
            except urllib.error.URLError as error:
                last_error = FetchError(f"网络错误：{error.reason}")
            except Exception as error:  # noqa: BLE001 - 任何异常都应降级为源失败
                last_error = FetchError(f"{type(error).__name__}: {error}")

            if attempt < self.retries:
                time.sleep(self.backoff ** attempt)

        self.stats.errors += 1
        if use_cache:
            self.cache.store(
                effective_url, status, "", "", error=str(last_error) if last_error else "unknown"
            )
        raise FetchError(str(last_error or "unknown error"), status or None)

    @staticmethod
    def _decode(raw: bytes, headers: dict[str, str]) -> str:
        """解码响应体。

        注意：请求时不主动声明 Accept-Encoding，交给 urllib 协商；urllib 会自动解压
        gzip/deflate 响应体。这里额外保留一次手动解压兜底，用于快照搬运等场景。
        """
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
            except (OSError, EOFError):
                pass
        charset = "utf-8"
        content_type = headers.get("Content-Type", "")
        if "charset=" in content_type.lower():
            charset = content_type.lower().split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        for candidate in (charset, "utf-8", "gb18030", "latin-1"):
            try:
                return raw.decode(candidate)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def get_json(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        cache_key: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        result = self.get(
            url,
            headers={"Accept": "application/json", **(headers or {})},
            cache_key=cache_key,
            accept="application/json",
        )
        try:
            return json.loads(result.body)
        except ValueError:
            return default
