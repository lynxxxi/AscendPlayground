#!/usr/bin/env python3
"""AscendPlayground 多模态 Infra 周报 · 一键入口。

周报是**两步**流程（报告只出中文，未补中文的条目不进正文）：

    1) 采集：python scripts/report/run_weekly.py            # 联网抓取 + 生成待补中文清单
       清单写在 scripts/report/.cache/pending_zh.json
    2) 补中文：把 {stableId: {title, digest}} 写进 config/curated_zh.json
    3) 渲染：python scripts/report/run_weekly.py --offline  # 只读快照，产出全中文报告

用法：
    python scripts/report/run_weekly.py                    # 联网采集并生成本周报告
    python scripts/report/run_weekly.py --refresh           # 忽略快照，强制重新抓取
    python scripts/report/run_weekly.py --offline           # 只用快照离线复现（不联网）
    python scripts/report/run_weekly.py --max-age-days 30   # 放宽采集窗口
    python scripts/report/run_weekly.py --only arxiv,github-repos
    python scripts/report/run_weekly.py --list-sources      # 列出全部信息源
    python scripts/report/run_weekly.py --dry-run           # 只采集与统计，不写文件

产出（report/ 下只有报告本身）：
    report/<week>.html          当周可视化报告（自包含，可离线打开）

采集缓存（非产物，可安全删除；仅用于 --offline 复现）：
    scripts/report/.cache/snapshots/
    scripts/report/.cache/pending_zh.json   待补中文清单
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from report.collect import build_sources, collect_all, load_config  # noqa: E402
from report.corpus import build_corpus  # noqa: E402
from report.lib.httpclient import DEFAULT_USER_AGENT, HttpClient, SnapshotCache  # noqa: E402
from report.lib.util import (  # noqa: E402
    UTC,
    dump_json,
    ensure_dir,
    iso,
    now_utc,
    parse_datetime,
    relative_display,
    week_key,
)
from report.render import render_html  # noqa: E402

REPO_ROOT = SCRIPTS_ROOT.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "sources.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "report"
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".cache"
TOOL_VERSION = "1.2.0"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 AscendPlayground 多模态 Infra 周报（HTML 输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="信息源配置文件")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="报告输出目录（只放产物）")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="采集缓存目录（快照，非产物）")
    parser.add_argument("--offline", action="store_true", help="只用快照离线复现，绝不联网")
    parser.add_argument("--refresh", action="store_true", help="忽略已有快照，强制重新抓取")
    parser.add_argument("--max-age-days", type=int, default=None, help="采集时间窗口（天）")
    parser.add_argument("--top", type=int, default=None, help="每个维度最多列出多少条")
    parser.add_argument("--only", default="", help="只采集指定 source id（逗号分隔）")
    parser.add_argument("--skip", default="", help="跳过指定 source id（逗号分隔）")
    parser.add_argument("--run-id", default=None, help="自定义 Run ID（默认按当前 UTC 时间生成）")
    parser.add_argument(
        "--generated-at",
        default=None,
        help="冻结生成时间（ISO8601，如 2026-09-16T00:00:00Z）。同一 Run ID + 同一快照 + 同一时间即为逐字节可复现。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只采集与统计，不写任何文件")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    parser.add_argument("--list-sources", action="store_true", help="列出全部信息源后退出")
    return parser.parse_args(argv)


def make_run_id(explicit: Optional[str] = None) -> str:
    return explicit or now_utc().strftime("%Y-%m-%dT%H%M%SZ")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = (lambda message: None) if args.quiet else (lambda message: print(message, flush=True))

    config_path = args.config.resolve()
    report_dir = args.report_dir.resolve()

    try:
        config = load_config(config_path, logger=log)
    except (OSError, ValueError) as error:
        print(f"配置加载失败：{error}", file=sys.stderr)
        return 2

    sources = build_sources(config)
    if args.list_sources:
        print(f"{'ID':<24} {'KIND':<20} {'GROUP':<12} LABEL")
        for source in sources:
            enabled = "" if source.get("enabled", True) else "  (disabled)"
            print(
                f"{str(source.get('id')):<24} {str(source.get('kind')):<20} "
                f"{str(source.get('group')):<12} {source.get('label')}{enabled}"
            )
        print(f"\n共 {len(sources)} 个信息源。")
        return 0

    defaults = config.get("defaults") or {}
    if args.max_age_days is not None:
        defaults["maxAgeDays"] = args.max_age_days
    config["defaults"] = defaults
    max_age_days = int(defaults.get("maxAgeDays", 14))
    top_per_section = int(args.top if args.top is not None else defaults.get("topPerSection", 25))

    only = [token.strip() for token in args.only.split(",") if token.strip()]
    skip = {token.strip() for token in args.skip.split(",") if token.strip()}
    if skip:
        for source in sources:
            if str(source.get("id")) in skip:
                source["enabled"] = False

    run_id = make_run_id(args.run_id)
    reference = parse_datetime(args.generated_at) or now_utc()
    week = week_key(reference)
    window = {
        "since": (reference - timedelta(days=max_age_days)).strftime("%Y-%m-%d"),
        "until": reference.strftime("%Y-%m-%d"),
        "maxAgeDays": max_age_days,
    }

    ensure_dir(report_dir)
    snapshot_dir = ensure_dir(args.cache_dir.resolve() / "snapshots")
    client = HttpClient(
        SnapshotCache(snapshot_dir),
        offline=args.offline,
        refresh=args.refresh,
        timeout=int(defaults.get("timeoutSeconds", 30)),
        retries=int(defaults.get("retries", 3)),
        backoff=float(defaults.get("retryBackoffSeconds", 1.6)),
        user_agent=str(defaults.get("userAgent") or "").strip() or DEFAULT_USER_AGENT,
        logger=log,
    )

    command = "python scripts/report/run_weekly.py"
    extra_flags: list[str] = []
    if args.offline:
        extra_flags.append("--offline")
    if args.refresh:
        extra_flags.append("--refresh")
    if args.max_age_days is not None:
        extra_flags.extend(["--max-age-days", str(args.max_age_days)])
    if args.only:
        extra_flags.extend(["--only", args.only])
    if args.generated_at:
        extra_flags.extend(["--generated-at", args.generated_at])
    if extra_flags:
        command += " " + " ".join(extra_flags)

    log(f"=== AscendPlayground 多模态 Infra 周报 · {week} · run {run_id} ===")
    log(f"信息源 {len(sources)} 个 | 窗口 {window['since']} ~ {window['until']}（{max_age_days} 天）")
    if args.offline:
        log("模式：offline（只读快照，不联网）")
    elif args.refresh:
        log("模式：refresh（忽略旧快照，强制重新抓取）")

    try:
        items, reports, release_history = collect_all(config, client, logger=log, only=only or None)
    except Exception as error:  # noqa: BLE001 - 顶层兜底
        print(f"采集阶段异常终止：{type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 3

    corpus = build_corpus(
        items,
        config=config,
        run_id=run_id,
        week=week,
        reference=reference,
        release_history=release_history,
    )

    log("")
    log(
        f"语料：原始 {corpus.stats['raw']} → 归一化 {corpus.stats['total']} 条"
        f"（本周新增 {corpus.stats['new']}，跨源合并 {corpus.stats['multiSource']}）"
        + (
            f" | 待补中文 {corpus.stats.get('pendingZh', 0)} 条（未进正文）"
            if corpus.stats.get("chineseOnly")
            else ""
        )
    )
    log(f"仓库：{len(corpus.repo_clusters)} 个活跃 | 主题：{len(corpus.themes)} 个")
    failed = [report for report in reports if report.errors]
    log(
        f"信息源：正常 {len([r for r in reports if not r.errors and r.enabled])} / "
        f"降级 {len(failed)} / 禁用 {len([r for r in reports if not r.enabled])}"
    )
    scope_blocked = sum(
        int(report.droppedOutOfScope or 0) + int(report.droppedOffTopic or 0) for report in reports
    )
    if scope_blocked:
        log(
            f"范围拦截：{scope_blocked} 条（不属多模态 infra：非系统工程/降本增效议题，或命中排除项；"
            "规则见 config/sources.json 的 scope 段）"
        )

    if corpus.stats["total"] == 0:
        log("")
        log("!! 正文条目为 0。常见原因：网络不可达、匿名 API 限流、范围闸门过严，")
        log("!! 或者（更常见）本轮新条目还没补中文——报告只出中文条目。")
        log("!! 先看 scripts/report/.cache/pending_zh.json，把中文补进 config/curated_zh.json 后 --offline 重跑。")

    if args.dry_run:
        log("dry-run：不写入任何文件。")
        return 0

    report_html = render_html(
        corpus,
        config=config,
        source_reports=[report.as_dict() for report in reports],
        window=window,
        run_id=run_id,
        generate_command=command,
        top_per_section=top_per_section,
    )
    # 每个周期只产出一个文件：report/<week>.html（直接覆盖当周）
    html_path = report_dir / f"{week}.html"
    html_path.write_text(report_html, encoding="utf-8")

    # 待补中文清单（工作清单，不是报告产物）：报告正文只收已补中文的条目，
    # 未补的条目在这里列出来，补进 config/curated_zh.json 后离线重跑即可进正文。
    pending_path = snapshot_dir.parent / "pending_zh.json"
    dump_json(pending_path, corpus.pending_zh)

    log("")
    log("产出：")
    log(
        f"  · {relative_display(html_path, REPO_ROOT)}"
        f"（{corpus.stats['total']} 条，{len(report_html) // 1024} KB）"
    )
    log(
        f"  · 采集缓存 {relative_display(snapshot_dir, REPO_ROOT)}"
        f"（本轮 {client.stats.requests} 次请求，缓存命中 {client.stats.cache_hits}）"
    )
    log("")
    log(f"离线复现：python scripts/report/run_weekly.py --offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
