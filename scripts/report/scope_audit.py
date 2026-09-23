#!/usr/bin/env python3
"""调研范围审计：看清「scope 闸门」到底拦了什么，用来调边界。

用法：
    python scripts/report/scope_audit.py                     # 联网采集后审计
    python scripts/report/scope_audit.py --offline           # 只用快照离线审计（推荐，结果可复现）
    python scripts/report/scope_audit.py --only arxiv        # 只看某个源
    python scripts/report/scope_audit.py --group papers      # 只看某个维度

做法：先用**跳过范围闸门**的配置采集一遍（即旧规则的全量结果），再用当前
`config/sources.json` 的 `scope` 规则逐条判定，于是「保留 / 排除法拦 / 标题缺 infra 证据拦」
三类都能列出来，并给出命中词——调 `scope.exclude` 与 `scope.infraEvidence` 时看这个输出即可。
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from report.collect import collect_all, load_config  # noqa: E402
from report.lib.httpclient import DEFAULT_USER_AGENT, HttpClient, SnapshotCache  # noqa: E402
from report.lib.relevance import Scorer  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "config" / "sources.json"
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计调研范围闸门（scope）的拦截结果")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true", help="只读快照，不联网")
    parser.add_argument("--only", default="", help="只审计指定 source id（逗号分隔）")
    parser.add_argument("--group", default="", help="只列出指定维度（papers / repos / teams / wechat / competitors）")
    parser.add_argument("--limit", type=int, default=0, help="每类最多列出多少条（0 = 不限制）")
    return parser.parse_args(argv)


def _classify(scorer: Scorer, items: list[dict[str, Any]]) -> tuple[list[dict], list[tuple[str, dict]], list[dict]]:
    kept: list[dict] = []
    blocked_exclude: list[tuple[str, dict]] = []
    blocked_infra: list[dict] = []
    for item in items:
        title = str(item.get("title") or "")
        text = f"{title} {item.get('summary') or ''}"
        require_infra = scorer.requires_infra_evidence(str(item.get("group") or ""))
        if scorer.should_check_exclude(str(item.get("group") or "")):
            reason = scorer.out_of_scope_reason(text)
            if reason:
                blocked_exclude.append((reason, item))
                continue
        if not scorer.passes_gate(text, require_infra=require_infra, title=title):
            blocked_infra.append(item)
            continue
        kept.append(item)
    return kept, blocked_exclude, blocked_infra


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quiet = lambda message: None  # noqa: E731

    config = load_config(args.config.resolve(), logger=quiet)
    baseline = copy.deepcopy(config)
    baseline.pop("scope", None)  # 旧规则：不做范围约束，得到全量候选

    client = HttpClient(
        SnapshotCache((args.cache_dir.resolve() / "snapshots")),
        offline=args.offline,
        logger=quiet,
        user_agent=str((config.get("defaults") or {}).get("userAgent") or "") or DEFAULT_USER_AGENT,
    )
    only = [token.strip() for token in args.only.split(",") if token.strip()]
    items, reports, _ = collect_all(baseline, client, logger=quiet, only=only or None)

    scorer = Scorer(config)
    kept, blocked_exclude, blocked_infra = _classify(scorer, items)

    def show(rows: list[dict], mapper) -> None:
        limited = rows if args.limit <= 0 else rows[: args.limit]
        for row in limited:
            print(f"  {mapper(row)}")
        if args.limit > 0 and len(rows) > args.limit:
            print(f"  … 其余 {len(rows) - args.limit} 条省略")

    group_filter = args.group.strip()
    print("=" * 72)
    print(f"范围审计 · {'offline（快照）' if args.offline else 'online（联网）'}")
    print(f"候选（跳过 scope 的全量）：{len(items)}")
    print(f"保留：{len(kept)} / 排除法拦：{len(blocked_exclude)} / 标题缺 infra 证据拦：{len(blocked_infra)}")
    print("=" * 72)

    print("\n【排除法拦截 · 命中词 → 标题】")
    rows = [(reason, item) for reason, item in blocked_exclude if not group_filter or item.get("group") == group_filter]
    show(rows, lambda row: f"[{row[0]}] {row[1].get('title')}")

    print("\n【标题缺 infra 证据拦截】")
    rows2 = [item for item in blocked_infra if not group_filter or item.get("group") == group_filter]
    show(rows2, lambda item: f"{item.get('title')}")

    print("\n【保留】")
    rows3 = [item for item in kept if not group_filter or item.get("group") == group_filter]
    show(rows3, lambda item: f"{item.get('title')}")

    print("\n【分源统计（按 scope 规则判定）】")
    ids = sorted({str(item.get("sourceId") or "?") for item in items})
    for source_id in ids:
        rows = [item for item in items if str(item.get("sourceId") or "?") == source_id]
        keep_ids = {id(item) for item in kept}
        excl_ids = {id(item) for _, item in blocked_exclude}
        n_keep = sum(1 for item in rows if id(item) in keep_ids)
        n_excl = sum(1 for item in rows if id(item) in excl_ids)
        print(
            f"  {source_id:<22} 候选 {len(rows):<4} 保留 {n_keep:<4} "
            f"排除法拦 {n_excl:<4} 缺 infra 证据拦 {len(rows) - n_keep - n_excl}"
        )
    degraded = [report for report in reports if report.errors and report.enabled]
    for report in degraded:
        print(f"  ! {report.id} 采集降级：{'；'.join(report.errors[:2])}")
    print("\n提示：想放宽边界，改 config/sources.json 的 scope（删词或删分组名）后重跑本脚本对比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
