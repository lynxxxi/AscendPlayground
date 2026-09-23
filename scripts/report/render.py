#!/usr/bin/env python3
"""L3 渲染层：把语料渲染成自包含 HTML 报告。

产出两份纯 HTML（无外部依赖，可离线打开）：
  report/assets/<week>.html   当周可视化报告
  report/index.html           周报存档页
"""

from __future__ import annotations

import html
from typing import Any, Iterable

from report.corpus import GROUP_LABELS, KIND_LABELS, Corpus
from report.lib.util import parse_datetime, truncate

DIMENSIONS = (
    ("papers", "维度 A · 多模态 Infra 论文", "arXiv / HuggingFace Daily Papers，附中文要点"),
    ("teams", "维度 B · 技术团队与官方动态", "团队博客、模型发布雷达，附内容说明"),
    ("repos", "维度 C · 多模态仓库更新细则", "只跟多模态仓库的 release 与关键 commit"),
    ("wechat", "维度 D · 公众号与中文媒体", "搜狗微信检索（标题直链原文）+ 大V/团队渠道 + 中文媒体 RSS"),
    ("peers", "维度 E · 周边团队工作", "芯片厂商 / 互联网厂商 / 初创公司 / 周边框架的版本特性与本周动态"),
)

NAV = (
    ("overview", "00 · 本期速览"),
    ("papers", "A · 论文"),
    ("teams", "B · 团队动态"),
    ("repos", "C · 仓库更新"),
    ("wechat", "D · 公众号"),
    ("peers", "E · 周边团队"),
    ("appendix", "F · 附录"),
)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _day_of(item: dict[str, Any]) -> str:
    published = parse_datetime(item.get("published"))
    return str(item.get("day") or (published.strftime("%Y-%m-%d") if published else "—"))


def _score(item: dict[str, Any]) -> float:  # 兼容旧调用；系统已取消打分
    return 0.0


def _sort_new_first(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """排序一律「新增优先 → 发布时间倒序」，不打分。"""
    return sorted(
        items,
        key=lambda entry: (1 if entry.get("isNew") else 0, str(entry.get("published") or "")),
        reverse=True,
    )


def _items_by_group(corpus: Corpus, group: str) -> list[dict[str, Any]]:
    return [item for item in corpus.items if str(item.get("group")) == group]


def _badge(item: dict[str, Any]) -> str:
    css = "new" if item.get("isNew") else "old"
    text = "NEW" if item.get("isNew") else "SEEN"
    return f'<span class="badge {css}">{text}</span>'


def _desc(item: dict[str, Any], fallback_limit: int = 200) -> str:
    """条目说明文字：优先用抽取好的 digest，退化到摘要。"""
    text = str(item.get("digest") or "").strip()
    if not text:
        text = truncate(str(item.get("summary") or ""), fallback_limit)
    return _esc(text) if text else '<span class="rp-desc muted">—</span>'


def _html_link(title: str, url: str, limit: int = 130) -> str:
    label = _esc(truncate(title or "", limit))
    if not url:
        return label
    return f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">{label}</a>'


def degraded_note(source_reports: list[dict[str, Any]], source_ids: set[str]) -> str:
    """当指定信息源降级（限流/失败）时，在该维度顶部给出显式提示。"""
    for report in source_reports:
        if str(report.get("id")) not in source_ids:
            continue
        if report.get("errors") or (report.get("enabled", True) and not report.get("kept")):
            reason = "；".join(report.get("errors", [])[:2]) or "本轮未取到结果（可能被限流）"
            return (
                '<div class="rp-warn"><b>该维度本期数据不完整</b>：'
                f'{_esc(str(report.get("label") or report.get("id")))} 未成功采集'
                f'（{_esc(truncate(reason, 200))}）。'
                "等待十几分钟后重跑即可补齐，已有快照不会浪费。</div>"
            )
    return ""


def _search_blob(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title") or ""),
        str(item.get("digest") or ""),
        str(item.get("summary") or "")[:200],
        str(item.get("sourceLabel") or ""),
        str(item.get("kind") or ""),
        str(item.get("group") or ""),
        " ".join(str(term) for term in (item.get("keywords") or [])),
    ]
    return _esc(" ".join(parts).lower())


# --------------------------------------------------------------------------
# 样式与脚本
# --------------------------------------------------------------------------

CSS = """
:root{--ink:#12262d;--night:#0b2027;--paper:#f2efe6;--card:#fffdf7;--line:#c9c3b5;--muted:#617077;--acid:#c9ef45;--coral:#ff6b4a;--cyan:#36c7c1;--shadow:0 20px 55px rgba(11,32,39,.13);--serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;--sans:"Aptos","Trebuchet MS",sans-serif;--mono:"Cascadia Code",Consolas,monospace}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:var(--sans);background-color:var(--paper);background-image:linear-gradient(rgba(11,32,39,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(11,32,39,.045) 1px,transparent 1px);background-size:30px 30px}a{color:#0b7c78;text-underline-offset:3px}
.rp-header{position:relative;overflow:hidden;padding:38px max(24px,calc((100vw - 1240px)/2)) 78px;color:#fff;background:var(--night)}
.rp-header:before{content:"WEEKLY";position:absolute;right:-20px;top:16px;color:rgba(255,255,255,.04);font:900 clamp(5rem,15vw,13rem)/.8 var(--mono);letter-spacing:-.1em}
.rp-top{position:relative;z-index:2;display:flex;justify-content:space-between;gap:18px;align-items:center;flex-wrap:wrap}
.rp-brand{font:800 .74rem/1 var(--mono);letter-spacing:.14em;color:var(--acid)}
.rp-back{padding:8px 11px;border:1px solid rgba(255,255,255,.25);color:#fff;text-decoration:none;font:700 .66rem/1 var(--mono)}
.rp-back:hover{border-color:var(--acid);color:var(--acid)}
.rp-hero{position:relative;z-index:2;margin-top:52px;max-width:960px}
.rp-overline{display:flex;gap:12px;align-items:center;color:var(--cyan);font:800 .68rem/1 var(--mono);letter-spacing:.15em}
.rp-overline:before{content:"";width:48px;height:3px;background:var(--coral)}
.rp-hero h1{margin:16px 0 16px;font:900 clamp(2.4rem,6vw,4.6rem)/.95 var(--serif);letter-spacing:-.05em}
.rp-hero p{margin:0;color:#bed0d5;max-width:820px;font-size:.94rem;line-height:1.7}
.rp-hero .rp-scope{margin-top:10px;padding-left:12px;border-left:3px solid var(--coral);color:#9fb6bc;font-size:.84rem;line-height:1.6}
.rp-meta{margin-top:22px;display:flex;flex-wrap:wrap;gap:8px}
.rp-chip{padding:7px 10px;border:1px solid rgba(255,255,255,.22);color:#cfe0e3;font:700 .63rem/1 var(--mono)}
.rp-stats{position:relative;z-index:3;display:grid;grid-template-columns:repeat(5,1fr);gap:1px;max-width:1240px;margin:-46px auto 0;background:var(--line);box-shadow:var(--shadow)}
.rp-stat{padding:18px 20px;background:var(--card)}
.rp-stat b{display:block;color:var(--night);font:900 1.85rem/1 var(--serif)}
.rp-stat span{display:block;margin-top:7px;color:var(--muted);font:.62rem/1.4 var(--mono)}
.rp-stat.accent{background:var(--coral)}.rp-stat.accent b,.rp-stat.accent span{color:#fff}
.rp-body{display:grid;grid-template-columns:250px minmax(0,1fr);gap:34px;max-width:1240px;margin:auto;padding:56px 24px 90px}
.rp-nav{position:sticky;top:18px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;padding:16px;background:rgba(255,254,250,.8);border-top:4px solid var(--cyan)}
.rp-nav h2{margin:0 0 12px;color:var(--night);font:800 .68rem/1 var(--mono);letter-spacing:.12em}
.rp-nav a{display:block;padding:6px 7px;margin:2px 0;color:var(--muted);font-size:.72rem;line-height:1.4;text-decoration:none;border-left:2px solid transparent}
.rp-nav a:hover,.rp-nav a.active{color:var(--night);border-color:var(--coral);background:rgba(255,107,74,.07)}
.rp-search{width:100%;height:36px;margin-bottom:12px;padding:0 10px;border:1px solid var(--line);background:var(--card);font:700 .72rem var(--mono);outline:0}
.rp-search:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(54,199,193,.16)}
.rp-hint{color:var(--muted);font:.6rem/1.5 var(--mono)}
.rp-main{min-width:0}
.rp-section{margin-bottom:56px;scroll-margin-top:18px}
.rp-section>header{border-bottom:3px double var(--night);padding-bottom:12px;margin-bottom:20px}
.rp-kicker{display:block;color:var(--coral);font:800 .64rem/1 var(--mono);letter-spacing:.14em;text-transform:uppercase}
.rp-section h2{margin:9px 0 8px;font:900 clamp(1.7rem,3.2vw,2.6rem)/1 var(--serif);letter-spacing:-.035em;color:var(--night)}
.rp-section header p{margin:0;color:var(--muted);font-size:.8rem;line-height:1.6}
.rp-sub{margin:26px 0 10px;font:800 1.05rem/1.3 var(--serif);color:var(--night)}
.rp-add{color:var(--coral);font-weight:800}
table{width:100%;border-collapse:collapse;font-size:.78rem;background:var(--card);box-shadow:0 6px 20px rgba(11,32,39,.05)}
th{padding:9px 10px;color:#fff;background:var(--night);text-align:left;font:700 .68rem/1.3 var(--mono);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top;line-height:1.55}
tr:hover td{background:#fdfbf3}
td.mono,th.mono{font-family:var(--mono);white-space:nowrap}
.num{text-align:right;font-family:var(--mono);white-space:nowrap}
.badge{display:inline-block;min-width:20px;padding:2px 6px;border-radius:99px;font:800 .6rem/1.4 var(--mono);text-align:center}
.badge.new{color:#fff;background:var(--coral)}
.badge.old{color:#5d6c72;background:#e6e2d7}
.badge.focus{color:var(--night);background:var(--acid)}
.rp-note{padding:12px 15px;margin:14px 0;color:#36545b;background:#e5f4f0;border-left:5px solid var(--cyan);font-size:.8rem;line-height:1.6}
.rp-warn{padding:12px 15px;margin:14px 0;color:#774131;background:#fff1eb;border-left:5px solid var(--coral);font-size:.8rem;line-height:1.6}
.rp-empty{padding:26px;text-align:center;color:var(--muted);border:1px dashed var(--line);font-size:.8rem}
.rp-card{padding:16px;margin-bottom:12px;background:var(--card);border:1px solid var(--line);border-left:5px solid var(--cyan)}
.rp-card h3{margin:0 0 8px;font:800 .95rem/1.3 var(--serif);color:var(--night)}
.rp-card ul{margin:6px 0 0;padding-left:18px}
.rp-card li{margin:5px 0;font-size:.78rem;line-height:1.55}
.rp-bar{display:inline-block;height:9px;background:var(--cyan);vertical-align:middle;border-radius:2px}
.rp-bar.ascend{background:var(--acid)}
.bar-cell{min-width:120px}
.rp-desc{color:#41565d;font-size:.8rem;line-height:1.62;min-width:260px}
.rp-desc.muted{color:var(--muted)}
.rp-hl{display:grid;gap:12px}
.rp-rel{padding:14px 16px;background:var(--card);border:1px solid var(--line);border-left:5px solid var(--cyan)}
.rp-rel.focus{border-left-color:var(--acid)}
.rp-rel-top{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-bottom:8px}
.rp-rel-tag{padding:4px 9px;color:var(--night);background:var(--acid);font:800 .7rem/1 var(--mono)}
.rp-rel-day{color:var(--muted);font:.65rem/1 var(--mono)}
.rp-rel p{margin:0;color:#31474e;font-size:.8rem;line-height:1.68}
.rp-rel a{font:.65rem/1 var(--mono)}
.rp-eng{margin:0 0 10px;font:800 1rem/1.3 var(--serif);color:var(--night)}
.rp-eng .rp-stack{margin-left:8px;color:var(--muted);font:700 .62rem/1 var(--mono)}
.rp-block{margin-bottom:24px}
footer{padding:26px 24px 42px;color:#aabfc4;background:var(--night);text-align:center;font:.66rem/1.6 var(--mono)}footer a{color:var(--acid)}footer code{color:var(--acid);font-family:var(--mono)}
.rp-legend{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-bottom:16px}.rp-legend span{padding:6px 10px;border:1px solid rgba(255,255,255,.18);color:#cfe0e3;font:.62rem/1.3 var(--mono)}.rp-legend b{color:var(--acid)}
mark{background:var(--acid);color:var(--night);padding:0 2px}
@media(max-width:980px){.rp-body{grid-template-columns:1fr}.rp-nav{position:relative;top:0}.rp-stats{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.rp-stats{grid-template-columns:1fr}.rp-hero h1{font-size:2.2rem}table{font-size:.72rem}th,td{padding:7px 6px}}
"""

SCRIPT = """
(function(){
  var search=document.getElementById('rpSearch');
  var sections=[].slice.call(document.querySelectorAll('.rp-section'));
  var navLinks=[].slice.call(document.querySelectorAll('.rp-nav a[href^="#"]'));
  function applyFilter(){
    var q=(search&&search.value||'').trim().toLowerCase();
    var visible=0;
    sections.forEach(function(section){
      var shown=0;
      [].slice.call(section.querySelectorAll('tbody tr')).forEach(function(row){
        var blob=row.getAttribute('data-search')||'';
        var hit=!q||blob.indexOf(q)>=0;
        row.hidden=!hit;
        if(hit)shown++;
      });
      var cards=[].slice.call(section.querySelectorAll('.rp-card'));
      if(q&&cards.length){
        shown=0;
        cards.forEach(function(card){
          var blob=card.getAttribute('data-search')||'';
          var hit=blob.indexOf(q)>=0;
          card.hidden=!hit;
          if(hit)shown++;
        });
      }else{cards.forEach(function(card){card.hidden=false})}
      var empty=section.querySelector('.rp-empty');
      if(empty)empty.hidden=!(q&&shown===0);
      section.hidden=!!(q&&shown===0&&section.getAttribute('data-filterable')==='1');
      if(!section.hidden)visible++;
    });
    var counter=document.getElementById('rpCount');
    if(counter)counter.textContent=visible;
  }
  if(search){search.addEventListener('input',applyFilter);applyFilter()}
  navLinks.forEach(function(link){
    link.addEventListener('click',function(event){
      var target=document.querySelector(link.getAttribute('href'));
      if(!target)return;
      event.preventDefault();
      target.scrollIntoView({behavior:'smooth',block:'start'});
    });
  });
  if('IntersectionObserver' in window){
    var observer=new IntersectionObserver(function(records){
      records.forEach(function(record){
        if(!record.isIntersecting)return;
        navLinks.forEach(function(link){
          link.classList.toggle('active',link.getAttribute('href')==='#'+record.target.id);
        });
      });
    },{rootMargin:'-20px 0px -75%'});
    sections.forEach(function(section){observer.observe(section)});
  }
})();
"""


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def render_html(
    corpus: Corpus,
    *,
    config: dict[str, Any],
    source_reports: list[dict[str, Any]],
    window: dict[str, Any],
    run_id: str,
    generate_command: str,
    top_per_section: int = 25,
) -> str:
    stats = corpus.stats

    def table(columns: list[str], rows: list[str], empty: str = "本期无条目") -> str:
        if not rows:
            return f'<div class="rp-empty">{_esc(empty)}</div>'
        head = "".join(f"<th>{column}</th>" for column in columns)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    def row(cells: list[str], item: dict[str, Any]) -> str:
        return f'<tr data-search="{_search_blob(item) if item else ""}">' + "".join(
            f"<td>{cell}</td>" for cell in cells
        ) + "</tr>"

    sections: list[str] = []

    # ---- 速览（无打分、无高分条目、无按检索词分布） ----
    theme_rows = [
        row(
            [
                f'<span class="mono">{_esc(theme["theme"])}</span>',
                f'<span class="num">{theme["count"]}</span>',
                f'<span class="num rp-add">{theme["newCount"]}</span>',
                _html_link(
                    ((theme.get("top") or [{}])[0]).get("title", ""),
                    ((theme.get("top") or [{}])[0]).get("url", ""),
                    90,
                ),
            ],
            {"title": theme["theme"]},
        )
        for theme in corpus.themes[:16]
    ]
    dimension_rows = [
        row(
            [
                _esc(label),
                f'<span class="num">{_dimension_total(corpus, group)}</span>',
                f'<span class="num rp-add">{_dimension_new(corpus, group)}</span>',
                _esc(note),
            ],
            {"title": label},
        )
        for group, label, note in DIMENSIONS
    ]
    sections.append(
        _section(
            "overview",
            "00",
            "本期速览",
            "维度分布与主题关键词热度。搜索框可跨全部维度过滤。",
            '<h3 class="rp-sub">0.1 维度分布</h3>'
            + table(["维度", "本期条目", "本周新增", "覆盖范围"], dimension_rows)
            + '<h3 class="rp-sub">0.2 主题热度（关键词聚类）</h3>'
            + table(["主题", "命中", "新增", "代表条目"], theme_rows),
        )
    )

    # ---- A 论文 ----
    papers = _sort_new_first(_items_by_group(corpus, "papers"))
    arxiv_queries = len(((config.get("papers") or {}).get("arxiv") or {}).get("queries") or [])
    paper_rows = [
        row(
            [
                _badge(item),
                f'<span class="mono">{_esc(_day_of(item))}</span>',
                _html_link(item.get("title", ""), item.get("url", "")),
                f'<div class="rp-desc">{_desc(item, 300)}</div>',
                f'<span class="mono">{_esc(item.get("sourceLabel",""))}</span>',
            ],
            item,
        )
        for item in papers[: top_per_section * 2]
    ]
    sections.append(
        _section(
            "papers",
            "A",
            "多模态 Infra 论文",
            f"arXiv 组合检索（{arxiv_queries} 组关键词）+ HuggingFace Daily Papers 交叉验证；"
            f"采集窗口 {window.get('maxAgeDays')} 天，本期命中 {len(papers)} 条。每条附中文要点。",
            table(["新", "日期", "标题", "内容说明", "来源"], paper_rows),
        )
    )

    # ---- B 团队 ----
    teams = _sort_new_first(_items_by_group(corpus, "teams"))
    model_rows = [
        row(
            [
                _badge(item),
                f'<span class="mono">{_esc(_day_of(item))}</span>',
                _html_link(item.get("title", ""), item.get("url", "")),
                f'<div class="rp-desc">{_desc(item, 200)}</div>',
                f'<span class="mono">{_esc((item.get("signals") or {}).get("org",""))}</span>',
                f'<span class="num">♥{_esc((item.get("signals") or {}).get("likes",0))}</span>',
            ],
            item,
        )
        for item in teams
        if item.get("kind") == "model-release"
    ][: top_per_section * 2]
    blog_rows = [
        row(
            [
                _badge(item),
                f'<span class="mono">{_esc(_day_of(item))}</span>',
                _html_link(item.get("title", ""), item.get("url", "")),
                f'<div class="rp-desc">{_desc(item, 280)}</div>',
                f'<span class="mono">{_esc(item.get("sourceLabel",""))}</span>',
            ],
            item,
        )
        for item in teams
        if item.get("kind") != "model-release"
    ][: top_per_section * 2]
    sections.append(
        _section(
            "teams",
            "B",
            "技术团队与官方动态",
            "团队官方博客 / 模型发布雷达；每条附内容说明，无需点开即可判断是否相关。",
            '<h3 class="rp-sub">B.1 新模型 / 新权重发布</h3>'
            + table(["新", "日期", "模型", "内容说明", "组织", "点赞"], model_rows)
            + '<h3 class="rp-sub">B.2 团队博客与行业资讯</h3>'
            + table(["新", "日期", "标题", "内容说明", "来源"], blog_rows),
        )
    )

    # ---- C 仓库 ----
    clusters = corpus.repo_clusters
    totals = [cluster.get("total", 0) for cluster in clusters]
    max_total = max(totals) if totals and max(totals) > 0 else 1
    repo_rows = [
        row(
            [
                f'<a href="https://github.com/{_esc(cluster["repo"])}" target="_blank" rel="noopener">{_esc(cluster["repo"])}</a>',
                f'<span class="mono">{_esc(cluster.get("group",""))}</span>',
                f'<span class="num">{cluster.get("counts",{}).get("release",0)}</span>',
                f'<span class="num">{cluster.get("counts",{}).get("commit",0)}</span>',
                f'<span class="bar-cell"><span class="rp-bar" style="width:{(cluster.get("total",0)/max_total*100):.0f}%"></span> {cluster.get("total",0)}</span>',
                f'<span class="mono">{_esc(cluster.get("latestIso",""))}</span>',
                _esc(cluster.get("note", "")),
            ],
            {"title": cluster["repo"], "summary": cluster.get("note", "")},
        )
        for cluster in clusters
    ]
    detail_cards = []
    for cluster in clusters[:16]:
        items_html = []
        for item in sorted(cluster.get("items", []), key=lambda entry: str(entry.get("published") or ""), reverse=True)[:5]:
            kind_label = KIND_LABELS.get(str(item.get("kind")), item.get("kind"))
            note = str(item.get("digest") or item.get("summary") or "")
            items_html.append(
                f'<li>{_badge(item)} <span class="mono">{_esc(_day_of(item))}</span> '
                f'<span class="mono">[{_esc(kind_label)}]</span> '
                f'{_html_link(item.get("title",""), item.get("url",""), 130)}'
                + (f'<div class="rp-desc">{_esc(truncate(note, 300))}</div>' if note else "")
                + "</li>"
            )
        blob = _esc(
            (
                cluster["repo"]
                + " "
                + " ".join(
                    f"{entry.get('title','')} {entry.get('digest','')}" for entry in cluster.get("items", [])
                )
            ).lower()
        )
        detail_cards.append(
            f'<div class="rp-card" data-search="{blob}"><h3>{_esc(cluster["repo"])} · {_esc(cluster.get("note",""))}</h3>'
            f'<ul>{"".join(items_html)}</ul></div>'
        )
    sections.append(
        _section(
            "repos",
            "C",
            "多模态仓库更新细则",
            f"监控 {len(((config.get('repos') or {}).get('watch') or []))} 个多模态仓库的 release 与关键 commit；"
            f"本期命中 {len(clusters)} 个活跃仓库，每条附变更说明。"
            "（通用 LLM 主仓只取其多模态子系统目录；训练框架与通用算子仓不在监控范围。）",
            table(["仓库", "分区", "发布", "提交", "信号量", "最近动态", "说明"], repo_rows)
            + '<h3 class="rp-sub">C.1 重点仓库变更明细</h3>'
            + ("".join(detail_cards) or '<div class="rp-empty">本期无仓库明细</div>'),
        )
    )

    # ---- D 公众号 ----
    wechat = _sort_new_first(_items_by_group(corpus, "wechat"))
    resolved = sum(1 for item in wechat if (item.get("signals") or {}).get("linkResolved"))

    def _channel(item: dict[str, Any]) -> str:
        signals = item.get("signals") or {}
        kol = str(signals.get("kol") or "")
        if kol:
            category = str(signals.get("kolCategory") or "")
            return f'{_esc(kol)}<span class="rp-desc muted">（{_esc(category or "渠道")}）</span>'
        account = str(signals.get("account") or "")
        return _esc(account) if account else "—"

    wechat_rows = [
        row(
            [
                _badge(item),
                f'<span class="mono">{_esc(_day_of(item))}</span>',
                _html_link(item.get("title", ""), item.get("url", "")),
                f'<div class="rp-desc">{_desc(item, 280)}</div>',
                _channel(item),
            ],
            item,
        )
        for item in wechat[: top_per_section * 2]
    ]
    kol_count = len(((config.get("wechat") or {}).get("kolChannels") or []))
    sections.append(
        _section(
            "wechat",
            "D",
            "公众号与中文媒体",
            f"搜狗微信检索 {len((config.get('wechat') or {}).get('searches') or [])} 组关键词"
            f" + {kol_count} 个明星大V / 关键员工 / 团队渠道 + 中文技术媒体 RSS。"
            f"标题即原文直链（本期解析成功 {resolved}/{len(wechat)} 条），可直接点开阅读。",
            degraded_note(source_reports, {"sogou-wechat"})
            + table(["新", "日期", "标题（点击进入原文）", "内容说明", "公众号 / 渠道"], wechat_rows),
        )
    )

    # ---- E 周边团队 ----
    peer_rows = corpus.peer_matrix.get("rows") or []
    peer_blocks = []
    for entry in peer_rows:
        releases = entry.get("recentReleases") or []
        if not releases:
            continue
        cards = []
        for release in releases:
            feature = str(release.get("digest") or "").strip()
            cards.append(
                '<div class="rp-rel">'
                '<div class="rp-rel-top"><span class="rp-rel-tag">{0}</span>'
                '<span class="rp-rel-day">{1}</span>'
                '<a href="{2}" target="_blank" rel="noopener">releases ↗</a></div>'
                "<p>{3}</p></div>".format(
                    _esc(release.get("tag", "")),
                    _esc(release.get("day", "")),
                    _esc(release.get("url", "")),
                    _esc(feature) if feature else '<span class="rp-desc muted">该版本未提供说明文本</span>',
                )
            )
        peer_blocks.append(
            f'<div class="rp-block" data-search="{_esc((str(entry.get("label","")) + " " + str(entry.get("org","")) + " " + str(entry.get("category",""))).lower())}">'
            f'<h3 class="rp-eng">{_esc(entry.get("label",""))}'
            f'<span class="rp-stack">{_esc(entry.get("category",""))} · {_esc(entry.get("org",""))} · {_esc(entry.get("repo",""))}</span></h3>'
            f'<div class="rp-hl">{"".join(cards)}</div></div>'
        )
    radar_rows = [
        row(
            [
                _esc(team.get("label", "")),
                _esc(team.get("category", "")),
                f'<span class="num">{team.get("matches",0)}</span>',
                f'<span class="num rp-add">{team.get("newMatches",0)}</span>',
                "<br>".join(
                    _html_link(evidence.get("title", ""), evidence.get("url", ""), 90)
                    for evidence in (team.get("evidence") or [])[:2]
                )
                or '<span class="rp-desc muted">本期无信号</span>',
            ],
            {"title": team.get("label", "")},
        )
        for team in corpus.peer_matrix.get("radar", [])
        if team.get("matches")
    ]
    sections.append(
        _section(
            "peers",
            "E",
            "周边团队工作",
            "芯片厂商 / 互联网厂商 / 有影响力的初创公司 / 周边推理框架："
            "E.1 列各团队多模态相关版本的官方特性说明；E.2 用团队名录反查本周语料，看谁有动静。"
            "（本维度与维度 C 的仓库更新分开呈现。）",
            '<h3 class="rp-sub">E.1 周边引擎与团队版本特性</h3>'
            + ("".join(peer_blocks) or '<div class="rp-empty">本期无周边团队发版数据</div>')
            + '<h3 class="rp-sub">E.2 团队动态雷达（芯片 / 互联网 / 初创）</h3>'
            + table(["团队", "类别", "本期命中", "本周新增", "代表条目"], radar_rows),
        )
    )


    # ---- 附录 ----
    source_rows = [
        row(
            [
                _esc(report.get("label", "")),
                f'<span class="mono">{_esc(report.get("kind",""))}</span>',
                f'<span class="num">{report.get("fetched",0)}</span>',
                f'<span class="num">{report.get("kept",0)}</span>',
                f'<span class="num">{int(report.get("droppedOutOfScope",0) or 0) + int(report.get("droppedOffTopic",0) or 0)}</span>',
                (
                    '<span class="badge old">禁用</span>'
                    if not report.get("enabled", True)
                    else (
                        '<span class="badge old">正常</span>'
                        if not report.get("errors")
                        else f'<span class="badge new">降级</span> {_esc("；".join(report.get("errors", [])[:2]))}'
                    )
                ),
            ],
            {"title": report.get("label", "")},
        )
        for report in source_reports
    ]
    scope = config.get("scope") or {}
    scope_statement = str(scope.get("statement") or "")
    scope_line = f"<b>收录范围：</b>{_esc(scope_statement)}<br>" if scope_statement else ""
    scope_blocked = sum(
        int(report.get("droppedOutOfScope", 0) or 0) + int(report.get("droppedOffTopic", 0) or 0)
        for report in source_reports
    )
    pending_zh = int(stats.get("pendingZh", 0) or 0)
    zh_line = (
        "<b>中文说明：</b>报告只收录已补中文的条目（<code>report.chineseOnly=true</code>）；"
        f"本轮另有 <b>{pending_zh}</b> 条因尚未补中文而未进正文，清单见 "
        "<code>scripts/report/.cache/pending_zh.json</code>，补进 <code>config/curated_zh.json</code> "
        "后离线重跑即可。<br>"
        if stats.get("chineseOnly")
        else ""
    )
    appendix = (
        table(["信息源", "类型", "抓取", "保留", "范围拦截", "状态"], source_rows)
        + f'<div class="rp-note">{scope_line}'
        f"<b>范围拦截：</b>本轮因不属多模态 infra（非系统工程/降本增效议题，或命中排除项）丢弃 {scope_blocked} 条；"
        f"范围规则见 <code>scripts/report/config/sources.json</code> 的 <code>scope</code> 段。<br>"
        + zh_line
        + f'<b>Run ID：</b><code>{_esc(run_id)}</code> · '
        f'<b>采集窗口：</b>{_esc(window.get("since",""))} ~ {_esc(window.get("until",""))}'
        f'（{_esc(window.get("maxAgeDays"))} 天）<br>'
        f"<b>复现命令：</b><code>{_esc(generate_command)}</code>；离线复现加 <code>--offline</code>（只读 <code>report/snapshots/</code>）。<br>"
        f"<b>标记：</b>NEW = 发布时间落在最近 {_esc(corpus.freshness_days)} 天内（本周新增）；"
        f"SEEN = 更早条目。同一份快照重跑，结果完全一致。<br>"
        "<b>排序：</b>本报告不打分，所有条目按发布时间排序。</div>"
    )
    sections.append(_section("appendix", "F", "附录 · 采集状态与复现", "每个信息源的抓取量、保留量与降级原因。", appendix))

    nav = "".join(f'<a href="#{section_id}">{label}</a>' for section_id, label in NAV)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="AscendPlayground 多模态 Infra 周报 {_esc(corpus.week)}：论文、团队动态、多模态仓库更新、公众号与周边团队工作。">
<meta name="theme-color" content="#0b2027">
<title>多模态 Infra 周报 · {_esc(corpus.week)} · AscendPlayground</title>
<style>{CSS}</style>
</head>
<body>
<header class="rp-header">
  <div class="rp-top"><div class="rp-brand">ASCEND PLAYGROUND / MULTIMODAL INFRA WEEKLY</div><a class="rp-back" href="https://github.com/lynxxxi/AscendPlayground" target="_blank" rel="noopener">ASCENDPLAYGROUND ↗</a></div>
  <div class="rp-hero">
    <div class="rp-overline">WEEKLY INTELLIGENCE REPORT</div>
    <h1>多模态 Infra 周报<br>{_esc(corpus.week)}</h1>
    <p>覆盖多模态 infra 论文、技术团队动态、多模态仓库更新、公众号与中文媒体渠道，以及芯片厂商 / 互联网厂商 / 初创公司等周边团队的工程节奏。</p>
    <p class="rp-scope">收录边界：论文与公众号维度只收「推理与服务的系统工程」（引擎 / 算子 / 并行 / 缓存与显存 / 量化压缩 / 硬件适配 / 部署评测）；模型能力、算法创新、应用落地、仿脑与具身机器人等非 infra 议题一律不收。仓库维度只看多模态仓库；报告不打分，全部按发布时间排序，且只出中文。</p>
    <div class="rp-meta">
      <span class="rp-chip">生成 {_esc(corpus.generated_at)}</span>
      <span class="rp-chip">窗口 {_esc(window.get("since",""))} → {_esc(window.get("until",""))}</span>
      <span class="rp-chip">Run {_esc(run_id)}</span>
      <span class="rp-chip">复现 {_esc(generate_command)}</span>
    </div>
  </div>
</header>
<div class="rp-stats">
  <div class="rp-stat"><b>{stats.get("total",0)}</b><span>归一化条目</span></div>
  <div class="rp-stat accent"><b>{stats.get("new",0)}</b><span>本周新增</span></div>
  <div class="rp-stat"><b>{stats.get("raw",0)}</b><span>原始抓取</span></div>
  <div class="rp-stat"><b>{len(corpus.repo_clusters)}</b><span>活跃仓库</span></div>
  <div class="rp-stat"><b>{stats.get("multiSource",0)}</b><span>跨源去重合并</span></div>
</div>
<div class="rp-body">
  <aside class="rp-nav">
    <h2>DIMENSIONS</h2>
    <input class="rp-search" id="rpSearch" type="search" placeholder="搜索标题 / 关键词…" autocomplete="off">
    <div class="rp-hint">显示 <b id="rpCount">{len(NAV)}</b> / {len(NAV)} 个维度<br>输入关键词可跨维度过滤</div>
    <nav>{nav}</nav>
  </aside>
  <main class="rp-main">
    {"".join(sections)}
  </main>
</div>
<footer>
  <div class="rp-legend">
    <span><b>NEW</b> 发布时间在最近 {_esc(corpus.freshness_days)} 天内</span>
    <span><b>SEEN</b> 更早、仍在采集窗口内</span>
    <span>不打分、按发布时间排序</span>
    <span>顶部搜索框可跨全部维度过滤</span>
  </div>
  ASCENDPLAYGROUND · MULTIMODAL INFRA WEEKLY · {_esc(corpus.week)} · 生成于 {_esc(corpus.generated_at)}<br>
  离线复现：<code>python scripts/report/run_weekly.py --offline</code> · 自检：<code>python scripts/report/selftest.py</code><br>
  <a href="https://github.com/lynxxxi/AscendPlayground" target="_blank" rel="noopener">SOURCE ON GITHUB</a>
</footer>
<script>{SCRIPT}</script>
</body>
</html>
"""


def _dimension_total(corpus: Corpus, group: str) -> int:
    return int(corpus.stats.get("byGroup", {}).get(group, 0))


def _dimension_new(corpus: Corpus, group: str) -> int:
    return int(corpus.stats.get("newByGroup", {}).get(group, 0))


def _section(
    section_id: str,
    index: str,
    title: str,
    description: str,
    body: str,
    *,
    filterable: bool = True,
) -> str:
    return (
        f'<section class="rp-section" id="{section_id}" data-filterable="{1 if filterable else 0}">'
        f'<header><span class="rp-kicker">{_esc(index)} / DIMENSION</span>'
        f"<h2>{_esc(title)}</h2><p>{description}</p></header>"
        f"{body}"
        f'<div class="rp-empty" hidden>该维度在当前关键词下没有匹配条目</div>'
        "</section>"
    )
