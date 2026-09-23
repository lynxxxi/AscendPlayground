#!/usr/bin/env python3
"""离线自检：不联网验证解析、打分、去重、渲染全链路。

    python scripts/report/selftest.py

任一断言失败即返回非 0 退出码，可直接用于 CI。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from report.collect import (  # noqa: E402
    build_sources,
    is_interesting_commit,
    parse_sogou_wechat,
    repo_index,
)
from report.corpus import build_corpus  # noqa: E402
from report.lib.feed import parse_feed  # noqa: E402
from report.lib.relevance import ScopeGate, TagExtractor, dedupe  # noqa: E402
from report.lib.util import (  # noqa: E402
    clean_title,
    iso,
    now_utc,
    parse_datetime,
    slugify,
    strip_html,
    title_key,
    week_key,
)
from report.render import render_html  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config" / "sources.json"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILED.append(f"{name} {detail}".strip())
        print(f"  FAIL {name} {detail}")


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>示例技术博客</title>
  <item>
    <title>多模态推理引擎的 KV Cache 优化实践</title>
    <link>https://example.com/post/1</link>
    <description><![CDATA[<p>介绍 <b>multimodal</b> serving 的 KV cache 复用与吞吐优化。</p>]]></description>
    <pubDate>Tue, 15 Sep 2026 08:30:00 +0800</pubDate>
    <author>infra-team</author>
    <category>推理加速</category>
  </item>
  <item>
    <title>视频生成模型的稀疏注意力算子</title>
    <link>https://example.com/post/2</link>
    <description>video generation 的 sparse attention kernel 实现。</description>
    <pubDate>Mon, 14 Sep 2026 10:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""

SAMPLE_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Query: all:multimodal AND all:serving</title>
  <entry>
    <id>http://arxiv.org/abs/2609.01234v1</id>
    <updated>2026-09-15T01:02:03Z</updated>
    <published>2026-09-15T01:02:03Z</published>
    <title>Efficient Multimodal Serving with Encoder Disaggregation</title>
    <summary>We present a multimodal inference engine that disaggregates the vision encoder.</summary>
    <author><name>Alice Zhang</name></author>
    <link href="http://arxiv.org/abs/2609.01234v1" rel="alternate" type="text/html"/>
    <category term="cs.DC"/>
  </entry>
</feed>
"""


def sample_item(**overrides):
    now = now_utc()
    item = {
        "title": "Multimodal Serving Optimization",
        "summary": "multimodal serving with kv cache reuse and quantization for inference throughput",
        "url": "https://example.com/a",
        "published": iso(now - timedelta(days=1)),
        "nativeId": "sample-1",
        "kind": "paper",
        "group": "papers",
        "sourceId": "arxiv",
        "sourceLabel": "arXiv",
        "weight": 1.35,
        "stableId": "arxiv:sample-1",
        "signals": {},
        "zhCurated": True,  # 报告只出中文条目，测试样本默认视为已补中文
    }
    item.update(overrides)
    return item


# --------------------------------------------------------------------------


def test_util() -> None:
    print("\n[1/6] 工具函数")
    check("parse_datetime/RFC822", parse_datetime("Tue, 15 Sep 2026 08:30:00 +0800") is not None)
    check("parse_datetime/ISO-Z", parse_datetime("2026-09-15T01:02:03Z") is not None)
    check("parse_datetime/日期", parse_datetime("2026-09-15") is not None)
    check("parse_datetime/非法输入", parse_datetime("not-a-date") is None)
    check("parse_datetime/None", parse_datetime(None) is None)
    check("strip_html", strip_html("<p>hello <b>world</b></p>") == "hello world")
    check("clean_title 去站点后缀", clean_title("某标题 - 机器之心") == "某标题")
    check("title_key 一致性", title_key("Hello, World!") == title_key("hello world"))
    check("title_key 差异性", title_key("A") != title_key("B"))
    check("slugify 中文保底", slugify("多模态 推理!!") != "")
    check("week_key 格式", len(week_key()) == 8 and "-W" in week_key(), week_key())


def test_feed() -> None:
    print("\n[2/6] RSS / Atom 解析")
    rss = parse_feed(SAMPLE_RSS)
    check("RSS 条目数", len(rss.entries) == 2, str(len(rss.entries)))
    check("RSS 标题", "KV Cache" in rss.entries[0].title if rss.entries else False)
    check("RSS 链接", rss.entries[0].link == "https://example.com/post/1")
    check("RSS 摘要去标签", "<b>" not in rss.entries[0].summary)
    check("RSS 时间解析", parse_datetime(rss.entries[0].published) is not None)
    check("RSS 分类", "推理加速" in rss.entries[0].tags)

    atom = parse_feed(SAMPLE_ATOM)
    check("Atom 条目数", len(atom.entries) == 1)
    check("Atom 链接", atom.entries[0].link == "http://arxiv.org/abs/2609.01234v1")
    check("Atom 作者", "Alice" in atom.entries[0].author)
    check("Atom 时间", parse_datetime(atom.entries[0].published) is not None)

    check("损坏 XML 不抛异常", len(parse_feed("<rss><channel><item>").entries) == 0)
    check("空输入不抛异常", len(parse_feed("").entries) == 0)


def test_relevance(config: dict) -> None:
    print("\n[3/6] 关键词抽取与入库闸门")
    extractor = TagExtractor(config.get("keywords") or {})
    hits = extractor.extract("multimodal serving with kv cache compression and quantization vlm inference")
    check("关键词命中", len(hits) >= 3, str(hits[:4]))
    check("英文大小写无关", bool(extractor.extract("MULTIMODAL SERVING")))
    check("中文关键词", bool(extractor.extract("昇腾 多模态 推理加速")))
    check("无命中返回空", extractor.extract("completely unrelated text about gardening") == [])

    gate = ScopeGate(config)
    check("主题闸门放行", gate.passes_gate("multimodal serving framework"))
    check("主题闸门拦截", not gate.passes_gate("gardening tips for spring"))
    check("系统不再提供打分方法", not hasattr(gate, "score"))


# 2026-W38 报告中出现过的真实标题，用于锁定「调研范围」边界：
# 前四条属多模态 infra（系统工程 / 降本增效），后四条不属 infra（仿脑、具身、能力评测、算法创新）。
SCOPE_SAMPLES: list[tuple[str, str, bool]] = [
    (
        "StackTok: Budget-Adaptive Visual Token Selection to Accelerate VLM Inference",
        "We select visual tokens adaptively under a compute budget, cutting the number of tokens the "
        "vision encoder feeds to the language model and speeding up multimodal inference with minimal accuracy loss.",
        True,
    ),
    (
        "OmniKVQuant: KV Cache Quantization for Omni-LLMs",
        "We apply post-training quantization to the KV cache of an omni-modal LLM to reduce peak memory "
        "and raise serving throughput on long-context multimodal workloads.",
        True,
    ),
    (
        "VC-Attention: Numerical Smoothing and Softmax Promotion for Low-Bit Attention",
        "Low-bit attention kernels in multimodal inference engines suffer from overflow; we stabilise them "
        "numerically and report throughput gains for quantized operators.",
        True,
    ),
    (
        "Efficient Multimodal Serving with Encoder Disaggregation",
        "We disaggregate the vision encoder from the prefill stage of a multimodal inference engine to "
        "improve end-to-end latency and GPU utilisation.",
        True,
    ),
    (
        "BrainFocus: EEG-Guided ROI Selection for Efficient Vision-Language Models",
        "Electroencephalogram (EEG) signals guide region-of-interest selection so that the VLM only "
        "processes part of the image, reducing its visual-side inference cost. A brain-inspired approach.",
        False,
    ),
    (
        "TEMPO: Temporal Context Learning for Dynamic Robot Manipulation",
        "We let a robot manipulation policy learn temporal context to improve decision making in dynamic tasks.",
        False,
    ),
    (
        "What Do Hallucinations Reveal About Multimodal Reasoning? Probing Visual Grounding Failures",
        "We use contrastive decoding probes to localise visual grounding failures in multimodal models.",
        False,
    ),
    (
        "Sparse MLLM Anchors with Dense Adaptation: Breaking the Self-Reference Loop",
        "We propose sparse anchors plus dense adaptation to stabilise online test-time adaptation of a "
        "multimodal large language model.",
        False,
    ),
]


def test_scope(config: dict) -> None:
    print("\n[+] 调研范围（scope）闸门")
    gate = ScopeGate(config)
    check("论文维度要求 infra 证据", gate.requires_infra_evidence("papers"))
    check("公众号维度要求 infra 证据", gate.requires_infra_evidence("wechat"))
    check("仓库维度不要求 infra 证据", not gate.requires_infra_evidence("repos"))
    check("博客类条目要求 infra 证据", gate.requires_infra_evidence("teams", None, "blog"))
    check("媒体文章要求 infra 证据", gate.requires_infra_evidence("teams", None, "media-article"))
    check("模型发布不要求 infra 证据", not gate.requires_infra_evidence("teams", None, "model-release"))
    check("论文维度启用排除法", gate.should_check_exclude("papers"))
    check("仓库维度不受排除法影响", not gate.should_check_exclude("repos"))
    check(
        "逐源开关可覆盖分组默认值",
        gate.requires_infra_evidence("papers", {"requireInfraEvidence": False}) is False,
    )
    check("scope 词表非空", bool(gate.infra_terms) and bool(gate.exclude_terms))

    for title, summary, expected in SCOPE_SAMPLES:
        text = f"{title} {summary}"
        reason = gate.out_of_scope_reason(text)
        passed = gate.passes_gate(text, require_infra=True, title=title) and not reason
        label = "收录" if expected else "拦截"
        detail = f"reason={reason or '-'}"
        check(f"scope {label}：{title[:42]}", passed == expected, detail)

    check(
        "排除法先于 infra 证据（脑电命中排除词）",
        gate.out_of_scope_reason(SCOPE_SAMPLES[4][0] + " " + SCOPE_SAMPLES[4][1]) == "eeg",
    )
    check(
        "中文排除词生效（公众号维度）",
        gate.out_of_scope_reason("具身机器人大模型推理加速实践") in {"机器人", "具身"},
    )
    check(
        "纯算法论文缺 infra 证据被拦",
        not gate.passes_gate(SCOPE_SAMPLES[7][0] + " " + SCOPE_SAMPLES[7][1], require_infra=True),
    )
    check(
        "不要求 infra 证据时仍按主题放行",
        gate.passes_gate(SCOPE_SAMPLES[7][0] + " " + SCOPE_SAMPLES[7][1], require_infra=False),
    )
    # 摘要里 throughput / latency / kernel 这类词人人都写，只有标题才算 infra 证据
    summary_only = (
        "MuSeR: Scalable Long-sequence Recommendation with Multi-interest Modeling",
        "We serve a multimodal recommender with high throughput, low latency and custom GPU kernels.",
    )
    check(
        "证据只在摘要里不算 infra（标题收紧）",
        not gate.passes_gate(
            f"{summary_only[0]} {summary_only[1]}", require_infra=True, title=summary_only[0]
        ),
    )
    check(
        "证据里出现 infra 关键词即放行（标题命中）",
        gate.passes_gate(
            "PixelFlow: Token-Level Workload Management for Efficient Distributed DiT Serving",
            require_infra=True,
            title="PixelFlow: Token-Level Workload Management for Efficient Distributed DiT Serving",
        ),
    )


def test_extractor_phrases(config: dict) -> None:
    print("\n[+] 关键词匹配器（短语/锚定）")
    extractor = TagExtractor(
        {
            "t": [
                "kv cache",
                "sparse attention",
                "multi-gpu",
                "npu",
                "graph mode",
                "world action model",
            ]
        }
    )
    check("多词短语精确命中", extractor.matches("kv cache", "Flash-dLLM: IO-Aware KV Cache and Parallel Decoding"))
    check("短语允许连字符", extractor.matches("kv cache", "a kv-cache aware scheduler"))
    check("短语允许单词共现兜底", extractor.matches("kv cache", "KV reuse and cache eviction policy"))
    check("连字符词允许空格写法", extractor.matches("multi-gpu", "scaling multi gpu inference"))
    check("连字符词允许连写", extractor.matches("multi-gpu", "multigpu serving"))
    check("短语命中真短语", extractor.matches("sparse attention", "block-sparse attention kernels"))
    check("短语不因词序/词形误命中", not extractor.matches("graph mode", "Graph-Guided Diffusion Language Models"))
    check("排除短语命中真短语", extractor.matches("world action model", "Degradation for World Action Models"))
    check("单词词首锚定（npu 不命中 Input）", not extractor.matches("npu", "Input Transformations and Confidence Routing"))
    check("单词词首锚定（npu 命中 NPU）", extractor.matches("npu", "Ascend NPU inference"))
    check("空文本安全", not extractor.matches("kv cache", ""))


def test_dedupe(config: dict) -> None:
    print("\n[4/6] 去重")
    first = sample_item(stableId="s:1", sourceId="arxiv", sourceLabel="arXiv", sources=["arxiv"], sourceLabels=["arXiv"])
    duplicate = sample_item(stableId="s:1", sourceId="hf", sourceLabel="HF", sources=["hf"], sourceLabels=["HF"])
    merged = dedupe([first, duplicate])
    check("同 stableId 合并", len(merged) == 1, str(len(merged)))
    check("来源被合并", set(merged[0]["sources"]) == {"arxiv", "hf"}, str(merged[0]["sources"]))
    check("来源标签被合并", set(merged[0]["sourceLabels"]) == {"arXiv", "HF"}, str(merged[0]["sourceLabels"]))

    title_a = sample_item(stableId="s:2", title="同一篇论文：Efficient Multimodal Serving")
    title_b = sample_item(stableId="s:3", title="同一篇论文 Efficient Multimodal Serving!", sourceId="s2")
    check("跨源标题指纹合并", len(dedupe([title_a, title_b])) == 1)

    distinct = dedupe([sample_item(stableId="s:4"), sample_item(stableId="s:5", title="另一篇完全不同的工作")])
    check("不同条目不误合并", len(distinct) == 2)


def test_corpus(config: dict) -> None:
    print("\n[5/6] 语料构建")
    now = now_utc()
    items = [
        sample_item(stableId="p:1", title="Multimodal serving with encoder disaggregation"),
        sample_item(
            stableId="r:1",
            title="[vllm-project/vllm-omni] 发布 v0.3.0：新增 video generation 支持",
            kind="repo-release",
            group="repos",
            sourceId="github-repos",
            sourceLabel="GitHub",
            repo="vllm-project/vllm-omni",
            repoGroup="serving",
            repoActivity="release",
            summary="add multimodal video generation serving support with cache reuse",
        ),
        sample_item(
            stableId="m:1",
            title="新模型发布：Qwen/Qwen3-VL-32B",
            kind="model-release",
            group="teams",
            sourceId="hf-models",
            sourceLabel="HuggingFace 模型发布雷达",
            summary="qwen-vl multimodal video image",
        ),
        sample_item(
            stableId="w:1",
            title="昇腾 MindIE 多模态推理加速实践",
            kind="wechat-article",
            group="wechat",
            sourceId="bing-wechat",
            sourceLabel="Bing · 微信公众号",
            summary="昇腾 多模态 推理 加速 mindie",
        ),
    ]
    corpus = build_corpus(items, config=config, run_id="test-run", week="2026-W38", reference=now)

    check("语料条目数", corpus.stats["total"] == 4, str(corpus.stats["total"]))
    check("窗口内全部为新增", corpus.stats["new"] == 4, str(corpus.stats["new"]))
    check("分组统计", corpus.stats["byGroup"].get("repos") == 1, str(corpus.stats["byGroup"]))
    check("仓库聚类", len(corpus.repo_clusters) == 1, str(len(corpus.repo_clusters)))
    check("周边团队矩阵有行", len(corpus.peer_matrix["rows"]) > 0)
    check("周边团队雷达结构", isinstance(corpus.peer_matrix.get("radar"), list))
    check("主题聚类可用", isinstance(corpus.themes, list))
    check("条目不含分数字段", all("score" not in item for item in corpus.items))

    # 中文约束：未补中文的条目进不了正文，只进待补清单
    uncurated = sample_item(stableId="p:2", title="An English-only paper", zhCurated=False)
    mixed = build_corpus([items[0], uncurated], config=config, run_id="test-run-zh", week="2026-W38", reference=now)
    check("未补中文的条目不进正文", mixed.stats["total"] == 1, str(mixed.stats["total"]))
    check("未补中文计入待补清单", mixed.stats["pendingZh"] == 1 and len(mixed.pending_zh) == 1)

    repeat = build_corpus(items, config=config, run_id="test-run-2", week="2026-W38", reference=now)
    check("重跑结果一致（可复现）", repeat.stats == corpus.stats)

    stale = sample_item(stableId="old:1", published=iso(now - timedelta(days=60)))
    aged = build_corpus([stale], config=config, run_id="test-run-3", week="2026-W38", reference=now)
    check("超出窗口不标记新增", aged.stats["new"] == 0, str(aged.stats["new"]))


def test_render(config: dict, source_reports: list[dict]) -> None:
    print("\n[6/6] 渲染")
    items = [
        sample_item(stableId=f"x:{index}", title=f"多模态推理论文 {index}") for index in range(6)
    ]
    items.append(
        sample_item(
            stableId="w:kol",
            title="大V渠道文章：多模态推理加速实践",
            kind="wechat-article",
            group="wechat",
            sourceId="sogou-wechat",
            sourceLabel="搜狗微信 · 公众号检索",
            signals={"kol": "某明星大V", "kolCategory": "明星大V"},
        )
    )
    # 未补中文的条目必须不出现在正文（中文约束的端到端断言）
    items.append(
        sample_item(stableId="en:1", title="English Only Leaked Title", zhCurated=False, group="papers")
    )
    corpus = build_corpus(items, config=config, run_id="render-run", week="2026-W38")
    window = {"since": "2026-09-02", "until": "2026-09-16", "maxAgeDays": 14}
    html = render_html(
        corpus,
        config=config,
        source_reports=source_reports,
        window=window,
        run_id="render-run",
        generate_command="python scripts/report/run_weekly.py",
    )
    check("未补中文的条目不进正文", "English Only Leaked Title" not in html)
    check("公众号维度标注大V渠道", "某明星大V" in html)
    for token in (
        "<!doctype html>",
        "rp-section",
        'id="overview"',
        'id="papers"',
        'id="teams"',
        'id="repos"',
        'id="wechat"',
        'id="peers"',
        'id="appendix"',
        "rpSearch",
        "维度",
    ):
        check(f"HTML 含 {token}", token in html)
    check("不再有竞品/昇腾竞争力章节", "竞品" not in html and "昇腾竞争力" not in html)
    check("不再有高分条目", "高分条目" not in html)
    check("不再有按检索词分布", "按检索词分布" not in html)
    check("不再输出分数列", ">分<" not in html)
    check("HTML 尺寸合理", 8000 < len(html) < 8_000_000, str(len(html)))
    check("HTML 无外部依赖", "http://cdn" not in html and "https://cdn" not in html)
    check("HTML 自包含（无外部资源引用）", not re.search(r'(?:src|@import)\s*=?\s*["\(]https?://', html))
    check("HTML 无未转义模板残留", "{" not in html.split("<style>")[0].replace("{{", ""))
    check("报告指向仓库而非存档页", "../" not in html.split("<body>")[1][:400])
    check("报告写明收录范围", "收录范围" in html and "不属多模态 infra" in html)
    check("附录含范围拦截列", "范围拦截" in html)
    check("首页含收录边界说明", "收录边界" in html)


def test_config(config: dict) -> None:
    print("\n[+] 配置与源构建")
    sources = build_sources(config)
    check("源数量 > 10", len(sources) > 10, str(len(sources)))
    ids = [source.get("id") for source in sources]
    check("source id 唯一", len(ids) == len(set(ids)), str([i for i in ids if ids.count(i) > 1]))
    kinds = {source.get("kind") for source in sources}
    check(
        "关键 kind 齐备",
        {"arxiv", "rss", "github", "sogou-wechat", "hf-daily-papers", "hf-models"} <= kinds,
        str(sorted(kinds)),
    )
    check("仓库索引非空", len(repo_index(config)) > 10, str(len(repo_index(config))))
    check("commit 噪声过滤", not is_interesting_commit("chore: bump version"))
    check("commit 信号保留", is_interesting_commit("feat: add multimodal video generation support"))
    check("Atom 提交前缀保留", is_interesting_commit("[Perf][CosyVoice3] bounded-window streaming vocoder"))
    check("GitHub 通道为 atom", str(config["repos"].get("channel")) == "atom")
    check("报告启用中文约束", bool(config["report"]["chineseOnly"]))
    check("配置不再有打分参数", "scoring" not in config and "minScore" not in config["defaults"])
    check(
        "仓库维度只看多模态",
        not {
            "hiyouga/LLaMA-Factory",
            "modelscope/ms-swift",
            "NVIDIA/Megatron-LM",
            "Ascend/MindIE-LLM",
        }
        & {entry["repo"] for entry in config["repos"]["watch"]},
    )
    check("通用主仓只保留多模态子系统通道", {"vllm-mm", "sglang-mm"} <= {entry.get("id") for entry in config["repos"]["watch"]})
    check("周边团队配置存在", bool(config["peers"]["engines"]) and bool(config["peers"]["roster"]))
    check("不再有昇腾竞争力配置", "mindie" not in config)
    check("大V渠道并入公众号维度", len(config["wechat"]["kolChannels"]) >= 8)
    check(
        "大V渠道都带检索词",
        all(str(channel.get("query") or "").strip() for channel in config["wechat"]["kolChannels"]),
    )
    try:
        json.dumps(config, ensure_ascii=False)
        check("配置可序列化", True)
    except (TypeError, ValueError) as error:
        check("配置可序列化", False, str(error))


SOGOU_SAMPLE = """
<ul class="news-list">
<li id="sogou_vr_11002601_box_0">
  <h3><a href="/link?url=abc123">视觉Token<em>剪枝</em>、扩散加速与Agent执行结构</a></h3>
  <p class="txt-info">涵盖3D推理效率、视频生成加速、LLM推理压缩</p>
  <div class="s-p"><a class="account" href="javascript:void(0)">机器之心</a></div>
  <script>document.write(timeConvert('1786147200'))</script>
</li>
<li id="sogou_vr_11002601_box_1">
  <h3><a href="https://mp.weixin.qq.com/s/xyz">多模态大模型推理加速实践</a></h3>
  <p class="txt-info">vLLM 与 MindIE 的部署对比</p>
  <script>document.write(timeConvert('1786233600'))</script>
</li>
</ul>
"""


def test_wechat_channels(config: dict) -> None:
    print("\n[+] 公众号检索词与大V渠道")
    from report.collect import Collector
    from report.lib.httpclient import HttpClient, SnapshotCache

    # 不写系统临时目录（受限环境下清理会失败），用脚本目录下的缓存子目录
    cache_root = SCRIPT_DIR / ".cache" / "_selftest_snapshots"
    client = HttpClient(SnapshotCache(cache_root), offline=True, logger=lambda message: None)
    collector = Collector(config, client, logger=lambda message: None)
    searches = collector.wechat_searches()
    kol = [search for search in searches if search.get("kol")]
    check("检索词含常规关键词", len(searches) >= 20, str(len(searches)))
    check("检索词含大V渠道", len(kol) >= 8, str(len(kol)))
    check("大V渠道带检索词与归属", all(search["query"] and search["kol"] for search in kol))
    check("检索词 id 唯一", len({search["id"] for search in searches}) == len(searches))
    check("HTTP 客户端支持拒绝落快照", "cache_when" in HttpClient.get.__code__.co_varnames)


def test_sogou(config: dict) -> None:
    print("\n[+] 搜狗微信结果解析")
    hits = parse_sogou_wechat(SOGOU_SAMPLE)
    check("解析条目数", len(hits) == 2, str(len(hits)))
    check("标题清洗掉 em 标签", hits and "<em>" not in hits[0]["title"], hits[0]["title"] if hits else "")
    check("摘要提取", hits and "视频生成加速" in hits[0]["summary"])
    check("公众号名提取", hits and hits[0]["account"] == "机器之心", hits[0]["account"] if hits else "")
    check("发布时间解析", hits and parse_datetime(hits[0]["published"]) is not None)
    check("相对链接补全为绝对地址", hits and hits[0]["url"].startswith("https://weixin.sogou.com/"))
    check("绝对链接保持原样", len(hits) > 1 and hits[1]["url"] == "https://mp.weixin.qq.com/s/xyz")
    check("空输入安全", parse_sogou_wechat("") == [])


def main() -> int:
    print("=" * 68)
    print("AscendPlayground 多模态 Infra 周报 · 离线自检")
    print("=" * 68)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_reports = [
        {
            "id": "arxiv",
            "label": "arXiv",
            "kind": "arxiv",
            "group": "papers",
            "fetched": 10,
            "kept": 6,
            "errors": [],
            "enabled": True,
        },
        {
            "id": "github-repos",
            "label": "GitHub 仓库动态",
            "kind": "github",
            "group": "repos",
            "fetched": 3,
            "kept": 2,
            "errors": ["vllm-project/vllm releases: HTTP 403"],
            "enabled": True,
        },
    ]

    test_util()
    test_feed()
    test_relevance(config)
    test_scope(config)
    test_extractor_phrases(config)
    test_dedupe(config)
    test_corpus(config)
    test_render(config, source_reports)
    test_config(config)
    test_wechat_channels(config)
    test_sogou(config)

    print("\n" + "=" * 68)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  - {name}")
        print("=" * 68)
        return 1
    print("全部通过 ✅")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
