# 多模态 Infra 情报系统 · 复现手册

AscendPlayground 的多模态 Infra 调研报告系统。面向 **weekly 频率** 的调研工作流：
自动采集多模态 infra 论文、主要技术团队动态、核心多模态仓库更新细则、公众号与中文媒体，
并横向对照 vLLM / SGLang / M\* / TensorRT-LLM / Dynamo 与 **昇腾 MindIE** 的工程节奏，
产出可直接阅读的 HTML 周报。

- **零第三方依赖**：只使用 Python 标准库（3.8+ 即可运行），无需 pip install。
- **免鉴权数据源**：全部走公开通道，不需要任何 API Key。
- **可复现**：每次抓取的原始响应都会落盘为快照，`--offline` 可完全离线重跑并得到一致结果。

---

## 1. 快速开始

```bash
cd AscendPlayground

# 生成当周报告（联网采集）
python scripts/report/run_weekly.py

# 离线复现（只读快照，完全不联网）
python scripts/report/run_weekly.py --offline

# 离线自检（112 项断言，不联网）
python scripts/report/selftest.py

# 查看当前配置里的全部信息源
python scripts/report/run_weekly.py --list-sources
```

`report/` 目录**只放 HTML 产物**：

| 产物 | 说明 |
|------|------|
| `report/<周>.html` | 当周可视化报告（自包含、无外部依赖，可离线打开；同一周重复运行直接覆盖） |

采集缓存放在脚本目录下，**不属于产物、可随时删除**（删掉后重新联网跑一次即可重建）：

| 缓存 | 说明 |
|------|------|
| `scripts/report/.cache/snapshots/` | 原始 HTTP 响应快照（`.body` 正文 + `.json` 元数据：URL/状态码/抓取时间/哈希），仅 `--offline` 复现时使用。已在 `.gitignore` 中忽略 |

> 设计取舍：报告只输出 HTML，不额外生成 Markdown / 清单 / 语料中间产物。
> “本周新增”由发布时间窗口直接推导，因此**同一份快照重跑结果完全一致**，无需状态文件。

---

## 2. 调研范围（Scope）：只收「多模态 infra」

主题边界由 `config/sources.json` 的 `scope` 段**硬约束**，不靠人工挑选、也不靠源自身可靠：
每条新增条目入库前必须过两道闸门（`lib/relevance.py` 的 `Scorer`，由 `collect.py::_finalize` 调用）。

| 闸门 | 作用对象 | 规则 | 命中后 |
|------|----------|------|--------|
| ① 排除法 `scope.exclude` | `scope.excludeGroups`（默认 `papers` / `wechat` / `teams`） | 标题+摘要命中任一排除词即出局：脑电/脑机/认知启发、医疗临床、教育、遥感农业、机器人具身、可信与幻觉、数据标注与仿真环境 | 丢弃，计入附录「范围拦截」 |
| ② infra 证据 `scope.infraEvidence` | `scope.requireInfraEvidenceGroups`（默认 `papers`） | **标题**必须出现任一「系统工程 / 降本增效」证据词：服务引擎、调度与批处理、prefill/decode、算子与 kernel、KV cache 与显存、量化压缩、并行分布式、昇腾/NPU 适配、部署与评测工具 | 丢弃，计入附录「范围拦截」 |

> 为什么证据词只看标题（`scope.evidenceInTitle`，默认 `true`）：
> 摘要里 `throughput` / `latency` / `kernel` / `deployment` 这类词人人都写——推荐系统、等离子体仿真、
> 密码学论文都会写。实测（2026-W39 同一份 arXiv 快照，150 条抓取）：
> 只看正文证据 ⇒ 保留 42 条，仍混入推荐、仿真、材料等非 infra 工作；
> 只看标题证据 ⇒ 保留 15 条左右，剩下的基本是投机解码、KV cache、视觉 token 剪枝、
> 稀疏注意力算子、DiT serving、低比特量化这类真 infra 工作。

**收录（in scope）**：多模态 / 生成式模型的**推理与服务的系统工程** ——
服务引擎（vLLM / SGLang / M\* / TensorRT-LLM / Dynamo / MindIE 等）、调度与 continuous batching、
prefill-decode 与 EPD 分离、KV cache 与显存管理、视觉 token 压缩、算子与 kernel、编译与图模式、
并行（TP/SP/CP/EP）、量化与低比特部署、硬件与 NPU 适配、扩散/视频生成的推理加速
（步数蒸馏、缓存复用、VAE 并行）、系统级评测与 profiling。

**不收录（out of scope）**：模型能力与算法创新（新架构、新目标函数、能力基准）、
应用落地（医疗、教育、遥感、工业质检）、仿脑与脑信号（EEG / fMRI / BCI）、
具身与机器人（VLA 策略、操作、导航、数据与仿真工厂）、可信与对齐（幻觉缓解、公平性、可解释性）、
纯训练方法（除非本身是系统工作：并行、显存、算子）。

> 边界示例（已固化为 `selftest.py` 断言，回归即报警）：
>
> - ❌ `BrainFocus: EEG-Guided ROI Selection for Efficient Vision-Language Models` —— 仿脑降算力，视觉侧算法，不是 infra；
> - ✅ `StackTok: Budget-Adaptive Visual Token Selection to Accelerate VLM Inference` —— 推理侧 token 压缩；
> - ❌ `TEMPO: Temporal Context Learning for Dynamic Robot Manipulation` —— 具身策略；
> - ✅ `OmniKVQuant: KV Cache Quantization for Omni-LLMs` —— 显存与吞吐优化。

**收放办法（改配置，不改代码）**：

- 想把具身/机器人也一起收：从 `scope.exclude` 删掉 `robot` / `robotic` / `embodied` / `机器人` / `具身`；
- 想让论文维度更宽：从 `scope.requireInfraEvidenceGroups` 删掉 `papers`（则只做排除法），
  或把 `scope.evidenceInTitle` 改成 `false`（退回「标题或摘要任一命中」）；
- 单个源单独开关：在该源上加 `"requireInfraEvidence": false` 或 `"skipScopeExclude": true`；
- **改完先审计再出报告**（不会写任何文件）：

```bash
# 列出被拦条目与命中词，用来判断边界是否合适
python scripts/report/scope_audit.py --offline --only arxiv
```

`scope_audit.py` 的做法是：用**跳过 scope** 的配置采一遍，得到全量候选，再用当前 scope 规则逐条判定，
于是「保留 / 排除法拦 / 标题缺 infra 证据拦」三类都能看到，并给出命中词。

---

## 3. 常用参数

| 参数 | 作用 |
|------|------|
| `--offline` | 只用快照离线复现，绝不联网；缺快照会明确报错 |
| `--refresh` | 忽略旧快照，强制重新抓取全部源 |
| `--max-age-days N` | 采集时间窗口（默认 14 天） |
| `--min-score F` | 相关性打分下限（默认 1.2）；调低可捞回更多条目 |
| `--top N` | 每个维度最多列出多少条（默认 25） |
| `--only a,b` / `--skip a,b` | 只采集 / 跳过指定 source id |
| `--run-id ID` | 自定义 Run ID（默认按当前 UTC 时间） |
| `--generated-at TS` | 冻结生成时间（如 `2026-09-16T00:00:00Z`），配合 `--run-id` 实现逐字节可复现 |
| `--dry-run` | 只采集与统计，不写任何文件 |
| `--quiet` | 静默模式 |

### 复现验证

```bash
# 同一 Run ID + 同一快照 + 冻结时间 ⇒ 输出逐字节一致
python scripts/report/run_weekly.py --offline --run-id repro-A --generated-at 2026-09-16T00:00:00Z
# 重复执行后比对报告哈希
Get-FileHash report/2026-W38.html -Algorithm SHA256    # PowerShell
sha256sum report/2026-W38.html                          # Linux/macOS
```

---

## 4. 信息源方案（当前实测状态）

配置集中在 `config/sources.json`，**改配置即可增删源，无需改代码**。

### 维度 A · 多模态 Infra 论文

| 源 | 接口 | 状态 |
|----|------|------|
| arXiv | `export.arxiv.org/api/query`，15 组关键词（`all:"a" AND all:"b"`） | ✅ 主力 |
| HuggingFace Daily Papers | `huggingface.co/api/daily_papers` | ✅ 热度交叉验证 |
| OpenAlex | `api.openalex.org/works` | ⛔ 默认禁用（见下） |

> **召回靠查询，精度靠 `scope`**：arXiv 的 AND 组合查询负责「捞全」，第 2 节的 `scope`
> 闸门负责「收窄」——像「VLM + inference」这种只在字面命中、实质是模型能力或应用落地的论文，
> 会被排除法或 infra 证据要求挡在报告之外（例：`BrainFocus` 这类仿脑降算力工作）。

> **OpenAlex 为什么默认关闭**：它是全文检索，返回结果里大量是建筑安全、医学影像、
> 教育等无关领域论文；而它的 metadata 又几乎不命中本项目的多模态 infra 关键词组，
> 导致任何足够严格的关键词阈值都会把真正相关的论文一起滤掉（实测：阈值设为
> `minKeywordHits>=1` 时保留数直接归零）。论文维度已由 arXiv 的精确 AND 组合查询
> 加 HuggingFace Daily Papers 充分覆盖，故默认关闭；如需机构/引用维度，
> 可在 `config/sources.json` 中收窄 `searches` 后重新启用。

### 维度 B · 技术团队与官方动态

| 源 | 状态 |
|----|------|
| HuggingFace Blog / NVIDIA Developer Blog / OpenAI News | ✅ 官方博客 RSS |
| HuggingFace 模型发布雷达 | ✅ 按组织轮询新模型（Qwen / DeepSeek / MiniMax / Tencent / Lightricks 等） |
| 量子位 / 雷峰网 AI科技评论 / 钛媒体 | ✅ 中文技术媒体 RSS |
| 机器之心 RSS | ❌ 已下线（返回 HTML 落地页），默认禁用，改由搜狗微信通道覆盖 |

### 维度 C · 主要多模态仓库更新细则

**默认走 GitHub Atom feed**（`github.com/<repo>/releases.atom` 与 `commits.atom`）：

- **无限流、无需鉴权**，含完整 release 说明与 commit 信息，是最稳的通道；
- 覆盖 **30 个启用的监控项**，按优先级排序（昇腾侧 → 服务框架 → 生成加速 → 模型仓 → 竞品基准）；
- 主仓的多模态子系统单独建通道：例如 **sglang 主仓**通过
  `commits/main/python/sglang/srt/multimodal.atom` 只跟踪多模态目录下的改动
  （配合 `pathFilter` 配置项），避免被通用提交淹没；
- 每条变更都抽取一句到两句**变更说明**（`lib/util.py` 的 `digest` / `commit_digest`），
  自动剔除 changelog 样板、`[Docs]`/`[CI]`/`[chore]`/`[branding]` 类杂务提交、
  `Co-authored-by` 等 trailer，因此报告正文可直接阅读；
- `Ascend/cann`、`Ascend/ops-nn` 在 GitHub 上没有 Atom feed（404），已在配置中禁用并注明原因。

> REST API 通道保留为回退（`repos.channel: "api"`），但匿名限额只有 60 次/小时。
> 走 API 时内置三层保护：`repos.requestBudget` 限制单轮请求量、release 优先逐仓裁剪、
> 连续 2 次 403 即停止该维度。命中本地快照的请求不消耗配额。

### 维度 D · 公众号与中文媒体

| 通道 | 状态 |
|------|------|
| **搜狗微信** `weixin.sogou.com/weixin` | ✅ **主力通道**：12 组关键词，标题为 **mp.weixin.qq.com 原文直链** |
| 量子位 / 雷峰网 AI科技评论 / 钛媒体 RSS | ✅ 作为公众号内容同源的稳定镜像 |
| Bing `site:mp.weixin.qq.com` | ❌ 实测匿名抓取被投毒，默认禁用 |
| DuckDuckGo HTML | ❌ 匿名 POST 返回 202 反爬页，默认禁用 |
| 机器之心 RSS | ❌ 官方 RSS 已下线，默认禁用，由搜狗通道覆盖 |

**原文直链解析**（`_resolve_sogou_link`）踩过的三个坑：

1. 搜狗返回的 `href` 内嵌换行等**控制字符**，必须先清理，否则请求非法（`InvalidURL`）；
2. 必须使用 **cookie 会话**（先请求搜索页拿到 `SUID`/`SNUID`），否则跳转请求返回反爬页；
3. 跳转页用 `url += '...'` 分片拼接真实地址，且 `&timestamp=` 会被 HTML 实体解码误伤成
   `×tamp=`，需要还原。

实现为：浏览器 UA + cookie 会话 → 逐条解析 → 失败则回退到检索结果链接。
解析请求走 `use_cache=False`（链接带时效签名，缓存会导致过期结果被复用）。
配置项 `wechat.primary.politeDelaySeconds`（默认 2.5 秒）控制请求间隔。

### 维度 E/F · 竞品版本特性与 MindIE 竞争力

- **竞品发版特性说明**：逐个引擎列出最近 3 个版本的官方 release 特性说明（版本号 + 日期 + 特性要点），
  重点是「这版做了什么」，不再以发版次数/日均信号等节奏指标作为展示重点。
- **自有仓库（MindIE-LLM / MindIE-SD / vLLM-Ascend）**：同样展示最新发版的特性说明；
  若该仓在采集窗口内没有发新版，会回退到窗口外的最近一次发版并标注「窗口外」。
- **能力覆盖探针**：用 8 个能力域（Qwen-VL、InternVL、视频生成、扩散加速、稀疏注意力、omni 服务、MoE、量化）
  反查生态语料，输出「有活跃信号 / 信号稀少 / 本周无信号」与代表条目，用于判断昇腾当前覆盖与缺口。

---

## 5. 代码结构

```
scripts/report/
├── run_weekly.py          CLI 入口：采集 → 归一化 → 渲染
├── collect.py             采集层：各源适配器 + 公众号/搜狗结果解析
├── corpus.py              语料层：去重、打分、主题聚类、竞品与能力矩阵
├── render.py              渲染层：自包含 HTML 报告
├── zh.py                  中文说明层：按 stableId / repo+tag 套用人工撰写的中文
├── selftest.py            离线自检（112 项断言，含调研范围边界回归）
├── scope_audit.py         调研范围审计：列出被 scope 拦下的条目与命中词（调边界用）
├── config/
│   ├── sources.json           信息源注册表（改这里就能增删源）
│   ├── curated_zh.json        人工撰写的中文标题与说明（按 stableId 匹配）
│   └── curated_releases_zh.json 人工撰写的中文发版说明（按 repo + tag 匹配）
└── lib/
    ├── httpclient.py      HTTP + 快照缓存 + 重试 + cookie 会话
    ├── feed.py            RSS 2.0 / Atom 1.0 解析（替代 feedparser）
    ├── relevance.py       关键词抽取、打分、跨源去重
    └── util.py            时间/文本/ID 工具 + 说明文字抽取
```

### 4.1 中文说明层（重要）

报告的标题与说明**以中文为准**，由 `config/curated_zh.json` 与
`config/curated_releases_zh.json` 提供，全部由人工撰写（不调用任何翻译服务）：

- 按 `stableId`（条目）与 `repo + tag`（发版）精确匹配，命中即覆盖标题与说明；
- 未命中的条目回退到自动抽取的原文说明，并在 `report/pending_zh.json` 中列出待补清单；
- 中文标题会同时保留原文标题（`originalTitle` / `originalTitleKey`），
  以确保跨源去重仍按原文标题指纹进行（否则中英标题会变成两条）；
- 首次运行新周期时，`report/pending_zh.json` 就是需要补写中文的清单。

> 设计取舍：说明文字**不由脚本自动翻译**（无 LLM、无翻译 API 调用），
> 因此质量与术语可控，且同一份快照重跑结果完全一致。

### 4.2 处理链路

```
信息源 (config) → 抓取 + 快照落盘 → 归一化条目 + 抽取说明 → 主题闸门 → 范围闸门（排除法 + infra 证据）
   → 相关性打分 → 跨源去重 → 套用中文说明层 → 分类聚合（维度 A–F）→ HTML 渲染
```

**范围闸门**（见第 2 节）：先按 `scope.exclude` 排掉非 infra 议题，再按 `scope.infraEvidence`
要求论文维度给出「系统工程 / 降本增效」证据；被拦条目数量在报告附录 G 与运行日志中可见。

**打分模型**（`config.scoring` 可调参）：

```
score = (1 + 加权关键词命中，上限 12) × 源权重 × 时效因子 × 新增加成
```

- 标题命中权重 3.0、摘要 1.0；`heavy` 类关键词 3.0 分、`medium` 1.5 分；
- 时效因子在 7 天内为 1.0，之后线性衰减到 0.35。

**去重策略**：先按 `stableId`（源内唯一）合并，再按规范化标题指纹跨源合并同一事件。

**新增标记**：发布时间落在最近 7 天（`scoring.lookbackDays`）内记为 `NEW`，否则 `SEEN`。

---

## 6. weekly 工作流建议

每周固定执行：

```bash
cd AscendPlayground
python scripts/report/run_weekly.py --max-age-days 8    # 覆盖上周以来
python scripts/build_site.py                            # 更新站点（含周报分区）
```

调研时的实用技巧：

- **想看得更全**：`--max-age-days 30 --min-score 0.8`（首次建基线时推荐）。
- **觉得范围收得不合适**：先跑 `python scripts/report/scope_audit.py --offline`，按拦截清单调
  `config/sources.json` 的 `scope`，再出报告（第 2 节）。
- **某一维度为空**：看报告附录 G 的源状态；若显示 403/429 即为限流，非代码问题。
- **离线交付/复现**：把 `scripts/report/.cache/` 一起带上，对方执行 `--offline` 即可得到一致结果。
- **新增公众号关键词**：编辑 `config/sources.json` 的 `wechat.searches`。
- **新增监控仓库**：编辑 `repos.watch`，越靠前优先级越高；无 Atom feed 的仓库加 `"enabled": false`。

---

## 7. 已知限制与诚实说明

1. **GitHub 通道**：默认 Atom feed 已无限流问题；若切回 `channel: "api"`，匿名限额 60 次/小时，
   27 个仓库 + 10 个竞品引擎单轮无法全覆盖，系统会明确报告未覆盖的仓库而非静默丢数据。
2. **公众号原文直链**：解析依赖搜狗会话 cookie，且直链带时效签名——因此
   `--offline` **无法**重建公众号维度的链接（会回退到检索结果链接），
   其余维度均可离线复现。解析请求不走快照缓存也是这个原因。
3. **搜狗限流**：连续快速请求会触发人机校验。系统会检测校验页并在连续 2 次后
   提前结束该维度（避免无效请求与加重限流），并在报告附录中标注。遇到这种情况
   等十几分钟重跑即可，已有快照不会浪费。
4. **搜索引擎**：Bing / DuckDuckGo 匿名抓取已不可用，适配器保留但默认禁用。
4. **机器之心**：官方 RSS 已下线；其内容通过搜狗微信通道按关键词覆盖，非全量订阅。
5. **OpenAlex** 在负载高时会返回 429，系统重试 3 次后降级，不影响其它源。
6. **中文媒体 RSS** 字段质量不一：部分源不提供发布时间，这类条目不会被标记为 `NEW`。
7. **说明文字是抽取而非生成**：`digest` 从原始正文抽取要点（无 LLM 参与），
   因此完全确定、可复现；极少数条目正文为空时会显示「—」。
7. **跨源去重**：当前快照窗口内 arXiv 与 HuggingFace Daily Papers 尚未出现重复条目
   （报告显示"跨源去重合并 0"），去重逻辑本身有单元测试覆盖。
8. **范围闸门是关键词规则而非语义判断**：它按「标题里的证据词 + 排除词表」工作，因此
   - 仍会有边界样本漏过（例：能力基准标题里写了 `Inference`、具身工作标题写了 `Quantization`）；
   - 也会误杀（例：标题只写 `Accelerating Diffusion Sampling` 而没写具体手段的加速论文）。
   两种情况都能用 `scope` 词表 + `scope_audit.py` 收敛；这是**可解释、可复现、零依赖**的取舍，
   换来的是每周结果稳定、不依赖任何 LLM 调用。**不要**把它当成语义分类器。
9. **关键词匹配按词首锚定**：`npu` 不会命中 `Input`、`dit` 不会命中 `audit`、`mode` 不会命中
   `Models`（`lib/relevance.py` 的 `_ascii_pattern` / `_tokens_nearby`）。改动词表时若发现
   命中异常，先确认是不是锚定规则导致的漏配。
10. **多词短语按「精确短语 + 整词邻域兜底」匹配**，不做词形还原：`kv cache` 命中
    `KV Cache` / `kv-cache`，但不命中 `KV Caching`（`cache` ≠ `caching`）。
    需要覆盖词形变化时，在词表里补一条变体（如再加 `kv caching`）即可。
    另注：`_ascii_pattern` 早期版本用「两次 `str.replace`」拼接，会把第一次插入的
    `[\s\-]+` 再次改写，产出必然失配的模式（多词短语因此静默退化成子串共现），
    现已改为按分隔符一次拼装，并加了回归断言。
