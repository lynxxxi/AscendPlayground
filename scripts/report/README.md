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

# 离线自检（68 项断言，不联网）
python scripts/report/selftest.py

# 查看当前配置里的全部信息源
python scripts/report/run_weekly.py --list-sources
```

`report/` 目录**只放产物**：

| 产物 | 说明 |
|------|------|
| `report/<周>.html` | 当周可视化报告（自包含、无外部依赖，可离线打开；同一周重复运行直接覆盖） |
| `report/README.md` | 阅读说明 |

采集缓存放在脚本目录下，**不属于产物、可随时删除**（删掉后重新联网跑一次即可重建）：

| 缓存 | 说明 |
|------|------|
| `scripts/report/.cache/snapshots/` | 原始 HTTP 响应快照（`.body` 正文 + `.json` 元数据：URL/状态码/抓取时间/哈希），仅 `--offline` 复现时使用。已在 `.gitignore` 中忽略 |

> 设计取舍：报告只输出 HTML，不额外生成 Markdown / 清单 / 语料中间产物。
> “本周新增”由发布时间窗口直接推导，因此**同一份快照重跑结果完全一致**，无需状态文件。

---

## 2. 常用参数

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

## 3. 信息源方案（当前实测状态）

配置集中在 `config/sources.json`，**改配置即可增删源，无需改代码**。

### 维度 A · 多模态 Infra 论文

| 源 | 接口 | 状态 |
|----|------|------|
| arXiv | `export.arxiv.org/api/query`，15 组关键词（`all:"a" AND all:"b"`） | ✅ 主力 |
| HuggingFace Daily Papers | `huggingface.co/api/daily_papers` | ✅ 热度交叉验证 |
| OpenAlex | `api.openalex.org/works` | ✅ 机构维度补充（偶发 429，已降级处理） |

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
- 覆盖 **27 个启用的仓库**，按优先级排序（昇腾侧 → 服务框架 → 生成加速 → 模型仓 → 竞品基准）；
- `Ascend/cann`、`Ascend/ops-nn` 在 GitHub 上没有 Atom feed（404），已在配置中禁用并注明原因，
  建议改从 GitCode 镜像跟踪。

> REST API 通道保留为回退（`repos.channel: "api"`），但匿名限额只有 60 次/小时。
> 走 API 时内置三层保护：`repos.requestBudget` 限制单轮请求量、release 优先逐仓裁剪、
> 连续 2 次 403 即停止该维度。命中本地快照的请求不消耗配额。

### 维度 D · 公众号与中文媒体

| 通道 | 状态 |
|------|------|
| **搜狗微信** `weixin.sogou.com/weixin` | ✅ **主力通道**：匿名可用，返回标题、摘要、发布日期、公众号名；12 组关键词 |
| 量子位 / 雷峰网 AI科技评论 / 钛媒体 RSS | ✅ 作为公众号内容同源的稳定镜像 |
| Bing `site:mp.weixin.qq.com` | ❌ 实测匿名抓取被投毒，返回与查询完全无关的结果，默认禁用 |
| DuckDuckGo HTML | ❌ 匿名 POST 返回 202 反爬页，默认禁用 |
| 机器之心 RSS | ❌ 官方 RSS 已下线（返回 HTML 落地页），默认禁用，由搜狗通道覆盖 |

> 搜狗结果的跳转链接需要人机校验才能解析出微信原文，因此条目链接指向检索结果页；
> 标题、摘要、公众号名、发布日期均为可用信息，足以判断公众号侧关注点。
> 通道间需间隔请求，配置项 `wechat.primary.politeDelaySeconds`（默认 2.5 秒）控制礼貌延时。

### 维度 E/F · 竞品节奏与 MindIE 竞争力

- **竞品发版观测**：对 8 个引擎仓库/轮抓 release tag 与时间，构建「最新版本 / 近 7 天 / 近 14 天 / 日均信号」矩阵。
- **能力覆盖探针**：用 8 个能力域（Qwen-VL、InternVL、视频生成、扩散加速、稀疏注意力、omni 服务、MoE、量化）
  反查生态语料，输出「有活跃信号 / 信号稀少 / 本周无信号」与代表条目，用于判断昇腾当前覆盖与缺口。

---

## 4. 代码结构

```
scripts/report/
├── run_weekly.py          CLI 入口：采集 → 归一化 → 渲染
├── collect.py             采集层：各源适配器 + 公众号/搜狗结果解析
├── corpus.py              语料层：去重、打分、主题聚类、竞品与能力矩阵
├── render.py              渲染层：自包含 HTML 报告
├── selftest.py            离线自检（68 项断言）
├── config/sources.json    信息源注册表（改这里就能增删源）
└── lib/
    ├── httpclient.py      HTTP + 快照缓存 + 重试
    ├── feed.py            RSS 2.0 / Atom 1.0 解析（替代 feedparser）
    ├── relevance.py       关键词抽取、打分、跨源去重
    └── util.py            时间/文本/ID 工具
```

### 处理链路

```
信息源 (config) → 抓取 + 快照落盘 → 归一化条目 → 相关性闸门 → 打分
   → 跨源去重 → 分类聚合（维度 A–F）→ HTML 渲染
```

**打分模型**（`config.scoring` 可调参）：

```
score = (1 + 加权关键词命中，上限 12) × 源权重 × 时效因子 × 新增加成
```

- 标题命中权重 3.0、摘要 1.0；`heavy` 类关键词 3.0 分、`medium` 1.5 分；
- 时效因子在 7 天内为 1.0，之后线性衰减到 0.35。

**去重策略**：先按 `stableId`（源内唯一）合并，再按规范化标题指纹跨源合并同一事件。

**新增标记**：发布时间落在最近 7 天（`scoring.lookbackDays`）内记为 `NEW`，否则 `SEEN`。

---

## 5. weekly 工作流建议

每周固定执行：

```bash
cd AscendPlayground
python scripts/report/run_weekly.py --max-age-days 8    # 覆盖上周以来
python scripts/build_site.py                            # 更新站点（含周报分区）
```

调研时的实用技巧：

- **想看得更全**：`--max-age-days 30 --min-score 0.8`（首次建基线时推荐）。
- **某一维度为空**：看报告附录 G 的源状态；若显示 403/429 即为限流，非代码问题。
- **离线交付/复现**：把 `scripts/report/.cache/` 一起带上，对方执行 `--offline` 即可得到一致结果。
- **新增公众号关键词**：编辑 `config/sources.json` 的 `wechat.searches`。
- **新增监控仓库**：编辑 `repos.watch`，越靠前优先级越高；无 Atom feed 的仓库加 `"enabled": false`。

---

## 6. 已知限制与诚实说明

1. **GitHub 通道**：默认 Atom feed 已无限流问题；若切回 `channel: "api"`，匿名限额 60 次/小时，
   27 个仓库 + 10 个竞品引擎单轮无法全覆盖，系统会明确报告未覆盖的仓库而非静默丢数据。
2. **公众号原文链接**：搜狗跳转需要人机校验，报告提供检索结果链接 + 摘要 + 公众号名 + 日期。
   若后续接入带鉴权的通道（如本地 AutoGLM 服务），在 `wechat` 下新增一个 `kind` 适配器即可。
3. **搜索引擎**：Bing / DuckDuckGo 匿名抓取已不可用，适配器保留但默认禁用。
4. **机器之心**：官方 RSS 已下线；其内容通过搜狗微信通道按关键词覆盖，非全量订阅。
5. **OpenAlex** 在负载高时会返回 429，系统重试 3 次后降级，不影响其它源。
6. **中文媒体 RSS** 字段质量不一：部分源不提供发布时间，这类条目不会被标记为 `NEW`。
7. **跨源去重**：当前快照窗口内 arXiv 与 HuggingFace Daily Papers 尚未出现重复条目
   （报告显示"跨源去重合并 0"），去重逻辑本身有单元测试覆盖。
