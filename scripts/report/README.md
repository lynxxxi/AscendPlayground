# 多模态 Infra 情报系统 · 复现手册

AscendPlayground 的多模态 Infra 调研报告系统。面向 **weekly 频率** 的调研工作流：
自动采集多模态 infra 论文、技术团队动态、多模态仓库更新细则、公众号与中文媒体（含大V渠道），
以及**周边团队**（芯片厂商 / 互联网厂商 / 有影响力的初创公司 / 周边推理框架）的版本特性，
产出可直接阅读的 HTML 周报。

- **只出中文**：报告正文只渲染已补中文说明的条目（第 3 节），英文条目不进正文。
- **不打分**：没有任何打分、分数阈值与排序权重，所有条目按发布时间排序。
- **零第三方依赖**：只使用 Python 标准库（3.8+ 即可运行），无需 pip install。
- **免鉴权数据源**：全部走公开通道，不需要任何 API Key。
- **可复现**：每次抓取的原始响应都会落盘为快照，`--offline` 可完全离线重跑并得到一致结果。

---

## 1. 快速开始

周报是**三步**流程：先联网采集得到「待补中文清单」，补完中文再离线渲染出全中文报告。

```bash
cd AscendPlayground

# 第 1 步：联网采集（新周期务必加 --refresh，否则会复用上一周期的快照）
python scripts/report/run_weekly.py --refresh

# 第 2 步：按 scripts/report/.cache/pending_zh.json 把中文补进 config/curated_zh.json
#         （{stableId: {title, digest}}，见第 3 节）

# 第 3 步：离线渲染，产出 report/<周>.html（全中文、可复现）
python scripts/report/run_weekly.py --offline

# 离线自检（137 项断言，不联网）
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
| `scripts/report/.cache/snapshots/` | 原始 HTTP 响应快照（`.body` 正文 + `.json` 元数据：URL/状态码/抓取时间/哈希），`--offline` 复现时使用。已在 `.gitignore` 中忽略 |
| `scripts/report/.cache/pending_zh.json` | 待补中文清单：本轮通过闸门但还没有中文说明的条目（即第 2 步的工作量） |

> 设计取舍：报告只输出 HTML，不额外生成 Markdown / 清单 / 语料中间产物。
> “本周新增”由发布时间窗口直接推导，因此**同一份快照重跑结果完全一致**，无需状态文件。
> 注意快照**没有过期时间**：跨周期重新采集必须 `--refresh`（缓存优先的设计是为了限流时断点续采）。

---

## 2. 调研范围（Scope）：只收「多模态 infra」

主题边界由 `config/sources.json` 的 `scope` 段**硬约束**，不靠人工挑选、也不靠源自身可靠：
每条新增条目入库前必须过两道闸门（`lib/relevance.py` 的 `Scorer`，由 `collect.py::_finalize` 调用）。

| 闸门 | 作用对象 | 规则 | 命中后 |
|------|----------|------|--------|
| ① 排除法 `scope.exclude` | `scope.excludeGroups`（默认 `papers` / `wechat` / `teams`） | 标题+摘要命中任一排除词即出局：脑电/脑机/认知启发、医疗临床、教育、遥感农业、机器人具身、可信与幻觉、数据标注与仿真环境 | 丢弃，计入附录「范围拦截」 |
| ② infra 证据 `scope.infraEvidence` | `scope.requireInfraEvidenceGroups`（默认 `papers` / `wechat`）+ `scope.requireInfraEvidenceKinds`（默认 `paper` / `blog` / `media-article`） | **标题**必须出现任一「系统工程 / 降本增效」证据词：服务引擎、调度与批处理、prefill/decode、算子与 kernel、KV cache 与显存、量化压缩、并行分布式、昇腾/NPU 适配、部署与评测工具 | 丢弃，计入附录「范围拦截」 |

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
| `--refresh` | 忽略旧快照，强制重新抓取全部源（**跨周期采集必须加**） |
| `--max-age-days N` | 采集时间窗口（默认 14 天） |
| `--top N` | 每个维度最多列出多少条（默认 25） |
| `--only a,b` / `--skip a,b` | 只采集 / 跳过指定 source id |
| `--run-id ID` | 自定义 Run ID（默认按当前 UTC 时间） |
| `--generated-at TS` | 冻结生成时间（如 `2026-09-16T00:00:00Z`），配合 `--run-id` 实现逐字节可复现 |
| `--dry-run` | 只采集与统计，不写任何文件 |
| `--quiet` | 静默模式 |

> 已取消 `--min-score`：系统不再打分，收录与否只由第 2 节的范围闸门决定，排序一律按发布时间。

### 中文汇报约束（report.chineseOnly）

`config/sources.json` 的 `report.chineseOnly = true` 表示：**报告正文只渲染已补中文的条目**。
判定标准是 `config/curated_zh.json` 里同时命中 `title` 与 `digest`（`zhCurated=True`）：

- 命中 ⇒ 进正文，标题与说明都用中文；
- 未命中 ⇒ **不进正文**，只写进 `.cache/pending_zh.json`，并在附录标注本轮有多少条待补。

这样英文条目不可能混进报告：要么补中文，要么不出现在报告里。每周的工作量就是清单长度。

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

### 维度 C · 多模态仓库更新细则

**默认走 GitHub Atom feed**（`github.com/<repo>/releases.atom` 与 `commits.atom`）：

- **无限流、无需鉴权**，含完整 release 说明与 commit 信息，是最稳的通道；
- **只看多模态仓库**：覆盖 19 个监控项（多模态服务框架 → 生成与加速 → 多模态模型 → 昇腾多模态侧）；
  训练框架（LLaMA-Factory / ms-swift / Megatron-LM）、纯 LLM 主仓与通用算子仓**不在监控范围**；
- 通用主仓只取多模态子系统：`vllm-project/vllm` 用 `pathFilter: ["vllm/multimodal"]`、
  `sgl-project/sglang` 用 `pathFilter: ["python/sglang/srt/multimodal"]`，
  避免被通用 LLM 提交淹没（这两个通道不取 release，release 特性归维度 E）；
- 每条变更都抽取一句到两句**变更说明**（`lib/util.py` 的 `digest` / `commit_digest`），
  自动剔除 changelog 样板、`[Docs]`/`[CI]`/`[chore]`/`[branding]` 类杂务提交、
  `Co-authored-by` 等 trailer，因此报告正文可直接阅读。

> REST API 通道保留为回退（`repos.channel: "api"`），但匿名限额只有 60 次/小时。
> 走 API 时内置三层保护：`repos.requestBudget` 限制单轮请求量、release 优先逐仓裁剪、
> 连续 2 次 403 即停止该维度。命中本地快照的请求不消耗配额。

### 维度 D · 公众号与中文媒体（含大V渠道）

| 通道 | 状态 |
|------|------|
| **搜狗微信** `weixin.sogou.com/weixin` | ✅ **主力通道**：12 组常规关键词 + `wechat.kolChannels` 的渠道词，标题为 **mp.weixin.qq.com 原文直链** |
| 明星大V / 关键员工 / 团队渠道 | ✅ 统一并入本维度（`wechat.kolChannels`：{name, org, category, query}，报告里标注是谁家的渠道） |
| 量子位 / 雷峰网 AI科技评论 / 钛媒体 RSS | ✅ 作为公众号内容同源的稳定镜像 |
| Bing `site:mp.weixin.qq.com` | ❌ 实测匿名抓取被投毒，默认禁用 |
| DuckDuckGo HTML | ❌ 匿名 POST 返回 202 反爬页，默认禁用 |
| 机器之心 RSS | ❌ 官方 RSS 已下线，默认禁用，由搜狗通道覆盖 |

- 本维度与论文维度一样要求 infra 证据词（`requireInfraEvidenceGroups` 含 `wechat`），
  所以 LLM 训练/纯资讯类文章会被范围闸门挡掉；
- 增删大V渠道只需编辑 `wechat.kolChannels` 一行，**无需改代码**；
- 既有 `wechat.curatedAccounts` 保留为“关注账号”说明清单。

**原文直链解析**（`_resolve_sogou_link`）踩过的三个坑：

1. 搜狗返回的 `href` 内嵌换行等**控制字符**，必须先清理，否则请求非法（`InvalidURL`）；
2. 必须使用 **cookie 会话**（先请求搜索页拿到 `SUID`/`SNUID`），否则跳转请求返回反爬页；
3. 跳转页用 `url += '...'` 分片拼接真实地址，且 `&timestamp=` 会被 HTML 实体解码误伤成
   `×tamp=`，需要还原。

实现为：浏览器 UA + cookie 会话 → 逐条解析 → 失败则回退到检索结果链接。
搜索页走快照缓存（这样 `--offline` 也能复现公众号维度），只有跳转链接解析走
`use_cache=False`（链接带会话与时效签名，缓存会拿到过期结果）。
配置项 `wechat.primary.politeDelaySeconds`（控制请求间隔）。

### 维度 E · 周边团队工作

「周边团队」= 芯片厂商 / 互联网厂商 / 有影响力的初创公司 / 周边推理框架。
本维度与维度 C（仓库更新细则）**分开呈现**，分两块：

- **E.1 周边引擎与团队版本特性**：`peers.engines` 里每个引擎列出最近几个版本的官方特性说明
  （版本号 + 日期 + 这版做了什么）。卡片只显示**已有中文说明**的版本，避免英文混排；
  在窗口内发版的条目不进 `pending_zh` 清单，随条目一起补中文即可。
- **E.2 团队动态雷达**：用 `peers.roster` 名录（NVIDIA（含 Sol-Engine）/ AMD / Intel / 华为昇腾 /
  寒武纪 / 摩尔线程 / 燧原 / 壁仞 / 沐曦 / 昆仑芯 / 字节 / 阿里 / 腾讯 / 百度 / 快手 /
  生数科技 Vidu / 爱诗科技 PixVerse / MiniMax / 智谱 / 月之暗面 / 无问芯穹 / 潞晨 / 硅基流动 / 阶跃等）
  反查本周语料，输出每个团队「本期命中 / 本周新增 / 代表条目」——谁这周有动静一目了然。
  增删团队或别名只改 `peers.roster`。

> 原先的「竞品发版特性说明」与「MindIE / 昇腾竞争力」两章已按需求移除：
> 前者改名为周边团队工作并与仓库更新分离，后者（含能力覆盖探针）整章删除，由使用者自行判断。

---

## 5. 代码结构

```
scripts/report/
├── run_weekly.py          CLI 入口：采集 → 归一化 → 渲染
├── collect.py             采集层：各源适配器 + 公众号/搜狗结果解析
├── corpus.py              语料层：去重、中文过滤、排序、主题聚类、周边团队矩阵
├── render.py              渲染层：自包含 HTML 报告
├── zh.py                  中文说明层：按 stableId / repo+tag 套用中文，并标记 zhCurated
├── selftest.py            离线自检（含调研范围、中文约束、匹配器回归）
├── scope_audit.py         调研范围审计：列出被 scope 拦下的条目与命中词（调边界用）
├── config/
│   ├── sources.json           信息源注册表（改这里就能增删源）
│   ├── curated_zh.json        人工撰写的中文标题与说明（按 stableId 匹配）
│   └── curated_releases_zh.json 人工撰写的中文发版说明（按 repo + tag 匹配）
└── lib/
    ├── httpclient.py      HTTP + 快照缓存 + 重试 + cookie 会话
    ├── feed.py            RSS 2.0 / Atom 1.0 解析（替代 feedparser）
    ├── relevance.py       关键词匹配器 + 入库闸门（ScopeGate）、跨源去重
    └── util.py            时间/文本/ID 工具 + 说明文字抽取
```

### 4.1 中文说明层（重要）

报告的标题与说明**以中文为准**，由 `config/curated_zh.json`（条目）与
`config/curated_releases_zh.json`（发版，按 repo+tag）提供，均由人工/代理撰写
（不调用任何翻译服务、不依赖运行时 LLM）：

- 按 `stableId` 精确匹配，命中即覆盖标题与说明，并标记 `zhCurated=True`；
- **只有 title 与 digest 都齐备才算命中**：只补一半的条目仍算待补，不会混进正文；
- `report.chineseOnly=true` 时，未命中的条目**不进正文**，只写进
  `.cache/pending_zh.json` 待补清单，并在附录显示本轮待补条数；
- 中文标题会同时保留原文标题（`originalTitle` / `originalTitleKey`），
  以确保跨源去重仍按原文标题指纹进行（否则中英标题会变成两条）；
- 维度 E 的版本卡片优先用「该 repo+tag 的条目中文说明」，其次 `curated_releases_zh.json`；
  两者都没有的版本卡片不显示（保证全中文）。

> 设计取舍：说明文字**不由脚本自动翻译**（无 LLM、无翻译 API 调用），
> 因此质量与术语可控，且同一份快照重跑结果完全一致。撰写可以在 agent 会话里批量完成，
> 但产物是一份可 review、可 diff 的 JSON。

### 4.2 处理链路

```
信息源 (config) → 抓取 + 快照落盘 → 归一化条目 + 抽取说明 → 主题闸门 → 范围闸门（排除法 + infra 证据）
   → 跨源去重 → 套用中文说明层 → 中文约束过滤 → 分类聚合（维度 A–E）→ HTML 渲染
```

**范围闸门**（见第 2 节）：先按 `scope.exclude` 排掉非 infra 议题，再按 `scope.infraEvidence`
要求论文/公众号/博客类条目给出「系统工程 / 降本增效」证据；被拦条目数量在报告附录 F 与运行日志中可见。

**不打分**：系统没有打分模型、没有分数阈值，也没有 `--min-score`。
条目顺序一律「新增优先 → 发布时间倒序」。

**中文约束**：正文只保留 `zhCurated=True` 的条目（详见 4.1）。

**去重策略**：先按 `stableId`（源内唯一）合并，再按规范化标题指纹跨源合并同一事件。

**新增标记**：发布时间落在最近 7 天内记为 `NEW`，否则 `SEEN`。

---

## 6. weekly 工作流建议

每周固定执行（**顺序很重要**：先联网采集 → 补中文 → 离线渲染）：

```bash
cd AscendPlayground
python scripts/report/run_weekly.py --refresh --max-age-days 8   # 覆盖上周以来（跨周期必须 --refresh）
# 按 .cache/pending_zh.json 把中文写进 config/curated_zh.json
python scripts/report/run_weekly.py --offline                     # 产出全中文报告
python scripts/build_site.py                                      # 更新站点（含周报分区）
```

调研时的实用技巧：

- **觉得范围收得不合适**：先跑 `python scripts/report/scope_audit.py --offline`，按拦截清单调
  `config/sources.json` 的 `scope`，再出报告（第 2 节）。
- **待补中文太多**：可以先把清单按维度分批补；未补的条目只是不出现在正文，不影响其余内容。
- **某一维度为空**：看报告附录 F 的源状态；若显示 403/429 即为限流，非代码问题。
- **离线交付/复现**：把 `scripts/report/.cache/` 一起带上，对方执行 `--offline` 即可得到一致结果。
- **新增公众号关键词 / 大V渠道**：编辑 `wechat.searches` / `wechat.kolChannels`。
- **新增监控仓库**：编辑 `repos.watch`（只加多模态仓库）；无 Atom feed 的仓库加 `"enabled": false`。
- **新增周边团队或别名**：编辑 `peers.engines`（版本特性）与 `peers.roster`（动态雷达）。

---

## 7. 已知限制与诚实说明

1. **GitHub 通道**：默认 Atom feed 已无限流问题；若切回 `channel: "api"`，匿名限额 60 次/小时，
   19 个仓库 + 9 个周边引擎单轮无法全覆盖，系统会明确报告未覆盖的仓库而非静默丢数据。
2. **公众号维度**：搜索页已走快照缓存，`--offline` 可以复现条目；但**原文直链**依赖搜狗会话
   cookie 且带时效签名，离线时会回退到检索结果链接（其余维度均可完整离线复现）。
   搜索页快照没有过期时间，所以跨周期采集必须 `--refresh`。
3. **搜狗限流**：连续快速请求会触发人机校验。系统会检测校验页并在连续 2 次后
   提前结束该维度（避免无效请求与加重限流），并在报告附录中标注。遇到这种情况
   等十几分钟重跑（加 `--refresh` 只重抓缺失的部分即可）不会浪费已有快照。
4. **搜索引擎**：Bing / DuckDuckGo 匿名抓取已不可用，适配器保留但默认禁用。
4. **机器之心**：官方 RSS 已下线；其内容通过搜狗微信通道按关键词覆盖，非全量订阅。
5. **OpenAlex** 在负载高时会返回 429，系统重试 3 次后降级，不影响其它源。
6. **中文媒体 RSS** 字段质量不一：部分源不提供发布时间，这类条目不会被标记为 `NEW`。
7. **说明文字是抽取而非生成**：未补中文时 `digest` 从原始正文抽取要点（无 LLM 参与），
   因此完全确定、可复现；但**这些条目不会进正文**（中文约束），只出现在待补清单里。
   正文里的说明一律来自 `config/curated_zh.json`。
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
