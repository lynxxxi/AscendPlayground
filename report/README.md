# 多模态 Infra 情报周报

本目录**只放产物**：每个周期一个 HTML 报告，直接以周次命名（如 `2026-W38.html`），
同一周重复运行会覆盖同一文件。直接双击即可打开（自包含 HTML，无外部依赖）。

## 报告覆盖六个维度

| 维度 | 内容 | 主要信息源 |
|------|------|-----------|
| **A** | 多模态 Infra 论文 | arXiv、HuggingFace Daily Papers、OpenAlex |
| **B** | 主要多模态技术团队工作 | 团队官方博客、HuggingFace 模型发布雷达、中文技术媒体 |
| **C** | 主要多模态仓库更新细则 | 27 个仓库的 release / 关键 commit（GitHub Atom feed） |
| **D** | 公众号与中文媒体 | 搜狗微信检索、量子位 / 雷峰网 / 钛媒体 RSS |
| **E** | 竞品发版与工程节奏 | vLLM、SGLang、M\*、TensorRT-LLM、Dynamo vs 昇腾 |
| **F** | MindIE / 昇腾竞争力 | 自有仓库活跃度 + 8 个能力覆盖探针 |

## 目录结构

```
report/
├── README.md               本文件
└── <周>.html                当周可视化报告（如 2026-W38.html）
```

采集缓存不在本目录，放在 `scripts/report/.cache/snapshots/`，可随时删除。

## 生成

```bash
cd AscendPlayground

python scripts/report/run_weekly.py              # 联网采集并生成当周报告
python scripts/report/run_weekly.py --offline    # 离线复现（只读缓存，不联网）
python scripts/report/selftest.py                # 离线自检（78 项断言）

python scripts/build_site.py                     # 更新站点（含本分区）
```

详细参数、打分模型、信息源实测状态与已知限制见
[`scripts/report/README.md`](../scripts/report/README.md)。

## 阅读约定

- **NEW**：发布时间落在最近 7 天内（本周新增）。
- **SEEN**：更早的条目，仍在采集窗口内。
- **FOCUS**（竞品表）：昇腾 / MindIE 侧跟踪目标。
- 顶部搜索框可跨全部维度实时过滤（标题、摘要、关键词、来源均可检索）。

> 本目录内容由脚本自动生成，请勿手工编辑；如需调整内容，请修改
> `scripts/report/config/sources.json` 后重新运行。
