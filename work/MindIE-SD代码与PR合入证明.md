# MindIE-SD 代码与 PR 合入证明

证据来源为 GitCode Ascend/MindIE-SD 合入提交。仓库原生 no-merge-commit 信息包含 PR 编号、Created-by、Commit-by、Co-authored-by、Merged-by、Purpose、Test Plan 与 Test Report。

| PR | 主题 | 合入提交 | 日期 | 规模/说明 | 归属证据 |
|---|---|---|---|---|---|
| !313 | MXFP4 QFA 与 Metadata 算子 | 6c9f080 | 2026-06-03 | 87 文件，+17711/-335 | weixin_44144262 / lijinxi2@huawei.com |
| !347 | MXFP4 Softmax 布局变体 | 0c5670c | 2026-06-11 | 14 文件，+913/-165 | weixin_44144262 / lijinxi2@huawei.com |
| !365 | 部署端 MXFP4 FA 接入 | f1b15b4 | 2026-06-15 | 7 文件，+857/-332 | weixin_44144262 / lijinxi2@huawei.com |
| !578 | USP 重叠与通信量化 | 56a1400 | 2026-09-03 | usp.py 新增 904 行 | weixin_44144262 / lijinxi2@huawei.com |
| !580 | USP 指南与测试 | 6dc71e5 | 2026-09-03 | 5 文件，新增 832 行 | weixin_44144262 / lijinxi2@huawei.com |
| !217 | Wan2.2 同步与精度修复 | a8b6277 | 2026-03-21 | 已合入 | lijinxi2@huawei.com |
| !254 | PEP517 版本覆盖修复 | 68c4d05 | 2026-04-17 | 已合入 | lijinxi2@huawei.com |
| !296 | Offload 幂等性与事件顺序修复 | fe5f819 | 2026-05-19 | 2 文件，+27/-11 | Co-authored lijinxi2@huawei.com |
| !70 | Layer 级 CPU Offload | 7417137 | 2026-01-17 | 2 文件，新增 330 行 | Created-by weixin_44144262 |
| !71 | 多实例共享显存 | aa21204 | 2026-01-17 | 3 文件，+427/-1 | Created-by weixin_44144262 |

## 核验入口

- https://gitcode.com/Ascend/MindIE-SD/merge_requests/313
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/347
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/365
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/578
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/580
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/217
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/254
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/296
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/70
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/71

平台最终 Merged-by 通常为 ascend-robot。个人归属结合 PR 创建账号、Commit-by/Co-authored-by、提交邮箱及变更内容判断。

