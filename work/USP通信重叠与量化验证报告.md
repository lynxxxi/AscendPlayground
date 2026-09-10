# USP 通信重叠与量化验证报告

## 1. 验证范围

覆盖 `usp_attention` 的基础注意力、Ulysses 分片、KV 聚合、通信量化、Head Chunk 重叠、Quant FA 结果复用、异常校验，以及 HunyuanVideo 4/8 卡端到端链路。

## 2. 测试设计

| 类别 | 测试内容 | 判定原则 |
|---|---|---|
| 功能 | 单 Rank 与多 Rank 基础路径 | 输出 shape、dtype 与参考路径一致 |
| 分片 | Head/Sequence 交换与恢复 | 分片前后语义一致 |
| 量化 | Q/K/V/Out Q/DQ 与 Scale | 误差满足项目门限，无 shape/dtype 错误 |
| 重叠 | 多 Stream、Event、Head Chunk | 无竞态、死锁，结果稳定 |
| 复用 | Quant FA 中间量化结果复用 | 避免重复量化且结果一致 |
| 异常 | 非法拓扑、维度和参数 | 明确报错并阻止错误执行 |
| E2E | HunyuanVideo 4/8 卡 | 模型可运行，精度与性能满足 PR 验证口径 |

## 3. 已有结果

- PR !578 于 2026-09-03 合入，`mindiesd/layers/usp.py` 新增 904 行核心实现。
- PR !580 同日合入，中英文指南与 `tests/layers/test_usp.py` 等共新增 832 行。
- 两个 PR 的 Test Plan 均记录模型 E2E 8 卡、4 卡测试，并提供 HunyuanVideo 多卡测试截图。
- 合入提交作者/提交者信息包含 `weixin_44144262` / `lijinxi2@huawei.com`，最终由机器人完成合入。

## 4. 结论及证据边界

现有合入记录证明核心代码、测试和文档已进入 MindIE-SD 主仓，形成“设计—开发—测试—文档”的完整交付。PR 给出的“小于 5%”是典型多模态模型的目标/场景结果，不能脱离硬件拓扑、模型、序列长度和量化配置泛化。正式材料建议附内部性能平台原始数据，以便评委核对基线、环境和统计方法。

## 5. 证据链接

- https://gitcode.com/Ascend/MindIE-SD/merge_requests/578
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/580
- commits：`56a1400`、`6dc71e5`

