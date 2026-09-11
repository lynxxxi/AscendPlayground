# MXFP4 Quant Flash Attention 验证报告

## 1. 验证对象

验证 MindIE-SD MXFP4 Quant Flash Attention、Metadata 算子、Softmax 布局变体以及模型部署端接入。关联 PR 为 !313、!347、!365。

## 2. 验证策略

采用“单算子 Golden—布局专项—框架集成—模型 E2E”四层验证：单算子层检查数值正确性；布局层覆盖 qs128/kvs32 与 qs128/kvs256；框架层检查量化配置与算子调用；模型层检查 Wan2.2 生成链路的精度和性能。

## 3. 验证项与结果

| 层级 | 验证项 | 预期结果 | 已有结果/证据 |
|---|---|---|---|
| 构建 | QFA 与 Metadata 自定义算子编译 | 构建成功 | PR !313、!347 已合入，远端 Hook 通过 |
| 单算子 | `quant_flash_attn_golden.py` | 输出满足项目精度门限 | PR !313 提供单算子精度测试截图 |
| 布局 | qs128/kvs32 | 对应 Kernel 正确执行 | PR !347 测试计划及本地 UT 记录 |
| 布局 | qs128/kvs256 | 对应 Kernel 正确执行 | PR !347 测试计划及本地 UT 记录 |
| 集成 | 量化配置、权重与算子调用 | 部署路径完整，无接口错误 | PR !365 已合入 |
| E2E | Wan2.2 模型推理 | 精度与性能完成验证 | PR !365 测试报告截图 |

## 4. 回归关注点

1. 不同序列长度、Head 数、Q/KV 分块组合的尾块边界。
2. Scale/Metadata 的 shape、dtype 和生命周期。
3. BF16、FP16 与 MXFP4 分支切换时的参数污染。
4. CANN 版本升级后的编译兼容与 Kernel 选择。
5. 模型长序列、多层误差累积和生成质量。

## 5. 结论

三组 PR 已形成从算子实现、布局扩展到模型部署接入的闭环并成功合入 MindIE-SD。构建、Golden/UT 和 Wan2.2 E2E 验证均已执行。当前归档记录包含功能与精度验证结论；硬件型号、CANN 版本、完整误差分布及性能明细以测试平台原始记录为准。

## 6. 可追溯证据

- https://gitcode.com/Ascend/MindIE-SD/merge_requests/313
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/347
- https://gitcode.com/Ascend/MindIE-SD/merge_requests/365
- commits：`6c9f080`、`0c5670c`、`f1b15b4`
