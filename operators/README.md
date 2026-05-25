# Operators Knowledge Base

Ascend operator development and optimization knowledge index.

## Contents

- [AscendC Operator Development](./ascendc_operator_dev.md)
- [Operator Optimization](./operator_optimization.md)
- [Operator Debug & Verification](./operator_debug.md)

## Operator Categories

| Category | Description | Related Repo |
|----------|-------------|--------------|
| NN Operators | Conv, Pooling, Norm, RNN, etc. | [ops-nn](../../ops-nn) |
| Transformer Operators | Flash Attention, MLA, Quantized ops | [ops-transformer](../../ops-transformer) |
| Matmul Operators | GEMM, SVD-Quant Matmul, etc. | [catlass](../../catlass) |
| Custom Operators | AscendC/Tik custom operators | - |

## Key Concepts

- **Host vs Device**: Operator dev splits into Host side (registration, tiling) and Device side (kernel impl)
- **Tiling**: Split large tasks into small tiles dispatched to multiple AI Cores in parallel
- **Operator Prototype Registration**: Register input/output/attribute info via `OpReg`
