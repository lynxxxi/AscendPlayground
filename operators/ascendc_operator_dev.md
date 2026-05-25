# AscendC Operator Development

## Overview

AscendC is the Ascend operator development language, providing operator-oriented programming APIs that abstract hardware differences and support multiple generations of Ascend processors.

## Development Workflow

1. **Prototype Definition**: Define inputs, outputs, and attributes in `op_host/`
2. **Tiling Implementation**: Implement tiling strategy in `op_host/` to compute data split params
3. **Kernel Implementation**: Implement device-side compute logic with AscendC API in `op_kernel/`
4. **Operator Registration**: Register prototype via `OpReg` macro
5. **Build & Deploy**: Generate project with `msopgen`, build with `bash build.sh`

## Core Data Copy APIs

| API | Description |
|-----|-------------|
| `DataCopy` | Data transfer between GM and Local memory |
| `DataCopyPad` | Data copy with padding |
| `CopyIn` / `CopyOut` | Data transfer via CubeNode |

## Compute APIs

| API | Description |
|-----|-------------|
| `Compute` | Vector compute entry |
| `Matmul` | Matrix multiplication |
| `Softmax` | Softmax computation |

## References

- [catlass operator library](../../catlass) - High-performance matmul operators based on AscendC
- [ops-nn](../../ops-nn) - NN operator implementations
