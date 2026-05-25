# Hardware Features

## Ascend Processor Series

| Processor | Architecture | AI Core Count | Typical Scenario |
|-----------|-------------|---------------|------------------|
| Ascend 910B | Da Vinci | - | Training & inference |
| Ascend 910_93 | Da Vinci | - | Training & inference |
| Ascend 950 | Da Vinci | - | Inference optimized |
| Atlas A2 | - | - | Training server |
| Atlas A3 | - | - | Training server |

## AI Core Architecture

### Compute Units

| Unit | Function |
|------|----------|
| Cube Unit (AI-Matrix) | Matrix multiplication (Matmul/Conv) |
| Vector Unit | Vector operations (activation, normalization, etc.) |
| Scalar Unit | Scalar operations (control flow, address computation) |

### Memory Hierarchy

| Memory Layer | Capacity | Bandwidth | Description |
|-------------|----------|-----------|-------------|
| L1 (L1-A / L1-B) | ~1MB | High | Cube/Vector dedicated |
| L0 (UB) | ~256KB | Very High | Universal Buffer |
| GM (Global Memory) | Tens of GB | Medium | HBM |

## Software Stack

```
Application   MindIE / vLLM / PyTorch
    ↓
Framework     torch_npu / CANN
    ↓
Runtime       ACL (Ascend Computing Language)
    ↓
Driver        HDK (Hardware Development Kit)
    ↓
Hardware      Ascend NPU
```

## References

- [catlass hardware adaptation](../../catlass) - Atlas A2/A3/Ascend 950 support
- [ops-nn hardware configs](../../ops-nn) - Per-processor operator configs
