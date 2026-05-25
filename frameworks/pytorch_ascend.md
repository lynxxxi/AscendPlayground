# PyTorch Ascend Adaptation

## Overview

The `torch_npu` plugin enables PyTorch to run on Ascend NPUs, providing a CUDA-equivalent programming experience.

## Core Components

| Component | Description |
|-----------|-------------|
| torch_npu | PyTorch Ascend backend, similar to torch.cuda |
| torch_npu.profiler | NPU profiling tool |
| apex Ascend adaptation | Mixed precision training adaptation |
| torch.compile | Graph mode compilation optimization |

## Basic Usage

```python
import torch
import torch_npu

# Device specification
device = torch.device("npu:0")
tensor = torch.randn(2, 3).npu()

# Model migration
model = model.npu()
```

## Common Adaptation Issues

| Issue | Solution |
|-------|----------|
| Unsupported operator | Use custom operator or replacement impl |
| Accuracy difference | Check data type, BF16 preferred over FP16 |
| OOM | Enable memory optimization, gradient checkpointing |
| Multi-card training | Use torch.distributed + hccl |

## References

- [ops-nn](../../ops-nn) - PyTorch extension operators
- [ops-transformer](../../ops-transformer) - Transformer operator extensions
