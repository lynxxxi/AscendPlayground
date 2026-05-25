# Frameworks Knowledge Base

Ascend ecosystem inference and training framework knowledge index.

## Contents

- [MindIE Inference Framework](./mindie_framework.md)
- [vLLM-Ascend](./vllm_ascend.md)
- [PyTorch Ascend Adaptation](./pytorch_ascend.md)

## Framework Overview

| Framework | Positioning | Related Repo |
|-----------|-------------|--------------|
| MindIE | Ascend native inference engine, LLM/SD inference | [MindIE-LLM](../../MindIE-LLM), [MindIE-SD_bugfix](../../MindIE-SD_bugfix), [MindIE-CI](../../MindIE-CI) |
| vLLM-Ascend | vLLM Ascend NPU adaptation | [vllm-ascend](../../vllm-ascend) |
| vLLM-Omni | Multi-modal inference framework | [vllm-omni](../../vllm-omni) |
| PyTorch (torch_npu) | PyTorch Ascend adaptation | - |

## Selection Guide

| Scenario | Recommended Framework | Reason |
|----------|----------------------|--------|
| LLM high-throughput inference | MindIE-LLM / vLLM-Ascend | PagedAttention, Continuous Batching |
| SD image generation | MindIE-SD | Optimized for Stable Diffusion |
| Multi-modal inference | vLLM-Omni | Unified multi-modal inference architecture |
| Model development & debug | torch_npu | PyTorch ecosystem compatible |
