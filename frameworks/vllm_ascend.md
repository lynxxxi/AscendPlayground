# vLLM-Ascend

## Overview

vLLM-Ascend is the Ascend NPU adaptation of the vLLM inference framework, providing an inference experience compatible with vLLM upstream.

## Architecture

```
vLLM (upstream)
  └── vllm-ascend (NPU adapter)
        ├── vllm_ascend/ops/       # NPU operator implementations
        ├── vllm_ascend/patch/     # vLLM adaptation patches
        ├── vllm_ascend/worker/    # NPU Worker implementation
        ├── vllm_ascend/quantization/ # Quantization schemes
        └── vllm_ascend/spec_decode/  # Speculative decoding
```

## Core Adaptation Layers

| Module | Description |
|--------|-------------|
| `ops/` | NPU operators: Attention, Linear, RMSNorm, etc. |
| `patch/` | Platform adaptation patches for vLLM |
| `worker/` | NPU Worker and ModelRunner |
| `quantization/` | W4A16, W8A8, MXFP4 quantization schemes |
| `spec_decode/` | EAGLE, DFLASH speculative decoding |
| `profiler/` | NPU profiling tools |

## Usage

```bash
# Install
pip install vllm-ascend

# Launch inference service
vllm serve <model_name> --device npu
```

## Comparison with MindIE

| Feature | vLLM-Ascend | MindIE-LLM |
|---------|-------------|------------|
| Upstream compat | Synced with vLLM upstream | Ascend native |
| Ecosystem | vLLM plugin ecosystem | Ascend ecosystem |
| Quantization | W4A16/W8A8/MXFP4 | W8A8/W4A16 |
| Speculative Decoding | EAGLE/DFLASH/N-gram | Supported |

## References

- [vllm-ascend repo](../../vllm-ascend)
- [vllm upstream repo](../../vllm)
