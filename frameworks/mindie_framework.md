# MindIE Inference Framework

## Overview

MindIE (Mind Inference Engine) is the Ascend native inference engine, providing high-performance inference for large language models and text-to-image models.

## Sub-Modules

### MindIE-LLM

Large language model inference engine, core features:

- **Continuous Batching**: Dynamic batching for improved throughput
- **PagedAttention**: Paged KV Cache memory management for larger cache capacity
- **Multi-modal Support**: Vision-language model inference
- **Quantized Inference**: W8A8, W4A16 and more quantization schemes

### MindIE-SD

Stable Diffusion inference engine:

- **Text-to-Image / Image-to-Image**: SD1.5, SDXL, SD3 support
- **LoRA Acceleration**: Hot-loading LoRA weights
- **Distributed Inference**: Multi-card distributed inference

### MindIE-CI

Continuous integration and build system:

- Build scripts and compile configs
- DT (Development Toolkit) integration
- Release packaging pipeline

## Quick Start

```bash
# MindIE-LLM launch example
mindieservice --config config.conf
```

## References

- [MindIE-LLM](../../MindIE-LLM)
- [MindIE-SD](../../MindIE-SD_bugfix)
- [MindIE-CI](../../MindIE-CI)
