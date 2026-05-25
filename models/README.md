# Models Knowledge Base

Models deployed and optimized on Ascend platform knowledge index.

## Contents

- [Large Language Models](./llm_models.md)
- [Diffusion Models](./diffusion_models.md)
- [Model Quantization](./model_quantization.md)

## Supported Models Overview

| Model Type | Representative Models | Inference Framework | Related Repo |
|------------|----------------------|--------------------|--------------|
| LLM | DeepSeek-V3/R1, Qwen3, LLaMA | MindIE-LLM / vLLM-Ascend | [vllm-ascend](../../vllm-ascend) |
| Multi-modal LLM | Qwen3-VL, InternVL | vLLM-Omni | [vllm-omni](../../vllm-omni) |
| Video Generation | Wan2.2 | - | [Wan2.2](../../Wan2.2) |
| Image Generation | Stable Diffusion / SD3 | MindIE-SD | [MindIE-SD_bugfix](../../MindIE-SD_bugfix) |
| Mamba | Mamba / Jamba | vLLM-Ascend | [vllm-ascend](../../vllm-ascend) |

## Model Deployment Workflow

1. **Model Acquisition**: Download weights from ModelScope / HuggingFace
2. **Format Conversion**: Convert weights to framework-required format
3. **Quantization (Optional)**: W8A8 / W4A16 / MXFP4 quantization
4. **Configure Inference Params**: max_batch_size, kv_cache, etc.
5. **Launch Service**: Start inference service via framework API
