# Large Language Models

## Supported Model List

### Dense Models

| Model | Parameters | Highlights |
|-------|-----------|------------|
| LLaMA 3/3.1 | 8B-70B | Meta open-source, wide community support |
| Qwen3 | 0.6B-32B | Alibaba Tongyi, strong Chinese capability |
| GLM-4 | 9B | Zhipu ChatGLM series |

### MoE Models

| Model | Total Params | Active Params | Highlights |
|-------|-------------|---------------|------------|
| DeepSeek-V3 | 671B | 37B | Auxiliary-loss-free load balancing |
| DeepSeek-R1 | 671B | 37B | Reasoning RL |
| Qwen3-MoE | - | - | MoE architecture |

### Mamba Models

| Model | Description |
|-------|-------------|
| Mamba2 | State space model, linear complexity |
| Jamba | Mamba + Transformer hybrid architecture |

## Key Inference Techniques

- **PagedAttention**: Paged KV Cache management, reduces memory fragmentation
- **Continuous Batching**: Dynamic request-level scheduling for higher throughput
- **Speculative Decoding**: Accelerate autoregressive generation with draft models
- **Tensor Parallel**: Multi-card inference for large models

## References

- [vllm-ascend supported models](../../vllm-ascend)
- [MindIE-LLM](../../MindIE-LLM)
