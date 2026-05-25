# Inference Acceleration & Caching

## cache-dit: Diffusion Model Inference Caching

### Core Idea

Diffusion model inference requires multiple denoising iterations, and DiT features across adjacent steps are highly similar. cache-dit caches intermediate features to skip redundant compute steps.

### How It Works

```
Step 1: Full compute → Cache features
Step 2: Detect feature change → Reuse cache if change is small
Step 3: Recompute if change is large → Update cache
...
```

### Key Features

| Feature | Description |
|---------|-------------|
| Feature Caching | Cache DiT Block intermediate features |
| Adaptive Strategy | Dynamically decide based on feature change magnitude |
| Distributed Support | Ray cluster distributed inference |
| Multi-Model Support | FLUX, SD3, PixArt and other DiT models |

### Usage

```python
from cache_dit import cache_forward

# Apply caching to DiT model
model = cache_forward(model, cache_threshold=0.1)
```

## Other Acceleration Techniques

### ACL Graph

- **Principle**: Compile Host-side operator dispatch logic into a fixed graph, reducing Host-Device interaction overhead
- **Use Case**: Fixed-shape inference requests
- **vLLM-Ascend support**: ACL Graph capture and replay

### PagedAttention

- **Principle**: Paged KV Cache management, similar to OS virtual memory
- **Advantage**: Eliminates memory fragmentation, supports larger batches
- **Implementation**: vLLM-Ascend / MindIE-LLM

### Continuous Batching

- **Principle**: Request-level dynamic scheduling, completed requests exit immediately, new requests enter
- **Advantage**: Significantly higher throughput vs static batching

## References

- [cache-dit](../../cache-dit)
- [vllm-ascend](../../vllm-ascend)
