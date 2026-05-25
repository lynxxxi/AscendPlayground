# Speculative Decoding

## Overview

Speculative Decoding accelerates autoregressive generation by using a small Draft Model to quickly generate candidate tokens, then verifying them in parallel with the large Target Model, maintaining accuracy while improving speed.

## Implementation Schemes

### EAGLE

- **Principle**: Feature-level autoregressive prediction using Target Model hidden states
- **Advantage**: High acceptance rate, 2-3x speedup
- **vLLM-Ascend support**: [eagle_proposer.py](../../vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py)

### DFLASH

- **Principle**: Model-based draft generation strategy
- **vLLM-Ascend support**: [dflash_proposer.py](../../vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py)

### N-gram

- **Principle**: N-gram statistical token prediction, no Draft Model needed
- **Advantage**: Zero extra memory overhead
- **vLLM-Ascend support**: [ngram_proposer_npu.py](../../vllm-ascend/vllm_ascend/spec_decode/ngram_proposer_npu.py)

### LLM Base Proposer

- **Principle**: Use a small LLM as the Draft Model
- **vLLM-Ascend support**: [llm_base_proposer.py](../../vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py)

## Performance Reference

| Scheme | Typical Speedup | Extra Memory | Use Case |
|--------|----------------|-------------|----------|
| EAGLE | 2-3x | Medium | General |
| DFLASH | 1.5-2x | Medium | General |
| N-gram | 1.2-1.5x | Minimal | Repetitive text |
| LLM Base | 1.5-2.5x | Large | With paired small model |

## References

- [vllm-ascend spec_decode](../../vllm-ascend/vllm_ascend/spec_decode)
