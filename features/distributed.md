# Distributed Training & Inference

## Parallelism Strategies

### Tensor Parallelism (TP)

Split model weights across multiple NPUs by dimension for parallel computation:

- **Column Parallel**: Split weights by column, applicable to Linear layers
- **Row Parallel**: Split weights by row, applicable to Linear layers
- **Communication Primitives**: AllReduce (forward), AllReduce (backward)

### Pipeline Parallelism (PP)

Split model by layers across different NPUs to form a pipeline:

- **Micro-batches**: Split batch into micro-batches to fill the pipeline
- **1F1B Schedule**: One forward one backward, reduces memory footprint

### Data Parallelism (DP)

Replicate the same model across multiple NPUs with different data:

- **AllReduce**: Gradient aggregation
- **ZeRO Optimization**: Optimizer state, gradient, parameter sharding

### Sequence Parallelism (SP)

Split long sequences by dimension, applicable to ultra-long contexts:

- Ring Attention implementation
- [Wan2.2 sequence parallelism](../../Wan2.2/wan/distributed/sequence_parallel.py)

## Communication Backend

| Backend | Description |
|---------|-------------|
| HCCL | Ascend Collective Communication Library, similar to NCCL |
| torch.distributed | PyTorch distributed interface |

## References

- [vllm-ascend distributed inference](../../vllm-ascend)
- [Wan2.2 distributed](../../Wan2.2)
