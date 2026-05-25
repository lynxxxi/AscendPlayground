# Operator Optimization

## Optimization Methodology

### 1. Compute-Bound Operator Optimization

- **Double Buffer**: Overlap compute and data transfer by async copying next tile while computing current tile
- **Multi-Core Parallelism**: Design tiling strategy to fully utilize all AI Cores
- **Instruction Pipeline**: Leverage Vector and Cube parallelism

### 2. Memory-Bound Operator Optimization

- **Data Reuse**: Reduce GM access, maximize Local memory reuse
- **Memory Alignment**: Ensure data address and length meet alignment requirements
- **Coalesced Access**: Consecutive address access for better bandwidth utilization

### 3. Communication-Bound Operator Optimization

- **Compute-Communication Overlap**: Use stream programming for parallel comm and compute
- **Gradient Aggregation Optimization**: AllReduce fusion, gradient compression

## Profiling Tools

| Tool | Usage |
|------|-------|
| msprof | NPU performance collection and analysis |
| torch_npu.profiler | PyTorch-level performance analysis |
| AscendCL Profiling | Low-level operator profiling |

## References

- [catlass optimization practices](../../catlass/docs/zh/1_Practice/11_matmul_optimization.md)
- [ops-nn operator library](../../ops-nn)
