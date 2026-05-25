# Operator Debug & Verification

## Common Debug Approaches

### 1. Accuracy Verification

- **Comparison**: Element-wise comparison against CPU/GPU reference, compute max error and mean error
- **Data Dump**: Capture intermediate data via `ASCEND_GLOBAL_LOG_LEVEL` and dump config
- **Finite Precision Analysis**: Analyze FP16/BF16/INT8 quantization accuracy loss

### 2. Functional Debugging

- **Operator UT Test**: Run single-operator verification with `oprunner`
- **msdebug**: Ascend debugging tool with breakpoint and step execution support
- **AscendCL Logging**: Control log level via environment variables

### 3. Performance Tuning

- **Profiling**: Collect operator execution time, bandwidth utilization, compute utilization
- **Bottleneck Identification**: Distinguish compute bottleneck vs memory bottleneck

## Common Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Accuracy error | Data type conversion, accumulation order | Use high-precision accumulation, adjust compute order |
| Operator execution failure | Tiling param out of bounds | Check tiling calculation logic |
| Suboptimal performance | Insufficient parallelism, non-contiguous access | Double buffer, memory alignment |

## References

- [catlass debug guide](../../catlass/docs/zh/1_Practice/evaluation/msdebug.md)
