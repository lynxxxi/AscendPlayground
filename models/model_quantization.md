# Model Quantization

## Quantization Schemes Overview

| Scheme | Weight Precision | Activation Precision | Use Case | Framework Support |
|--------|-----------------|---------------------|----------|-------------------|
| W8A8 | INT8 | INT8 | Balanced accuracy & perf | vLLM-Ascend, MindIE |
| W4A16 | INT4 | FP16/BF16 | Memory priority | vLLM-Ascend, MindIE |
| W4A8 | INT4 | INT8 | Max compression | vLLM-Ascend |
| W4A4 MXFP4 | MXFP4 | MXFP4 | Experimental | vLLM-Ascend |
| W8A8 MXFP8 | MXFP8 | MXFP8 | Experimental | vLLM-Ascend |
| W4A4 LaOS | INT4 | INT4 | Dynamic quantization | vLLM-Ascend |

## Quantization Workflow

### 1. Calibration Data Preparation

```python
# Prepare calibration dataset
calib_data = load_calibration_dataset()
```

### 2. Quantize Model

```bash
# Quantize with modelslim
modelslim quantize --model <model_path> --method w8a8 --calib-data <calib_path>
```

### 3. Deploy Inference

```bash
# vLLM-Ascend quantized model inference
vllm serve <model_path> --quantization w8a8 --device npu
```

## Accuracy Impact

| Scheme | Typical Accuracy Loss | Applicable Model Size |
|--------|----------------------|-----------------------|
| W8A8 | < 0.5% | 7B+ |
| W4A16 | < 1% | 13B+ |
| W4A8 | 1-3% | 70B+ |
| MXFP4 | 2-5% | Experimental |

## References

- [vllm-ascend quantization](../../vllm-ascend/vllm_ascend/quantization)
- [ops-transformer quantized operators](../../ops-transformer)
