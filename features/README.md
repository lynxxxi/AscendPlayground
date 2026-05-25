# Features Knowledge Base

Ascend platform key features and capabilities knowledge index.

## Contents

- [Distributed Training & Inference](./distributed.md)
- [Speculative Decoding](./speculative_decoding.md)
- [Inference Acceleration & Caching](./inference_acceleration.md)
- [Hardware Features](./hardware_features.md)

## Feature Overview

| Feature | Description | Frameworks |
|---------|-------------|------------|
| Distributed Inference | TP/PP/DP multi-card parallelism | MindIE, vLLM-Ascend |
| Speculative Decoding | EAGLE/DFLASH/N-gram accelerated decoding | vLLM-Ascend |
| Inference Caching | DiT feature caching, skip redundant steps | cache-dit |
| Quantized Inference | Multiple quantization schemes for memory & speed | MindIE, vLLM-Ascend |
| Multi-modal Inference | Unified image/video/audio inference | vLLM-Omni |
| ACL Graph | Compute graph optimization, reduce Host-Device interaction | vLLM-Ascend |
