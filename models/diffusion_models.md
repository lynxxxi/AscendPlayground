# Diffusion Models

## Supported Models

### Image Generation

| Model | Architecture | Resolution | Inference Framework |
|-------|-------------|------------|---------------------|
| SD 1.5 | UNet | 512x512 | MindIE-SD |
| SDXL | UNet | 1024x1024 | MindIE-SD |
| SD3 | DiT | 1024x1024 | MindIE-SD |
| FLUX | DiT | 1024x1024 | - |

### Video Generation

| Model | Architecture | Description | Related Repo |
|-------|-------------|-------------|--------------|
| Wan2.2 | DiT | Text-to-Video / Image-to-Video | [Wan2.2](../../Wan2.2) |

## Inference Optimization

### cache-dit Acceleration

[cache-dit](../../cache-dit) provides diffusion model inference cache optimization:

- **Feature Caching**: Cache DiT intermediate features, skip redundant compute steps
- **Adaptive Caching**: Dynamically decide whether to cache based on feature change magnitude
- **Distributed Inference**: Multi-NPU parallel inference via Ray

### Other Optimizations

- **VAE Tiling**: Tile-based decoding for high-resolution images, reduces memory
- **CFG Guidance**: Classifier-Free Guidance for generation quality control
- **LoRA**: Lightweight style/theme fine-tuning

## References

- [MindIE-SD](../../MindIE-SD_bugfix)
- [cache-dit](../../cache-dit)
- [Wan2.2](../../Wan2.2)
