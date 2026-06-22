# Diffusion Studio 🎨

Training and inference toolkit for diffusion models with LoRA fine-tuning, INT8 quantization, and real-time generation.

## Features

- **LoRA Training**: Fine-tune Stable Diffusion with minimal VRAM (4GB)
- **INT8 Quantization**: 75% memory reduction with <2% quality loss
- **Real-time Generation**: 512x512 in 0.3s on A100
- **ControlNet Integration**: Pose, depth, canny, segment control
- **Batch Pipeline**: Process 1000+ images/hour
- **ONNX Export**: Deploy to any hardware

## Benchmarks

| Model | Resolution | A100 | MI300X | RTX 4090 |
|-------|-----------|------|--------|----------|
| SD 1.5 | 512×512 | 0.31s | 0.28s | 0.35s |
| SDXL | 1024×1024 | 1.2s | 1.0s | 1.5s |
| Flux-schnell | 1024×1024 | 0.8s | 0.6s | 1.1s |

## Quick Start

```bash
pip install diffusion-studio
```

```python
from diffusion_studio import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
image = pipe.generate("A sunset over mountains, oil painting", steps=4)
image.save("output.png")
```

## License

Apache 2.0