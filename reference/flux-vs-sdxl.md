---
type: reference
tags:
  - foundations
  - flux
  - sdxl
  - models
date: 2026-06-29
updated: 2026-06-29
---

# Flux vs SDXL — Model Comparison

Reference page comparing **FLUX.1** (Black Forest Labs, 2024) with **SDXL** (Stability AI, 2023). Flux is the primary model for this sprint; SDXL is what most existing ComfyUI workflows assume.

Some entries below are explicit in the model cards / Black Forest Labs GitHub. Others are "practitioner knowledge" (widely-cited Flux config details that aren't in the official card) — those are marked with †.

## High-level

| | FLUX.1 [dev] | SDXL Base 1.0 |
|--|--------------|---------------|
| Released | August 2024 | July 2023 |
| Parameters | 12B | 3.5B (base) + 6.6B (refiner) |
| Architecture | [[dit\|DiT]]-family (MMDiT†) | [[u-net\|U-Net]] |
| VAE channels | 16† | 4 |
| Text encoder | T5-XXL + CLIP-L† | CLIP-L + OpenCLIP-bigG |
| Training framing | Rectified flow | DDPM ε-prediction |
| Distillation | Guidance-distilled | None (refiner is a second model) |
| Recommended CFG | 3.5 | 7-8 |
| Recommended steps | 50 (dev), 4 (schnell) | 25-40 |
| License | FLUX.1-dev Non-Commercial (commercial output OK) | OpenRAIL++ |

## What makes Flux different in practice

### Bigger and more capable text encoder

Flux uses T5-XXL alongside CLIP-L. T5 is a much larger encoder than the CLIP variants SDXL uses, and is better at long, descriptive prompts. This is why Flux follows complex prompts much more reliably — the conditioning signal carries more information per prompt token.

### 16-channel VAE — text and fine detail actually work

SDXL's 4-channel VAE was famously bad at rendering text in images. Flux's 16-channel VAE has 4× the information per spatial location, which is the single biggest reason Flux renders text legibly where SDXL produces gibberish. See [[latent-diffusion]] for why VAE channel count is load-bearing.

### MMDiT instead of U-Net

Flux uses a **Multi-Modal Diffusion Transformer** — image tokens and text tokens share attention layers, instead of cross-attention from image-only attention. This was introduced by Stable Diffusion 3 and adopted/extended by Flux. The result: cleaner gradient flow between text and image, better prompt adherence.

See [[dit]] for the architecture story and [[papers/dit]] for the original scaling result that made transformers the default backbone.

### Rectified flow training

Flux trains in the **rectified flow** framework instead of standard DDPM ε-prediction. Conceptually similar (both learn to map noise to data via iterative refinement), but rectified flow learns straight-line trajectories through latent space, which is meant to enable fewer-step sampling. This is part of why FLUX.1 schnell can sample in 4 steps.

Not yet ingested into the wiki — would be a worthwhile future read for understanding modern fast-sampling models.

### Guidance distillation

FLUX.1-dev is **distilled** to produce high-quality outputs at lower CFG (~3.5) than SDXL needs (~7). This means:
- Fewer extra forward passes per step (CFG always costs 2× compute per step)
- Less risk of over-saturation / "burnt" artifacts at high CFG
- See [[classifier-free-guidance]] and [[papers/cfg]] for the underlying CFG dynamics

FLUX.1 schnell goes further — it's distilled to need no CFG at all, sampled in 4 steps.

## The FLUX.1 family (from official GitHub)

| Variant | License | Use |
|---------|---------|-----|
| FLUX.1 pro | Closed API only (bfl.ml) | Highest quality, paid only |
| FLUX.1 dev | Non-commercial | Open weights, dev-friendly, base of most LoRAs |
| FLUX.1 schnell | Apache-2.0 | Fully open, 4-step distilled, fastest |
| FLUX.1 Fill [dev] | Non-commercial | Inpainting / outpainting |
| FLUX.1 Canny [dev] | Non-commercial | Structural conditioning via Canny edges |
| FLUX.1 Depth [dev] | Non-commercial | Structural conditioning via depth maps |
| FLUX.1 Redux [dev] | Non-commercial | Image variation |
| FLUX.1 Kontext [dev] | Non-commercial | Image editing |
| FLUX.1 Krea [dev] | Non-commercial | Text-to-image variant |

For this sprint, the relevant variants are **FLUX.1 dev** (the base for LoRA training and most workflows) and **FLUX.1 schnell** (if you need fast iteration on Apache-2.0).

## License nuance — important for the sprint

- **FLUX.1-dev's "Non-Commercial License"** restricts how you use the *weights*. You can use it for research and personal projects.
- **Outputs from FLUX.1-dev can be used commercially** under the Non-Commercial License's terms.
- **FLUX.1 schnell is Apache-2.0** — fully permissive, both weights and outputs.

If you're training a LoRA you intend to sell or distribute commercially as a Flux LoRA, the base-model license restriction applies. If you're distributing your own checkpoints/derivatives, read the actual LICENSE.md text — there are conditions. For pure "use Flux to generate images for a client project," outputs are unencumbered.

The Apache-2.0 schnell is the safer base for commercial work where the licensing posture matters.

## Recommended inference settings (from HF model card)

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # optional, saves VRAM

image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
```

| Setting | Value | Notes |
|---------|-------|-------|
| dtype | `bfloat16` | fp16 also works; fp32 wastes memory |
| Resolution | 1024×1024 baseline | Flux handles non-square reasonably well; weird aspect ratios degrade |
| Guidance scale | 3.5 | Distilled — higher values over-saturate faster than SDXL |
| Steps | 50 (dev) / 4 (schnell) | More than 50 rarely helps on dev |
| Max sequence length | 512 | T5 token budget for the prompt |

## VRAM requirements (practitioner knowledge†)

| Setup | Comfortable on |
|-------|----------------|
| Full bf16 inference | 24GB (RTX 4090, A6000) |
| With CPU offload | 16GB |
| With NF4 quantization | 10-12GB |
| For LoRA training (rank 16) | 24GB minimum, 48GB comfortable |

LoRA training at 1024×1024 on Flux can spike to 30GB+; some trainers (AI-Toolkit) use gradient checkpointing and lower-precision optimizers to fit in 24GB.

## See Also

- [[dit]]
- [[papers/dit]]
- [[latent-diffusion]]
- [[u-net]]
- [[classifier-free-guidance]]
- [[lora]]
- [[fine-tuning]]
- [[foundations]]
