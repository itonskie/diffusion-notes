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

Two text-to-image models from different generations. **SDXL** (Stability AI, 2023) is what nearly every existing ComfyUI workflow on the internet assumes you're using. **FLUX.1** (Black Forest Labs, 2024) is what came after — bigger, smarter about text in images, much better at following long prompts. This sprint trains on Flux, but you'll constantly bump into SDXL workflows and need to know what's different.

The short version of why you care: Flux gets text right and follows prompts. SDXL gets text wrong and needs a lot of negative-prompt babysitting. Flux is also 3–4× bigger, which costs VRAM.

Some entries below are explicit in the official model cards / Black Forest Labs GitHub. Others are **practitioner knowledge** — widely-known config details from people actually using Flux, but not stated in the official card. Those are marked with `†`.

## High-level

A few terms you'll see in the table below, decoded once:

- **Parameters** — total count of trainable numbers in the model. More = more capacity, more VRAM.
- **[[dit|DiT]]** — Diffusion Transformer. A transformer (the same architecture family as GPT) used as the model backbone, instead of the older convolutional [[u-net|U-Net]].
- **MMDiT** — Multi-Modal Diffusion Transformer. A DiT variant where text tokens and image tokens share the same attention layers.
- **VAE** — Variational Autoencoder. Compresses a full-res image into a small "latent" grid the diffusion model actually operates on. Channel count = how much information per pixel of that grid.
- **Text encoder** — turns your prompt into vectors the diffusion model can condition on. CLIP and T5 are two families; T5 is bigger and better at long sentences.
- **Rectified flow** — a newer training recipe where the model learns straight-line paths from noise to image, which lets it sample in fewer steps.
- **DDPM ε-prediction** — the classic training recipe: the model predicts the noise that was added at each step.
- **Distillation** — taking a slow, high-quality model and training a smaller/faster student model to mimic it.
- **CFG** — classifier-free guidance. A knob that pushes the model to follow the prompt harder. Higher = more obedient but also more burnt-looking. See [[classifier-free-guidance]].

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

### Bigger and smarter text encoder

The text encoder is the thing that reads your prompt and hands the diffusion model a numerical summary. SDXL uses two CLIP variants — CLIP was built for matching short captions to images, so it tops out somewhere around "a red cat on a chair." Flux uses **T5-XXL** (a much larger encoder originally built for text-to-text tasks like translation) *alongside* CLIP-L. T5 swallows long descriptive prompts and preserves the structure of what you wrote. This is the single biggest reason Flux follows complex prompts much more reliably — the conditioning signal carries more information per prompt token.

### 16-channel VAE — text and fine detail actually work

The VAE is the compressor that turns a 1024×1024 RGB image into a small grid (roughly 128×128) of "latent" vectors. The diffusion model never sees the real image — it works entirely on that grid. The **channel count** is how many numbers per cell of that grid.

SDXL's VAE has 4 channels per cell. Flux's has 16. That's 4× more information per spatial location, which is the single biggest reason Flux renders text legibly where SDXL produces gibberish. Letters require precise high-frequency detail in a tiny number of pixels, and 4 channels just don't have the bandwidth. See [[latent-diffusion]] for why VAE channel count is load-bearing.

### MMDiT instead of U-Net

The old way (U-Net): image data flows through one stack of layers, text gets *injected* into it via "cross-attention" — image tokens look at text tokens but text stays in its own lane.

The new way (MMDiT): image tokens and text tokens sit side by side in the *same* attention layers. They look at each other freely the whole way down. This was introduced by Stable Diffusion 3 and adopted/extended by Flux. The result: cleaner gradient flow between text and image during training, better prompt adherence at inference.

See [[dit]] for the architecture story and [[papers/dit]] for the original scaling result that made transformers the default backbone.

### Rectified flow training

**DDPM ε-prediction** (SDXL): the model is taught "given this noisy image, predict the noise that was added to it." You repeat that prediction many times to walk back from pure noise to a clean image. The path the model takes through "noise space" is curvy.

**Rectified flow** (Flux): same broad idea — map noise to data — but the training objective is set up so the model learns *straight-line* paths instead of curvy ones. Straight lines mean you can take bigger steps without missing the target. That's part of why FLUX.1 schnell can sample in 4 steps where SDXL needs 25+.

Not yet ingested into the wiki — would be a worthwhile future read for understanding modern fast-sampling models.

### Guidance distillation

Classifier-free guidance (CFG) is the knob that makes the model "listen harder" to your prompt. Mechanically, the model runs twice per step — once with your prompt, once without — and the difference between the two predictions gets amplified. So CFG always costs 2× compute per sampling step.

**FLUX.1-dev is distilled.** Someone trained it specifically to produce high-quality outputs at much lower CFG (~3.5) than SDXL needs (~7). That means:
- Fewer effective forward passes per step (CFG-2× still applies, but at a lower amplification factor that doesn't push the model into bad territory)
- Less risk of over-saturation / "burnt" artifacts at high CFG
- See [[classifier-free-guidance]] and [[papers/cfg]] for the underlying CFG dynamics

FLUX.1 schnell is distilled even further — it needs no CFG at all and samples in 4 steps.

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

A few terms from the table:
- **Inpainting / outpainting** — fill in a masked region of an image / extend an image beyond its borders.
- **Canny edges** — a classic edge-detection algorithm that traces outlines. Used to guide generation along a structure.
- **Depth maps** — a grayscale image where brightness encodes distance from the camera. Same idea — guides generation along 3D structure.
- **Image variation** — given an input image, produce variants of it.

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

`bfloat16` and `fp16` are half-precision number formats — each weight takes 2 bytes instead of 4. Same model, half the VRAM, negligible quality hit for inference.

## VRAM requirements (practitioner knowledge†)

VRAM is the GPU's own memory — the model's weights have to fit there, plus the working tensors during a forward pass.

| Setup | Comfortable on |
|-------|----------------|
| Full bf16 inference | 24GB (RTX 4090, A6000) |
| With CPU offload | 16GB |
| With NF4 quantization | 10-12GB |
| For LoRA training (rank 16) | 24GB minimum, 48GB comfortable |

- **CPU offload** — when a layer isn't actively running, swap its weights out to system RAM. Slower but fits in less VRAM.
- **NF4 quantization** — store weights at 4 bits each (a special 4-bit number format from the QLoRA paper) instead of 16. Roughly 4× compression with a small quality hit.

LoRA training at 1024×1024 on Flux can spike to 30GB+; some trainers (AI-Toolkit) use gradient checkpointing (recompute activations instead of storing them) and lower-precision optimizers to fit in 24GB.

## See Also

- [[dit]]
- [[papers/dit]]
- [[latent-diffusion]]
- [[u-net]]
- [[classifier-free-guidance]]
- [[lora]]
- [[fine-tuning]]
- [[foundations]]
