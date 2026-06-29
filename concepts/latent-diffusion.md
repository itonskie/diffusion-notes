---
type: concept
tags:
  - foundations
  - diffusion
  - latent-diffusion
  - architecture
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Latent Diffusion

The trick that made high-resolution image generation tractable: run the diffusion process in a **compressed latent space** instead of on raw pixels. Introduced by Rombach et al. 2022 (the "LDM" paper, foundation of Stable Diffusion). Every modern image diffusion model (SD 1.5, SDXL, Flux, SD3) is a latent diffusion model.

## The problem with pixel-space diffusion

Running [[diffusion-models|diffusion]] directly on 1024×1024 RGB images:
- Each forward pass processes 3,145,728 values
- 30 sampling steps × 3M values × N-billion-parameter model = infeasible compute for consumer GPUs
- The model has to learn both **perceptual** detail (textures, sharpness) and **semantic** structure (what's in the image) in the same pixel space

Most of the compute is wasted on perceptual detail that doesn't change much across denoising steps.

## The solution

1. Train a separate **VAE** (variational autoencoder) to compress images:
   - `encoder E: image → latent` (downsamples spatially by 8×, channels = 4 or 16)
   - `decoder D: latent → image`
   - Trained with reconstruction loss + KL regularization (or VQ)
2. Run the [[diffusion-models|diffusion process]] **entirely in latent space** `z = E(x)`.
3. At inference, sample `z_T ~ N(0, I)`, denoise to `z_0`, then `image = D(z_0)`.

The diffusion model never sees raw pixels. It learns to denoise compressed representations.

## What this buys

- **64× less data per token** at the same effective image resolution (8× downsampling in each spatial dim)
- Encoder removes perceptual high-frequencies the diffusion model doesn't need to model
- Diffusion model focuses on **semantic** structure; VAE decoder handles perceptual reconstruction
- A 1024×1024 image becomes a 128×128 latent — tractable on consumer GPUs

## What this costs

- **The VAE is a separate component** — its quality bottlenecks final output quality. SDXL's VAE is famously a weak point.
- **VAE training is its own problem** — you can fine-tune just the VAE for specific domains (anime, faces, photo) to improve final quality without retraining the diffusion model.
- **Reconstruction errors are baked in** — anything the VAE can't decode well (text, fine textures) will be wrong no matter how good the diffusion model is.

This is why text in images was a known weakness of SD 1.5 / SDXL: the VAE struggled with fine glyphs. Flux's larger 16-channel VAE handles text dramatically better.

## VAE channel counts in practice

| Model | VAE latent shape (for 1024×1024 image) | Channels |
|-------|----------------------------------------|----------|
| SD 1.5 | 128×128 | 4 |
| SDXL | 128×128 | 4 |
| Flux | 128×128 | **16** |
| SD3 | 128×128 | 16 |

More channels = more information per spatial location = better reconstruction = better final outputs. The shift to 16-channel VAEs is one of the biggest practical improvements in 2024-era models.

## Conditioning in latent diffusion

Text (or other) conditioning is injected into the diffusion model via cross-attention in the [[u-net|U-Net]] or as tokens in the [[dit|DiT]]. The text encoder (CLIP, T5, etc.) is a separate frozen network — also a quality bottleneck. T5-XXL is meaningfully better at long/specific prompts than CLIP.

```
text → [text encoder] → text embedding
noisy latent z_t → [diffusion model](t, text embedding) → noise prediction
```

The "Stable Diffusion stack" is therefore three networks: VAE encoder, diffusion model, VAE decoder — plus the text encoder.

## See Also

- [[diffusion-models]]
- [[u-net]]
- [[dit]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[foundations]]
