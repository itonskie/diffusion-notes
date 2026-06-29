---
type: concept
tags:
  - foundations
  - diffusion
  - latent-diffusion
  - architecture
date: 2026-06-29
updated: 2026-06-29
---

# Latent Diffusion

The single trick that made high-resolution image generation possible on consumer GPUs: don't run diffusion on pixels — run it on a *compressed* version of the image, then expand back to pixels at the end. Introduced by Rombach et al. 2022 (the "LDM" paper, the foundation of Stable Diffusion). Every modern image diffusion model — SD 1.5, SDXL, Flux, SD3 — is a latent diffusion model.

## The picture

Imagine you want to generate a 1024×1024 image by gradually denoising it. If you do the denoising directly on the 1024×1024 pixels, every step of the model has to chew through ~3 million numbers. Do 30 steps. That's a lot of compute for the GPU.

Most of that compute is wasted: the model is busy painting tiny texture details (skin pores, fabric grain) when really it's still figuring out the big picture (what's in the image, where things are). The model is doing low-level work and high-level work in the same space.

Latent diffusion's bet: train a separate, smaller network to compress the image down to a much smaller representation — say 128×128 — and then run diffusion in *that* compressed space. At the very end, expand the result back up to 1024×1024 pixels.

The diffusion model gets to focus on "what's in the image" because the compressed space has already thrown away "what does the skin look like up close." A second network handles the painting-the-pixels job once.

## The compressor: VAE

The compressor is a **VAE** (variational autoencoder — an autoencoder is just "encoder squishes input down, decoder reconstructs it"; the "variational" part adds a regularization that keeps the squished space well-behaved and continuous instead of full of weird gaps).

It has two halves:
- `encoder E: image → latent` — downsamples spatially by 8× in each dimension, outputs a small tensor with 4 or 16 channels per spatial location.
- `decoder D: latent → image` — reverses the squish, expands back to full resolution.

A **latent** is just the squished representation — the encoder's output, the decoder's input. Spatially smaller than the image, but with more channels per location (each location now packs more abstract info).

Trained once, separately, with reconstruction loss (decoded image should match original) plus a KL regularization term (keeps the latent space well-behaved). Once trained, both halves are frozen and reused by every diffusion model that targets this latent space.

## The process, end to end

1. Train a VAE on lots of images.
2. To train the diffusion model: take an image, run it through the encoder to get the latent `z = E(image)`, do all the diffusion training on `z`.
3. To generate an image at inference: sample pure noise `z_T ~ N(0, I)` in latent space, run the [[diffusion-models|diffusion process]] to denoise it down to `z_0`, then run `image = D(z_0)` to get the actual pixels.

The diffusion model never sees raw pixels. It only ever sees the VAE's squished representation. Pixels only appear at the very last step.

## What this buys

- **64× less data per step** — 8× downsampling in each spatial direction = 8 × 8 = 64× fewer spatial locations to process. (Channels go up a bit but nowhere near enough to cancel this out.)
- The VAE encoder throws away high-frequency perceptual detail (the stuff that varies pixel-to-pixel) that the diffusion model doesn't need to bother modeling.
- Clean division of labor: the diffusion model handles **semantic** structure (what's in the image); the VAE decoder handles **perceptual** reconstruction (what it looks like up close).
- A 1024×1024 image becomes a 128×128 latent — small enough to denoise on consumer GPUs.

## What this costs

- **The VAE is a separate component, and its quality bottlenecks everything.** The decoder can only restore detail it was trained to restore. If the VAE can't represent something well, the diffusion model can't fix that. SDXL's VAE is famously a weak link.
- **VAE training is its own problem.** You can fine-tune just the VAE for specific domains (anime, faces, photography) to improve final quality without touching the diffusion model.
- **Reconstruction errors are baked in.** Anything the VAE can't decode well — text, fine textures — will be wrong no matter how good the diffusion model is at predicting the latent.

This is why text in images was a known weakness of SD 1.5 / SDXL: the VAE was bad at fine glyphs (the small shapes that make up letters). Flux's bigger 16-channel VAE handles text dramatically better.

## VAE channel counts in practice

| Model | VAE latent shape (for 1024×1024 image) | Channels |
|-------|----------------------------------------|----------|
| SD 1.5 | 128×128 | 4 |
| SDXL | 128×128 | 4 |
| Flux | 128×128 | **16** |
| SD3 | 128×128 | 16 |

"Channels" = how many numbers describe each spatial location in the latent. 4 channels means each location is a 4-dimensional vector; 16 means a 16-dimensional vector. More channels = more information per location = better reconstruction = better final outputs. The shift from 4-channel to 16-channel VAEs is one of the biggest quiet improvements in 2024-era models — most people see the better outputs but don't realize it's mostly the VAE doing it.

## Conditioning in latent diffusion

"Conditioning" = the prompt or other input that steers what gets generated. Text conditioning gets injected into the diffusion model either via cross-attention in the [[u-net|U-Net]] or as extra tokens in the [[dit|DiT]].

The text encoder (CLIP, T5, etc.) is a *separate, frozen* network that turns text into vectors. **CLIP** is OpenAI's text+image model that maps both into the same vector space. **T5** is Google's pure text encoder, much better than CLIP at long or specific prompts. The text encoder is also a quality bottleneck — Flux uses T5-XXL alongside CLIP because T5 understands complex prompts noticeably better.

```
text → [text encoder] → text embedding
noisy latent z_t → [diffusion model](t, text embedding) → noise prediction
```

So the full "Stable Diffusion stack" at inference is *four* networks running together: text encoder, VAE encoder (only at training — not needed at inference since you start from random noise), diffusion model, VAE decoder.

## See Also

- [[diffusion-models]]
- [[u-net]]
- [[dit]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[foundations]]
