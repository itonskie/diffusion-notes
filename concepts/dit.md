---
type: concept
tags:
  - foundations
  - architecture
  - dit
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Diffusion Transformer (DiT)

The transformer-based replacement for [[u-net|U-Net]] as the backbone of modern diffusion models. Introduced by Peebles & Xie 2023 ([[papers/dit|paper]]). Used by Flux, Stable Diffusion 3, and Wan 2.2.

## The architecture

1. **Patchify** the noisy [[latent-diffusion|latent]] image: split into non-overlapping patches (typically 2×2), flatten each patch into a token. A 64×64 latent at patch size 2 gives 1024 tokens.
2. **Add positional embeddings** (learned or sinusoidal) — transformers don't have built-in spatial awareness.
3. **Stack N transformer blocks** (self-attention + MLP), conditioned on time `t` and class/text embedding `c`.
4. **Unpatchify** the output tokens back into noise predictions in image space.

The transformer blocks are mostly standard, with one diffusion-specific twist for conditioning: **adaLN-Zero**.

## adaLN-Zero — the conditioning trick

Standard LayerNorm has fixed scale `γ` and shift `β`. DiT replaces these with conditioning-dependent versions:

```
LN(x) → γ(t,c) · LN(x) + β(t,c)
```

where `γ` and `β` come from an MLP applied to the `t + c` embedding. Additionally, a per-dimension scaling factor `α(t,c)` is applied *before* the residual connection:

```
x + α(t,c) · attention(LN(x))
```

**Zero init**: `α` is initialized to zero so each block is initially the identity. The model starts as a "no-op" and learns to do useful work — significantly more stable training.

This is what makes DiT work where naive "ViT for diffusion" doesn't.

## Why DiT wins at scale

The headline result from the paper: DiT shows **clean log-linear scaling** of FID vs compute (Gflops). U-Nets do not — they plateau.

Why does it scale better?
- **No spatial inductive bias** — convolutions assume translation equivariance and local locality; this helps small models but constrains big ones. Transformers learn whatever bias the data wants.
- **Cleaner conditioning** — `adaLN-Zero` plus tokens for the prompt scales naturally; adding modalities is just adding more tokens.
- **Compute is uniform** — every block does the same operation, so optimization and parallelization are simpler.

Cost: attention is `O(N²)` in token count. Patchifying lets you control `N` directly (smaller patches → more tokens → quadratically more compute).

## Patch size trade-off

- Patch size 2 (default) — high quality, expensive
- Patch size 4 — 4× fewer tokens, 16× less attention compute, lower quality
- Flux uses patch size 2 on a 16-channel VAE latent

## DiT in modern models

| Model | DiT variant | Notes |
|-------|-------------|-------|
| Flux | MMDiT (multi-modal DiT) | Text and image tokens share attention; uses RoPE |
| SD3 | MMDiT | Similar to Flux; published as the architecture co-author |
| Wan 2.2 | Spatiotemporal DiT | Adds temporal axis for video |
| Pixart-α / Pixart-Σ | Standard DiT | Smaller, efficient, demo platforms |

The trend across new releases is clearly toward DiT — no major image/video diffusion model launched in 2024-2025 uses U-Net.

## See Also

- [[u-net]]
- [[diffusion-models]]
- [[latent-diffusion]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[papers/dit]]
- [[foundations]]
