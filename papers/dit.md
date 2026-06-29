---
type: paper
tags:
  - foundations
  - dit
  - architecture
  - paper
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# DiT — Scalable Diffusion Models with Transformers

> Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.
> https://arxiv.org/abs/2212.09748 • [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2212.09748)

## Status

**Developing** — content from ar5iv HTML render. This paper introduced the architecture used by Flux, SD3, and Wan 2.2.

## Abstract (verbatim)

> "We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches."

## The contribution

The paper makes one big argument and supports it experimentally:

> **The U-Net's inductive biases are not load-bearing for diffusion. A plain transformer scales better.**

Specifically, the paper shows that:

1. A transformer that operates on patches of [[latent-diffusion|latent images]] can replace [[u-net|U-Net]]
2. Done right (with `adaLN-Zero` conditioning), it beats U-Net on ImageNet at the same compute budget
3. **Gflops, not parameter count**, predicts FID — transformer scaling laws apply

This kicked off the wave of DiT-based diffusion models (Pixart, SD3, Flux, Wan).

## Architecture

### Patchify

For a `32×32×4` latent (from the [[latent-diffusion|VAE]]) and patch size `p`, you get `T = (32/p)²` tokens:

| `p` | Tokens | Gflops impact |
|-----|--------|---------------|
| 8 | 16 | minimal |
| 4 | 64 | medium |
| 2 | 256 | high |

Halving `p` **quadruples Gflops** while leaving parameter count unchanged. This is how the paper decouples model size from compute: same `DiT-XL` config can be sampled at `p=2` (expensive, high quality) or `p=8` (cheap, lower quality).

### Conditioning mechanism — the four candidates

This is where the paper does its real work. They test four ways to inject timestep + class into the transformer:

1. **In-context conditioning** — append `t` and `c` as extra tokens. Standard ViT, no architectural changes. Negligible compute overhead.
2. **Cross-attention** — add a cross-attention layer per block that attends from image tokens to a `[t, c]` conditioning sequence. ~15% Gflop overhead.
3. **Adaptive layer norm (adaLN)** — replace LayerNorm's fixed `γ, β` with MLP-regressed `γ(t,c), β(t,c)`. Compute-efficient.
4. **adaLN-Zero** — extends adaLN with an additional `α(t,c)` scaling applied *before residuals*, AND zero-initializes so the block starts as the identity function.

### The winner: adaLN-Zero

Figure 5 of the paper compares all four at 400K training iters. **adaLN-Zero reached nearly half the FID of in-context conditioning, with the lowest compute overhead.**

What makes adaLN-Zero special isn't the adaLN part — it's the **zero initialization**. The identity-init means each block adds nothing at the start of training and gradually learns to do useful work. Without this, the conditioning swamps the early-training signal.

The paper concludes:
> "adaLN-Zero significantly outperforms vanilla adaLN."

## Model scale lineup

| Config | Layers | Hidden | Heads | Params | Gflops (p=4) |
|--------|--------|--------|-------|--------|--------------|
| DiT-S | 12 | 384 | 6 | ~33M | 1.4 |
| DiT-B | 12 | 768 | 12 | ~131M | 5.6 |
| DiT-L | 24 | 1024 | 16 | ~459M | 19.7 |
| DiT-XL | 28 | 1152 | 16 | ~676M | 29.1 |

DiT-XL/2 (patch size 2): **118.6 Gflops** — this is the configuration that beat ADM at much lower compute.

## Scaling result

The headline finding: **plot FID vs Gflops and you get a clean negative log-linear trend across configs and patch sizes**. The plot is the entire reason this paper got cited everywhere.

| Comparison | Gflops | FID-50K (256×256) |
|------------|--------|-------------------|
| ADM (U-Net SOTA) | 1120 | 3.60 |
| LDM-4 (U-Net) | 103.6 | 3.60 |
| **DiT-XL/2 (no guidance)** | 118.6 | **9.62** |
| **DiT-XL/2 + CFG 1.5** | 118.6 | **2.27** |

DiT-XL/2 with CFG beats ADM at ~10× less compute, and beats LDM at comparable compute.

For 512×512:

| Config | Gflops | FID |
|--------|--------|-----|
| ADM (previous SOTA) | — | 3.85 |
| DiT-XL/2 + CFG 1.5 | 524.6 | **3.04** |

## Patch size matters more than parameters

DiT-S/2 and DiT-B/4 have similar Gflops but very different parameter counts; they achieve **similar FID**. This is the paper's strongest evidence for the "Gflops, not params" claim — `param count alone does not predict FID; compute does.`

## Training hyperparameters

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| LR | 1e-4 (constant) |
| Batch size | 256 |
| EMA decay | 0.9999 |
| Weight decay | none |
| Augmentation | horizontal flips only |
| Iterations | 400K (scaling) / 7M (final 256) / 3M (final 512) |
| LR warmup | none |

Strikingly simple recipe. The architecture, not the training tricks, is the contribution.

## How `t` and class embeddings get combined

- `t` → 256-D sinusoidal frequency embedding → 2-layer MLP with SiLU → transformer hidden dim
- `c` → class embedding → same projection
- **Sum** `t_emb + c_emb`, pass through SiLU + linear
- Output: 4× (adaLN) or **6× (adaLN-Zero)** the hidden dim, sliced into `γ, β, α` for the two LayerNorms in the block

That 6× output is what's regressed for the per-block conditioning. For text-to-image variants like Flux, `c` becomes the text encoder output (T5 / CLIP) instead of a class embedding.

## Limitations the paper acknowledges

- Compute-quadratic in token count — no free lunch at high resolution
- Plain class-conditional setup; doesn't address text-to-image directly (later papers like SD3, Flux extend it)
- ImageNet only — they didn't try high-aesthetic data

## What DiT became

| Model | DiT lineage |
|-------|-------------|
| Pixart-α (2023) | Standard DiT, ImageNet-pretrained then fine-tuned for text-to-image |
| SD3 (2024) | MMDiT — multi-modal DiT, image and text tokens share attention |
| Flux (2024) | MMDiT, double-stream + single-stream blocks |
| Wan 2.2 (2024) | Spatiotemporal DiT for video |

The "T2I/T2V models all use DiT" trend in 2024-2025 traces back directly to this paper.

## See Also

- [[dit]]
- [[u-net]]
- [[diffusion-models]]
- [[latent-diffusion]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[foundations]]
