---
type: paper
tags:
  - foundations
  - dit
  - architecture
  - paper
date: 2026-06-29
updated: 2026-06-29
status: summary
---

# DiT — Scalable Diffusion Models with Transformers

> Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.
> https://arxiv.org/abs/2212.09748 • [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2212.09748)

## Status

**Developing** — content from ar5iv HTML render. This paper introduced the architecture used by Flux, SD3, and Wan 2.2.

## What this paper actually claims

Before this paper, every serious diffusion model used a **U-Net** — a convolutional architecture that downsamples an image through a stack of blocks, hits a bottleneck, then upsamples back. U-Nets work great. People assumed the U-Net's specific shape (the convolutions, the skip connections, the spatial downsample-then-upsample) was important — that the architecture itself was helping the diffusion model "see" images.

This paper says: nope, throw all that away. Use a **plain transformer** — the same architecture as a Vision Transformer (ViT), the one that just chops an image into patches and treats each patch as a token. Replace the U-Net entirely. Result: at the same compute budget, the transformer is *better*, and it scales more predictably (you can plot FID against Gflops and get a clean log-linear line).

That single result kicked off the wave of DiT-based diffusion models you see today: Pixart, SD3, Flux, Wan. The U-Net era is basically over for new state-of-the-art models.

> **Domain decode first.** Words you need: **U-Net** = the convolutional encoder-decoder architecture diffusion models used before DiT (see [[u-net]]); **transformer** = the attention-based architecture from "Attention Is All You Need"; **ViT** (Vision Transformer) = a transformer that processes images by splitting them into fixed-size square patches and treating each patch as a token; **latent** = a compressed representation of an image produced by a separately-trained autoencoder (the VAE), so the diffusion model works on a small grid like 32×32×4 instead of a 256×256×3 image; **inductive bias** = the assumptions baked into an architecture's structure (convolutions assume locality and translation-invariance; transformers assume essentially nothing); **Gflops** = billions of floating-point operations per forward pass, a measure of compute cost independent of parameter count.

## Abstract (verbatim)

> "We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches."

## The big argument, in one line

The paper makes one big argument and supports it experimentally:

> **The U-Net's inductive biases are not load-bearing for diffusion. A plain transformer scales better.**

Specifically, the paper shows that:

1. A transformer that operates on patches of [[latent-diffusion|latent images]] can replace [[u-net|U-Net]]
2. Done right (with `adaLN-Zero` conditioning), it beats U-Net on ImageNet at the same compute budget
3. **Gflops, not parameter count**, predicts FID — transformer scaling laws apply

This kicked off the wave of DiT-based diffusion models (Pixart, SD3, Flux, Wan).

## How DiT actually works

### Patchify (input handling)

DiT doesn't work on pixels — it works on a **latent** produced by a pretrained VAE (the same VAE Stable Diffusion uses). For a 256×256 image, the latent is `32×32×4`. So already you've shrunk the problem by 8×.

Then DiT slices that latent into square patches of side `p`. Each patch becomes one token. The whole latent becomes a sequence of `T = (32/p)²` tokens that you feed into a transformer.

| `p` | Tokens | Gflops impact |
|-----|--------|---------------|
| 8 | 16 | minimal |
| 4 | 64 | medium |
| 2 | 256 | high |

Halving `p` **quadruples Gflops** while leaving parameter count unchanged. That's a big deal: it means the *same* DiT-XL config can be sampled at `p=2` (expensive, high quality) or `p=8` (cheap, lower quality). Compute and capacity are now separate dials.

### How to inject "what timestep are we at" and "what class do we want"

The transformer doesn't know what timestep `t` it's denoising at, and it doesn't know what class the user asked for. Those have to be injected somehow. The paper tested four ways:

1. **In-context conditioning** — append `t` and `c` as extra tokens at the front of the sequence. Standard ViT, no architectural changes. Negligible compute overhead.
2. **Cross-attention** — add a separate cross-attention layer per block that lets image tokens attend to a `[t, c]` conditioning sequence. ~15% Gflop overhead.
3. **Adaptive layer norm (adaLN)** — LayerNorm normally has fixed scale and shift parameters (`γ, β`). adaLN instead has a small MLP compute those from `(t, c)`, so the normalization itself is conditioned on what we want.
4. **adaLN-Zero** — adaLN plus an additional `α(t, c)` scaling applied *before* each residual connection, AND the MLP that produces `(γ, β, α)` is **zero-initialized** so each block starts as the identity function.

### The winner: adaLN-Zero

Figure 5 of the paper compares all four at 400K training iters. **adaLN-Zero reached nearly half the FID of in-context conditioning, with the lowest compute overhead.**

What makes adaLN-Zero special isn't the adaLN part — it's the **zero initialization**. Identity-init means each transformer block adds nothing on training step one. Conditioning information has to gradually push the block away from being a no-op, which lets training stabilize before the conditioning swamps the early-training signal. (Same trick as LoRA's zero-init `B` matrix, for the same reason.)

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

> Decode: `DiT-XL/2` is shorthand for "DiT-XL config, patch size 2." That naming convention follows DiT-XL/4, DiT-XL/8, etc. Same params, different compute.

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

This is the part that surprised people the most. Conventional wisdom was "more parameters = more capacity = better." DiT shows that for diffusion, what matters is how much compute you spend per forward pass. A small model run at small patch size (more tokens, more compute) beats a big model run at big patch size (fewer tokens, less compute) when their Gflops match.

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

## Math sketch — how `t` and class embeddings get combined

This is the wiring that feeds `(t, c)` into adaLN-Zero. Following the paper:

- `t` → 256-D sinusoidal frequency embedding → 2-layer MLP with SiLU → transformer hidden dim
- `c` → class embedding → same projection
- **Sum** `t_emb + c_emb`, pass through SiLU + linear
- Output: 4× (adaLN) or **6× (adaLN-Zero)** the hidden dim, sliced into `γ, β, α` for the two LayerNorms in the block

> Decode the symbols: **sinusoidal embedding** = the same trick transformers use for positional encoding — turns a scalar into a fixed-size vector by feeding it through sines/cosines at different frequencies; **SiLU** = a smooth nonlinearity, sometimes called swish; **`γ, β, α`** = scale, shift, and residual-gate parameters produced by the MLP, applied inside the LayerNorm and around the residual connection. Why 6× the hidden dim? Because each transformer block has two LayerNorm spots (one before attention, one before MLP), and each needs three vectors (`γ, β, α`), so 2 × 3 = 6 slots of hidden-dim length.

For text-to-image variants like Flux, `c` becomes the text encoder output (T5 / CLIP) instead of a class embedding.

## Limitations the paper acknowledges

- Compute-quadratic in token count — no free lunch at high resolution
- Plain class-conditional setup; doesn't address text-to-image directly (later papers like SD3, Flux extend it)
- ImageNet only — they didn't try high-aesthetic data

## What this paper changed downstream

| Model | DiT lineage |
|-------|-------------|
| Pixart-α (2023) | Standard DiT, ImageNet-pretrained then fine-tuned for text-to-image |
| SD3 (2024) | MMDiT — multi-modal DiT, image and text tokens share attention |
| Flux (2024) | MMDiT, double-stream + single-stream blocks |
| Wan 2.2 (2024) | Spatiotemporal DiT for video |

The "T2I/T2V models all use DiT" trend in 2024-2025 traces back directly to this paper. The U-Net era didn't immediately end — SDXL (2023) was still U-Net-based — but every new flagship model since SD3 has been some flavor of DiT.

## See Also

- [[dit]]
- [[u-net]]
- [[diffusion-models]]
- [[latent-diffusion]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[foundations]]
