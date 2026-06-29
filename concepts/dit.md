---
type: concept
tags:
  - foundations
  - architecture
  - dit
date: 2026-06-29
updated: 2026-06-29
---

# Diffusion Transformer (DiT)

The "what if we just used a transformer instead?" answer to [[u-net|U-Net]]. Same job — given a noisy image and a prompt, predict the noise — but built out of transformer blocks instead of convolutions. Introduced by Peebles & Xie 2023 ([[papers/dit|paper]]). It's what Flux, Stable Diffusion 3, and Wan 2.2 use under the hood.

## The picture

A transformer can only process a list of tokens — it has no concept of "image." So the trick is: chop the image into a grid of small squares, treat each square as one token, and feed the list of tokens into a regular transformer. At the end, paste the squares back together into an image-shaped output.

Compared to [[u-net|U-Net]]: no down-path, no up-path, no skips, no special image plumbing. Just a stack of transformer blocks all working at the same resolution. It throws away U-Net's "I know images are 2D and multi-scale" assumptions and lets the model learn whatever it wants from the data — which turns out to scale better when you have a lot of data and compute.

## The architecture

1. **Patchify** the noisy [[latent-diffusion|latent]] image: split into non-overlapping patches (typically 2×2 pixels per patch), flatten each patch into a token (a vector). A 64×64 latent at patch size 2 gives 1024 tokens. ("Token" here is just transformer-speak for "one input vector in the sequence.")
2. **Add positional embeddings** (learned or sinusoidal) — transformers see an unordered list; positional embeddings tell each token where it sits in the grid. Without these the model literally can't tell top from bottom.
3. **Stack N transformer blocks** (self-attention + MLP — standard transformer block: every token looks at every other token, then each token gets passed through a small feed-forward network). Each block is told the current timestep `t` (which denoising step we're on) and the class/text embedding `c` (what we're trying to generate).
4. **Unpatchify** the output tokens — reshape back from list-of-tokens into a grid of patches, glue patches back into an image-shaped tensor. That tensor is the noise prediction.

The transformer blocks are mostly off-the-shelf, with one diffusion-specific twist for how `t` and `c` get plugged in: **adaLN-Zero**.

## adaLN-Zero — the conditioning trick

A standard transformer has **LayerNorm** layers (normalize each token's vector so it has zero mean / unit variance, then apply a fixed learned scale `γ` and shift `β`). DiT modifies this: the scale and shift are *computed from the conditioning* `t + c` instead of being fixed.

So the same LayerNorm behaves differently at every timestep and for every prompt. That's how the transformer "knows" what step it's on and what to generate.

There's a second modification too: a per-dimension scaling factor `α` (also computed from `t + c`) is applied to each block's output *before* the residual connection (the "add the block's input back into its output" trick that keeps transformers trainable).

The "Zero" in adaLN-Zero: `α` is initialized to **zero**, so at the start of training each block is the identity — input passes through untouched. The model literally starts as a no-op and learns its way to doing useful work. This turns out to matter a lot for stability — DiT trained naively (without the Zero init) is much harder to train.

### The math, if you want it

Standard transformer block with LayerNorm:
```
x + attention(LN(x))
```
where `LN` is LayerNorm with fixed scale/shift.

DiT's adaLN-Zero block:
```
LN(x) → γ(t,c) · LN(x) + β(t,c)
x → x + α(t,c) · attention(adaLN(x))
```

- `x` — the token sequence (what the block is processing).
- `LN` — vanilla LayerNorm (normalize to zero mean, unit variance).
- `γ(t,c)`, `β(t,c)` — scale and shift, but computed by a small MLP that eats `t + c`. Replaces LayerNorm's fixed parameters with conditioning-driven ones.
- `α(t,c)` — extra per-dimension scale on the block's output, also from the same MLP. Initialized to zero.
- Initializing `α = 0` means at training step zero, `attention(...)` is multiplied by zero and the block returns just `x`. Pure identity. The model has to *earn* its non-identity behavior.

## Why DiT wins at scale

The headline result from the [[papers/dit|paper]]: DiT shows **clean log-linear scaling** of **FID** (Fréchet Inception Distance — a quality metric where lower is better, computed by comparing generated images to real images in a pretrained classifier's feature space) versus **Gflops** (giga-floating-point-operations — how much compute one forward pass takes). U-Nets do not — their quality plateaus as you scale them up.

Concretely:

- DiT-XL/2 at **118.6 Gflops** → FID 2.27 on ImageNet 256×256 (with [[classifier-free-guidance|CFG]])
- ADM (the best U-Net of the time) at **1120 Gflops** → FID 3.60 on the same task
- **DiT beat U-Net at ~10× less compute**

The paper also showed that **Gflops, not parameter count, predict FID**: DiT-S/2 and DiT-B/4 have similar Gflops but very different parameter counts, and reach essentially identical FID. This is the strongest evidence that transformer scaling laws apply to diffusion too — you can trade parameters for resolution and get the same quality as long as the compute budget matches.

Why does it scale better?
- **No spatial inductive bias** — convolutions hard-code "things that work in one spot work in another" and "neighbors matter more than distant pixels." Those assumptions help when data is scarce but constrain the model when data is plentiful. Transformers learn whatever bias the data actually wants.
- **Cleaner conditioning** — `adaLN-Zero` plus tokens for the prompt scales naturally; adding a new conditioning modality is just adding more tokens to the sequence.
- **Compute is uniform** — every block does the same operation (attention + MLP), so optimization, hardware utilization, and parallelization are all simpler.

Cost: attention is `O(N²)` in token count — double the tokens, quadruple the compute. Patchifying gives you the knob to control `N`: smaller patches = more tokens = quadratically more compute = higher quality.

## Patch size trade-off

- Patch size 2 (default) — high quality, expensive
- Patch size 4 — 4× fewer tokens, 16× less attention compute, lower quality
- Flux uses patch size 2 on a 16-channel VAE latent (see [[latent-diffusion]])

## DiT in modern models

| Model | DiT variant | Notes |
|-------|-------------|-------|
| Flux | MMDiT (multi-modal DiT) | Text and image tokens share attention; uses RoPE |
| SD3 | MMDiT | Similar to Flux; published as the architecture co-author |
| Wan 2.2 | Spatiotemporal DiT | Adds temporal axis for video |
| Pixart-α / Pixart-Σ | Standard DiT | Smaller, efficient, demo platforms |

**MMDiT** (multi-modal DiT) means text tokens and image tokens go into the *same* attention layers and can attend to each other directly, rather than text only entering via cross-attention. **RoPE** (rotary positional embedding) is a particular way of encoding token positions that works better than the original sinusoidal scheme — it bakes position into the attention computation itself.

The trend across new releases is clearly toward DiT — no major image/video diffusion model launched in 2024-2025 uses U-Net.

## See Also

- [[u-net]]
- [[diffusion-models]]
- [[latent-diffusion]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[papers/dit]]
- [[foundations]]
