---
type: concept
tags:
  - foundations
  - architecture
  - u-net
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# U-Net

The convolutional encoder-decoder backbone used by SD 1.5, SD 2.x, and SDXL. Originally introduced for biomedical image segmentation (Ronneberger et al. 2015), adapted for diffusion in [[ddpm|DDPM]] and refined heavily afterward.

## Shape

A U-Net has two halves joined at the bottom:

- **Down path (encoder)**: repeated `Conv → activation → downsample` blocks. Spatial resolution halves at each level; channels double.
- **Bottleneck**: deepest level, smallest spatial resolution, most channels.
- **Up path (decoder)**: repeated `upsample → Conv → activation` blocks. Spatial resolution doubles; channels halve.
- **Skip connections**: concatenate features from the down path into the up path at matching resolutions. This is what makes it "U"-shaped.

Why skips? Without them, fine-grained spatial detail destroyed in the down path can't be recovered in the up path. Skips preserve high-frequency information directly.

## What changed for diffusion

The original U-Net was for segmentation. Diffusion adapts it:

- **Time embedding** — `t` (the noise step) is encoded (sinusoidal + MLP) and injected at every resolution via adaptive group norm (AdaGN) or addition to feature maps.
- **Attention blocks** at lower resolutions (typically 8×8 and 16×16 latents) — self-attention captures long-range structure that convolutions can't, without quadratic cost everywhere.
- **Cross-attention** for conditioning (text prompts, etc.) — keys and values come from text encoder embeddings, queries come from image features.
- **BigGAN-style residual blocks** — more stable training than vanilla Conv-Norm-ReLU.
- **Wider channel counts** — diffusion U-Nets are much bigger than the original biomedical one (hundreds of millions of parameters for SDXL).

## Three ways to inject conditioning into a U-Net

From the HF Diffusion Course Unit 2 — the canonical patterns:

1. **Extra input channels.** Concatenate the conditioning into the input alongside the noisy image. Used when the condition has the same spatial shape as the image — segmentation masks, depth maps, blurred low-res images (for super-resolution). Also works for class labels: embed the label, expand spatially, concatenate.
2. **Project-and-add to internal feature maps.** Compute an embedding (timestep, CLIP image embedding, class embedding), project to match each layer's channel count, add to layer outputs. This is how time conditioning is always done. Also how the "Image Variations" Stable Diffusion variant takes a CLIP image embedding.
3. **Cross-attention.** Pass the conditioning as a sequence of embeddings. Add cross-attention layers in the U-Net where queries come from image features and keys/values come from the conditioning sequence. This is how text conditioning works in Stable Diffusion / SDXL — text → CLIP/T5 tokens → cross-attention in the U-Net.

The three patterns aren't mutually exclusive: a typical text-to-image U-Net uses (2) for time, (3) for text, and could use (1) if you also pass a depth map.

## What it's good at

- **Inductive bias for images** — convolutions are translation-equivariant, U-shape biases toward multi-scale structure. Trains efficiently from less data than transformers.
- **Memory-efficient** — convolutions scale well to large images; you don't pay quadratic cost across all pixels.

## Where it struggles

- **Long-range coherence** — without attention everywhere, things at opposite sides of the image can be inconsistent.
- **Scaling** — U-Net's empirical scaling laws are less clean than transformers; doubling parameters doesn't always double quality.
- **Hard to add modalities** — every new condition (depth, pose, sketch) needs new architectural plumbing (ControlNet was invented specifically to solve this).

## U-Net vs DiT

| Aspect | U-Net (SD 1.5 / SDXL) | DiT (Flux / SD3 / Wan 2.2) |
|--------|------------------------|-----------------------------|
| Inductive bias | Translation-equivariant, multi-scale | Minimal — patches and attention |
| Parameter scaling | Diminishing returns past ~5B | Clean log-linear scaling |
| Compute pattern | Convs (linear in pixels) + attention at low res | Attention dominant — quadratic in tokens |
| Conditioning | Cross-attention plumbing per modality | Tokens concatenated/cross-attended cleanly |
| Memory | Low at high resolution | High at high resolution |

The current consensus: U-Net is fine for smaller models and limited compute; DiT wins at scale.

See [[dit]] for the alternative.

## See Also

- [[diffusion-models]]
- [[dit]]
- [[ddpm]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[latent-diffusion]]
- [[foundations]]
