---
type: concept
tags:
  - foundations
  - architecture
  - u-net
date: 2026-06-29
updated: 2026-06-29
---

# U-Net

The image-eating neural network shape used by older Stable Diffusion models (SD 1.5, SD 2.x, SDXL). Its job inside a diffusion model is to look at a noisy image and predict the noise in it. Newer models (Flux, SD3) replaced it with a transformer, see [[dit|DiT]].

Originally invented for biomedical image segmentation (Ronneberger et al. 2015 — "segmentation" meaning: label every pixel as tumor/not-tumor). Adapted for diffusion in [[ddpm|DDPM]] and souped up heavily after.

## The picture

Draw a capital U. The left side of the U goes downward — it eats the image and squishes it smaller and smaller, each step keeping less spatial detail but more abstract meaning. The bottom of the U is the smallest, most abstract version. The right side of the U comes back up — it reconstructs the image, going from abstract back to full resolution.

The trick that makes it a "U" and not just an hourglass: at every level on the way down, you copy the features sideways across the U and paste them into the matching level on the way up. These sideways copies are called **skip connections**. Without them, all the fine detail (sharp edges, texture) gets crushed on the way down and the way up can't recover it. Skips give the up-path a cheat sheet of what the image *used to* look like at each resolution.

## Shape, more concretely

Two halves joined at the bottom:

- **Down path (encoder)**: repeated `Conv → activation → downsample` blocks. **Conv** is a convolution — a small filter that slides over the image and combines nearby pixels, the standard image-processing building block. **Downsample** means shrink the image in half (e.g. 64×64 → 32×32). At each step, spatial resolution halves; the number of **channels** (parallel feature maps at each pixel — think "how many different things does this pixel know about") doubles.
- **Bottleneck**: the deepest level — smallest spatial resolution, most channels. The image is now a tiny, very rich abstract summary.
- **Up path (decoder)**: repeated `upsample → Conv → activation` blocks. Spatial resolution doubles each step; channels halve. Goes back up to original size.
- **Skip connections**: at each matching level, glue the down-path's features onto the up-path's input. This is what makes it "U"-shaped on a diagram.

Why skips? Without them, fine-grained spatial detail destroyed on the down path can't be recovered on the up path. Skips preserve high-frequency information (sharp transitions, edges, texture) directly.

## What changed when U-Net got used for diffusion

The original U-Net just ate an image and spat out a labeled image. Diffusion needs more — the same network has to handle many noise levels and respond to text prompts. So:

- **Time embedding** — the network has to know *which noise step* it's denoising (early step = mostly noise, late step = almost clean image). The step number `t` gets turned into a vector (sinusoidal positional encoding + small MLP — same trick transformers use for position) and injected at every level. The injection mechanism is usually **AdaGN** (adaptive group normalization — a normalization layer whose scale/shift parameters are computed from the time embedding instead of being fixed).
- **Attention blocks** at lower resolutions — typically at the 8×8 and 16×16 [[latent-diffusion|latent]] sizes. **Self-attention** is the transformer trick where every pixel can look at every other pixel; convolutions can't do that directly. It's only added at low resolutions because attention is quadratic in pixel count — adding it at full resolution would be ruinously expensive.
- **Cross-attention** for conditioning — attention where one side is the image and the other side is something else, like a text prompt. Image asks "what should I look like?", text answers. This is how the prompt actually influences the image.
- **BigGAN-style residual blocks** — a specific block design (Conv-Norm-something + a skip) borrowed from the BigGAN paper because plain Conv-Norm-ReLU was unstable to train at diffusion scale.
- **Wider channel counts** — diffusion U-Nets are way bigger than the biomedical original. SDXL's U-Net is hundreds of millions of parameters.

## Three ways to inject conditioning into a U-Net

"Conditioning" means: anything that steers what the model generates — text prompt, class label, depth map, sketch, low-res image. From the HuggingFace Diffusion Course Unit 2, three patterns:

1. **Extra input channels.** Just paste the conditioning onto the image as extra channels. Works when the condition has the same spatial shape as the image — segmentation masks, depth maps, blurred low-res images (for super-resolution). Also works for class labels: embed the label as a vector, broadcast it to fill the whole spatial size, concatenate.
2. **Project-and-add to internal feature maps.** Compute an embedding (timestep, CLIP image embedding, class embedding — **CLIP** is OpenAI's model that maps images and text into the same vector space), project it to match each layer's channel count, add it to layer outputs. This is how time conditioning is always done. Also how the "Image Variations" Stable Diffusion variant accepts a CLIP image embedding.
3. **Cross-attention.** Pass the conditioning as a *sequence* of vectors (e.g. one vector per text token). Add cross-attention layers in the U-Net where the image queries the conditioning sequence. This is how text conditioning works in Stable Diffusion / SDXL — text → CLIP/T5 tokens → cross-attention in the U-Net. (**T5** is Google's text encoder, used in Flux and SD3 for better long-prompt understanding than CLIP.)

The three patterns aren't mutually exclusive: a typical text-to-image U-Net uses (2) for time, (3) for text, and could use (1) on top if you also pass a depth map.

## What it's good at

- **Inductive bias for images** — "inductive bias" means assumptions baked into the architecture before any training. Convolutions assume the same filter should work anywhere on the image (translation-equivariance). The U-shape assumes things matter at multiple scales. Both are true for images, so U-Nets train efficiently from less data than transformers.
- **Memory-efficient** — convolutions scale linearly in pixel count; you don't pay quadratic cost across all pixels.

## Where it struggles

- **Long-range coherence** — without attention everywhere, a hand on the left side of the image and a hand on the right side of the image might not match. The model literally hasn't compared them.
- **Scaling** — U-Net's empirical scaling laws (how much quality you get per extra parameter) are messier than transformers; doubling parameters doesn't reliably double quality.
- **Hard to add modalities** — each new condition (depth, pose, sketch) needs new architectural plumbing. **ControlNet** was invented specifically to bolt new conditioning onto a frozen U-Net without retraining it.

## U-Net vs DiT

| Aspect | U-Net (SD 1.5 / SDXL) | DiT (Flux / SD3 / Wan 2.2) |
|--------|------------------------|-----------------------------|
| Inductive bias | Translation-equivariant, multi-scale | Minimal — patches and attention |
| Parameter scaling | Diminishing returns past ~5B | Clean log-linear scaling |
| Compute pattern | Convs (linear in pixels) + attention at low res | Attention dominant — quadratic in tokens |
| Conditioning | Cross-attention plumbing per modality | Tokens concatenated/cross-attended cleanly |
| Memory | Low at high resolution | High at high resolution |

Current consensus: U-Net is fine for smaller models and limited compute; DiT wins at scale.

See [[dit]] for the alternative.

## See Also

- [[diffusion-models]]
- [[dit]]
- [[ddpm]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[latent-diffusion]]
- [[foundations]]
