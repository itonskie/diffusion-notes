---
type: concept
tags:
  - foundations
  - lora
  - fine-tuning
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# LoRA (Low-Rank Adaptation)

The default [[fine-tuning|fine-tuning]] technique for modern diffusion models. Replaces full fine-tuning by training tiny rank-`r` adapters on top of frozen base weights. A typical Flux LoRA is 100-300 MB vs Flux base at 12 GB — 40-100× smaller, 100× cheaper to train, and at inference adds zero latency once merged.

For the original paper, technique formalism, and NLP results, see [[papers/lora]]. This page focuses on the **diffusion-specific application** that this sprint actually does.

## How it works (one paragraph)

For each chosen weight matrix `W₀` in the base model, freeze it and train two small matrices `B` (`d × r`) and `A` (`r × k`). At inference, the effective weight is `W = W₀ + (α/r) · BA`. Because `r ≪ d, k`, the trainable matrices are tiny. Initialization sets `B = 0` so the adapter is a no-op at the start of training; gradients flow only through `A` and `B`. After training, you can either keep `B` and `A` separate (and combine them on-the-fly) or merge them once into `W` and discard them. See [[rank-decomposition]] for the math intuition on why low-rank is enough.

## Which layers do diffusion LoRAs adapt?

The original LoRA paper adapted only Q and V in attention. Diffusion LoRAs go broader:

| Tool | Default layer selection | Notes |
|------|-------------------------|-------|
| AI-Toolkit | All linear in attention + FFN, all conv in [[u-net]] | Most aggressive default |
| Kohya | Configurable; defaults to attention only | More conservative |
| Musubi Tuner | All linear in [[dit\|DiT]] blocks | Video diffusion |

Why broader? Diffusion fine-tuning has to teach the model new visual concepts, not just shift task behavior. Adapting only attention can underfit characters or specific styles.

## Choosing rank `r`

Rough rules of thumb for Flux LoRAs:

| Concept type | Typical `r` |
|--------------|-------------|
| Simple style transfer | 8-16 |
| Character / face | 16-32 |
| Complex style or multiple concepts | 32-64 |
| Very rich or photorealistic concept | 64-128 |

These are larger than the `r=1-8` that worked for NLP. Visual concepts have higher intrinsic dimensionality than text task adaptations.

**Diminishing returns** kick in fast — going from r=32 to r=64 usually buys little quality at 2× the file size. The right `r` is the smallest that still captures the concept.

## Alpha and LoRA strength

`α` controls the scaling at inference. Effective update is `(α / r) · BA`. The most common convention now:

- **Set `α = r` during training** → effective scale is 1
- **Adjust LoRA strength at inference** (a separate scalar applied at runtime) for tuning the LoRA's effect intensity

In ComfyUI, the LoRA loader's "strength" slider is essentially a multiplier on `α/r`. Strength 1.0 = trained intensity. Strength 0.5 = half effect. Strength 1.5 = amplified beyond what the LoRA learned. You can mix multiple LoRAs at independent strengths.

## Training cost (Flux LoRA, single 4090 or A6000)

| Concept | Steps | Wall time | Vast.ai cost |
|---------|-------|-----------|--------------|
| Small style (~15 images, r=16) | 1000 | 30-60 min | $0.50-$1.50 |
| Character (~25 images, r=32) | 2000 | 1-2 hr | $1-$3 |
| Rich style (~50 images, r=64) | 3000 | 3-5 hr | $3-$10 |

These ranges assume Flux-dev with sane settings. Add 30-50% if training Flux-schnell distilled or if using 512×512 instead of 1024×1024 latent resolution.

## Caption strategy for diffusion LoRAs

This is where most LoRA quality is won or lost. The captions you attach to each training image determine what the model learns.

- **Trigger words** — invent a unique token (e.g. `OHWX person` or `myartstyle`) and include it in every caption. At inference, prompting that token activates the LoRA's concept.
- **Describe what varies, NOT what's constant** — if every image is the same character in different scenes, caption the scene, not the character. The model will learn "constant = the LoRA concept; varying = compositional details I should respect."
- **Length** — long, descriptive captions tend to outperform short ones for Flux.

Bad captions are the #1 failure mode for Flux LoRAs. Many people use a vision-language model (Florence-2, Joy-Caption, BLIP-3) to auto-caption their dataset, then edit the captions to fix the trigger word and over/under-described elements.

## Merging vs runtime application

Two ways to apply a LoRA at inference:

- **Runtime** — keep `B` and `A` as separate tensors. At each forward pass, compute `W₀·x + (α/r)·B·A·x`. Adds a small matmul but lets you swap LoRAs instantly and apply multiple at once.
- **Merge** — compute `W₀ + (α/r)·BA` once and replace `W₀`. Pure inference is then identical to a fully fine-tuned model. Faster per-step but you can't swap LoRAs cheaply.

ComfyUI uses runtime application by default — that's why you can stack 3 LoRAs in a workflow.

## Sprint application

For Week 2 of the [diffusion sprint](https://github.com/itonskie/diffusion-sprint), the target is one trained Flux LoRA. The workflow is:

1. Pick concept (style, character, product) — Week 2 decision
2. Collect 15-50 images, write captions in `.txt` files
3. Use AI-Toolkit's example Flux LoRA config; change dataset path and output name
4. Train on Vast.ai (A6000 or 4090) for ~1-2 hours
5. Download `.safetensors`, load in ComfyUI, prompt with trigger word, compare with/without LoRA at the same seed

Configs that complete a run go into `configs/`. The trained `.safetensors` files do not get committed (already gitignored).

## See Also

- [[papers/lora]]
- [[rank-decomposition]]
- [[fine-tuning]]
- [[diffusion-models]]
- [[u-net]]
- [[dit]]
- [[foundations]]
