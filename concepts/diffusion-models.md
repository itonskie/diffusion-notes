---
type: concept
tags:
  - foundations
  - diffusion
  - generative
date: 2026-06-29
updated: 2026-06-29
---

# Diffusion Models

The picture: you have a clean image. You add a tiny bit of random noise to it. Then a tiny bit more. Then more. Keep doing this and after enough steps the image is indistinguishable from pure static — every pixel a random number. That's the **forward process**, and it's not learned; it's just "add noise on a schedule."

A diffusion model learns to run that movie **in reverse.** Given a noisy image, predict what got added so it can be undone. Start from pure static, ask the model to denoise a little, get something slightly less noisy. Ask it again. Again. After enough steps, you're holding a brand-new image that came out of nothing but noise and the model's learned sense of "what real images look like."

That's a diffusion model. A [[generative-models|generative model]] (one that produces new data) that works by reversing a noising process one small step at a time.

## Why iterate instead of one-shot?

The obvious question: why not just train a model to spit out a clean image directly from noise?

Because **that's a brutally hard prediction.** "Here's pure noise, give me a photorealistic cat" — the model has to nail every pixel, every shape, every shadow in a single forward pass. Errors stack up everywhere with no chance to correct them.

"Here's mostly-clean noise, give me a slightly cleaner version" is enormously easier. Each step is a small, local correction. Mistakes early on (when the input is mostly noise and the model's guess is rough) get cleaned up by later steps. The iteration trades difficulty for accuracy.

This makes diffusion structurally different from one-shot generators like GANs (the adversarial generator-vs-discriminator setup) or autoregressive samplers (predict one token/pixel at a time given all the prior ones). The trade-off: many denoising steps = slower inference. That cost is the whole reason the field obsesses over fast [[scheduler|schedulers]] — the algorithms that decide how big each denoising step is — and step-reducing tricks like distillation (training a small fast student model to copy a big slow teacher in fewer steps). DPM++, Euler, and LCM are all named schedulers you'll see in ComfyUI. More in [[scheduler]].

## How training works

Surprisingly mechanical:

1. Load a batch of images from the training data.
2. Add noise — **in different amounts** to different images in the batch.
3. Feed the noisy inputs into the model along with how much noise was added.
4. Have the model predict the noise (or equivalently, the clean image — depends on what the code calls **parameterization**, i.e. which target you ask the model to predict).
5. Compute loss against the true noise. Backprop. Update weights.

The "different amounts" part matters and tripped me up first time I saw it. Why not just train on one consistent noise level? Because at sampling time the model has to handle every noise level — from "almost pure noise" at the start of generation down to "almost clean" at the end. If you only trained it on lightly-noised images, it'd be useless at the start of sampling. Only heavily-noised, useless at the end. So you sample the noise level uniformly at random per training example. Every gradient update teaches it some part of the noise spectrum.

The forward (noise-adding) process is fixed math, not learned. Only the reverse (denoising) process is learned. The actual math of why this whole setup is mathematically sound is the contribution of [[ddpm|DDPM]] (Denoising Diffusion Probabilistic Models, the 2020 paper that made this work at scale).

## How sampling works

Sampling = the reverse of training, running for real.

1. Start with pure random noise of the target output shape (e.g. a 1024×1024×3 tensor of Gaussian numbers).
2. Feed it through the trained model. Get back a prediction of the noise.
3. Use that prediction to take a small step toward "less noisy."
4. Repeat until the noise level is approximately zero.

How big each step is, and whether each step also adds a tiny bit of fresh randomness (stochasticity) or is fully deterministic, is the job of the [[scheduler]]. This is where the same trained model can produce wildly different results at 4 steps vs 1000 steps, depending on which scheduler you pick.

## Why it works well in practice

A few properties that come almost for free with this design:

- **Training is stable.** No adversarial dynamics like GANs have. The loss is just "predict the noise correctly." Smooth, well-behaved.
- **Sample diversity is good.** Every generation starts from a different random noise tensor, so the model isn't locked onto a single mode of the distribution.
- **The architecture is decoupled from the algorithm.** You can swap [[u-net|U-Net]] (the older convolutional backbone shaped like a U with downsample-then-upsample stages) for [[dit|DiT]] (Diffusion Transformer, the modern Transformer-based backbone) without touching the training loop or sampling code.
- **It scales.** The same recipe trains a 10M-parameter toy model on MNIST and a 12B-parameter Flux model on the public internet's worth of images.

## Connection to score matching (math, optional)

This is the math section. You can skip it on a first read; the practical picture above is enough to use the tools. But you'll see "score" used everywhere in papers, so a quick decode:

The model's noise prediction, written `ε_θ(x_t, t)`, can be rewritten — up to a known scaling — as predicting something called the **score**, written `∇_{x_t} log p(x_t)`.

What each symbol means in words:

- `ε_θ(x_t, t)` — the model (with learned parameters `θ`) given a noisy image `x_t` at noise level `t`, returning its prediction of the noise inside it.
- `p(x_t)` — the probability of seeing a particular noisy image at that noise level.
- `log p(x_t)` — the log of that probability.
- `∇_{x_t}` — gradient with respect to the image pixels. "If I nudge each pixel a tiny bit, how does the log-probability change?"
- So `∇_{x_t} log p(x_t)`, the **score**, is a vector pointing in the direction that makes the noisy image more probable — i.e. more like a real image.

The relationship:

```
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)
```

Where `ᾱ_t` (alpha-bar-t) is a number from the noise schedule that says how much of the original image is still present at noise level `t`.

The takeaway: a diffusion model is also a **score model** at many noise levels simultaneously. Once you frame it that way, sampling can be derived as Langevin dynamics over learned scores — a method from physics for sampling from a probability distribution by following its gradient with a bit of noise added. This is the "score-based" or "SDE" (Stochastic Differential Equation — a differential equation with random noise terms) framing from Song et al. The noise-prediction "DDPM" framing and the score-based "SDE" framing are mathematically equivalent. Both views are useful: SDE is cleaner for theory, ε-prediction is cleaner for code.

## What this page does not cover

- The actual math of the forward/reverse process — see [[ddpm]].
- The variance schedule `β_t` (beta-t, the per-step noise variance) and the difference between training-time noise schedules and inference-time samplers — see [[scheduler]].
- Conditioning the model on text or class labels — see [[classifier-free-guidance]] and Unit 2 of the [[reference/diffusion-course]].
- Latent diffusion (Stable Diffusion / Flux), which runs the entire diffusion process in compressed [[latent-diffusion|latent space]] — a small numerical representation of the image produced by a VAE — instead of in raw pixels, for huge speed and memory wins.

## See Also

- [[generative-models]]
- [[ddpm]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[u-net]]
- [[dit]]
- [[reference/diffusion-course]]
- [[foundations]]
