---
type: concept
tags:
  - foundations
  - diffusion
  - generative
date: 2026-06-29
updated: 2026-06-29
status: exploring
---

# Diffusion Models

A class of [[generative-models|generative model]] that learns to produce data by reversing a step-by-step noising process. Generation starts from pure random noise and is refined into a coherent output across many small denoising steps.

## The core insight: iterative denoising

A diffusion model does not try to generate an image in one shot. Instead, generation is iterative:

1. Start from pure random noise.
2. Predict a slightly less noisy version.
3. Apply that small denoising step.
4. Repeat — often dozens to thousands of times, depending on the [[scheduler]].

Why iterate? Because **predicting the final clean output from pure noise is extremely hard**. Predicting a slightly less noisy version of the current input is much easier. Errors early on (when the input is mostly noise and the model's estimate is bad) get corrected by later steps. The iteration is what trades difficulty for accuracy.

This makes diffusion models structurally different from one-shot generators like GANs or autoregressive samplers. The trade-off: many denoising steps means slower inference. That cost is the whole reason fast schedulers (DPM++, Euler, LCM) and step-reducing techniques like distillation matter — see [[scheduler]].

## How training works

The training loop is straightforward:

1. Load a batch of images from the training data.
2. Add noise — **in different amounts**. The model has to be good at denoising both barely-noisy images and almost-pure-noise images.
3. Feed the noisy inputs into the model.
4. Have the model predict the noise (or the clean image, depending on parameterization).
5. Compute loss against the true noise and update the model weights.

The "different amounts" matters: training only on lightly noised images would leave the model useless at the start of sampling; training only on heavily noised images would leave it useless at the end. The noise level is sampled randomly per training example.

The forward (noise-adding) process is fixed and not learned. Only the reverse (denoising) process is learned. The math behind why this works is the contribution of [[ddpm|DDPM]].

## How sampling works

Generation reverses the training direction:

1. Start with pure random noise of the target output shape.
2. Feed it through the trained model.
3. Use the model's prediction to take a small step toward "less noisy."
4. Repeat until the noise level is approximately zero.

The exact step size, schedule, and stochasticity at each step is the job of the [[scheduler]]. Different schedulers trade quality for speed — some need 1000 steps, some need 4.

## Why it works well in practice

- **Training is stable** compared to GANs (no adversarial dynamics).
- **Sample diversity is good** — the model isn't tied to a single mode, because every step starts from a different noise sample.
- **The model architecture is decoupled from the generative process** — you can swap [[u-net]] for [[dit]] without changing the training algorithm.
- **It scales** — the same recipe trains a 10M-parameter toy model on MNIST and a 12B-parameter Flux model on web-scale data.

## What this page does not cover

- The actual math of the forward/reverse process — see [[ddpm]].
- Conditioning the model on text or class labels — see [[classifier-free-guidance]] and Unit 2 of the [[reference/diffusion-course]].
- Latent diffusion (Stable Diffusion / Flux), which runs the diffusion process in compressed latent space rather than pixel space — covered in later units.

## See Also

- [[generative-models]]
- [[ddpm]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[u-net]]
- [[dit]]
- [[reference/diffusion-course]]
- [[foundations]]
