---
type: concept
tags:
  - foundations
  - generative
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Generative Models

The class of algorithms that learn to **produce** new data — images, audio, text — given a corpus of training examples. The output should resemble the training distribution without being a copy.

A generative model is judged on two axes:

- **Fidelity** — do the outputs look like the training data? (A model that produces noise fails here.)
- **Diversity** — does it cover the breadth of the training distribution? (A model that memorizes one image looks high-fidelity but is diversity-collapsed.)

A model that produces exact copies of training examples is not a good generative model — it's a database. The point is to learn the *distribution*, not the *samples*.

## Major families

| Family | Mechanism | Notes |
|--------|-----------|-------|
| Autoregressive | Predict next token / pixel conditioned on the prior ones | LLMs are the canonical example |
| GAN | Generator vs discriminator adversarial game | Sharp images, training instability, mode collapse risk |
| VAE | Encoder → latent → decoder, trained with reconstruction + KL loss | Smooth latent space, often blurry outputs |
| Normalizing flows | Invertible transforms with tractable log-likelihood | Exact density, but architecturally constrained |
| Diffusion | Iteratively denoise from random noise back to data | Stable training, slow sampling, currently dominant for image/video |

## Where diffusion fits

[[diffusion-models|Diffusion models]] are the current frontier for image and video generation (Flux, SDXL, SD3, Wan 2.2). They beat GANs on diversity and stability, and beat VAEs on fidelity. The trade-off they accept is slow sampling — each output requires many forward passes through the model — which is why so much research and tooling exists around [[scheduler|schedulers]] and step-reducing techniques.

## See Also

- [[diffusion-models]]
- [[foundations]]
