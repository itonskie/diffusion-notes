---
type: concept
tags:
  - foundations
  - generative
date: 2026-06-29
updated: 2026-06-29
---

# Generative Models

A generative model is the family of algorithms whose job is to **make new stuff that looks like the stuff you trained it on**. Show it ten thousand cat photos; it should hand you a brand-new cat photo that no camera ever took. Show it the entire internet's text; it should write a sentence that was never written. Same idea for audio, video, 3D meshes, protein structures.

The thing you're really trying to learn isn't the specific training examples — it's the *distribution* they came from. The "shape" of what a cat photo can be. If you only spit back exact copies of training images, you didn't build a generative model, you built a database with extra steps.

## The two things we judge it on

Any time you look at a generated sample, you're asking two separate questions:

- **Fidelity** — does this output actually look like the training data? A model that hands you static noise fails here.
- **Diversity** — does it cover the *breadth* of what the training data contained? A model that has memorized one really good cat photo and hands you that same photo every time looks high-fidelity but is **mode-collapsed** — collapsed onto a single mode of the distribution, ignoring the rest.

Both matter. Optimizing for only one is the classic way to ship a useless model.

## The major families

Different bets on *how* to learn that distribution. Each row is a different mechanism.

| Family | Mechanism | Notes |
|--------|-----------|-------|
| Autoregressive | Predict next token / pixel conditioned on the prior ones | LLMs are the canonical example |
| GAN | Generator vs discriminator adversarial game | Sharp images, training instability, mode collapse risk |
| VAE | Encoder → latent → decoder, trained with reconstruction + KL loss | Smooth latent space, often blurry outputs |
| Normalizing flows | Invertible transforms with tractable log-likelihood | Exact density, but architecturally constrained |
| Diffusion | Iteratively denoise from random noise back to data | Stable training, slow sampling, currently dominant for image/video |

Quick decode of the jargon in that table, since most of it is field-specific:

- **Autoregressive** — generate one piece at a time, each piece conditioned on what you generated before it. GPT generating one token at a time is the obvious example. Slow but principled.
- **GAN** (Generative Adversarial Network) — train two networks against each other. A **generator** makes fakes; a **discriminator** tries to tell fakes from real training images. They improve together. Famous for sharp images and for being a nightmare to train stably.
- **Mode collapse** — when the generator finds one or two outputs the discriminator can't tell from real and just produces those forever. Bad diversity.
- **VAE** (Variational Autoencoder) — squash the image into a small **latent** (a compact numerical fingerprint), then learn to expand a latent back into an image. Plus a constraint that the latent space is smooth and well-shaped. Good for editing, infamously blurry on its own.
- **KL loss** — Kullback-Leibler divergence. A measure of how different two probability distributions are. In a VAE it's the regularizer pulling the latent space into a clean shape.
- **Normalizing flow** — build the generator out of fully invertible math, so you can both sample from it and exactly compute the probability of any data point under it. Mathematically tidy, architecturally restrictive.
- **Diffusion** — start from pure noise and learn to undo noise step by step until you're back to an image. The one we actually care about for this sprint.

## Where diffusion fits

[[diffusion-models|Diffusion models]] are what's currently winning for image and video generation — Flux, SDXL, SD3, Wan 2.2 are all diffusion under the hood. Compared to the alternatives:

- vs **GANs** — diffusion wins on diversity and is dramatically more stable to train. No adversarial dynamics blowing up at step 40k.
- vs **VAEs** — diffusion wins on fidelity. Sharper, more detailed outputs.

The price diffusion pays: **sampling is slow.** Generating one image requires many forward passes through the model — dozens, sometimes hundreds. That's why so much of the tooling and research around diffusion is about making sampling faster: smarter [[scheduler|schedulers]] (the algorithms that decide how big each denoising step should be), step-reducing techniques like distillation, etc.

The bet diffusion makes is that trading one big hard problem ("generate an image from nothing") for many small easy problems ("clean up this slightly noisy image a tiny bit") gives you stability and quality, and that you can spend engineering effort recovering speed later. So far the bet is winning.

## See Also

- [[diffusion-models]]
- [[foundations]]
