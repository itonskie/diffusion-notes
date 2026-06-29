---
type: paper
tags:
  - foundations
  - diffusion
  - ddpm
  - paper
date: 2026-06-29
updated: 2026-06-29
status: inbox
---

# DDPM — Denoising Diffusion Probabilistic Models

> Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* 2020.
> https://arxiv.org/abs/2006.11239

## Status

**Inbox** — referenced as the foundational paper in the [[reference/diffusion-course]] but not yet read in depth. The figure on the HF Unit 1 intro page is from this paper.

## Why it matters

DDPM is the paper that formalized the forward/reverse noising process for [[diffusion-models]] and made them competitive with GANs on image quality. Almost every later diffusion paper (Stable Diffusion, Flux, SD3, DiT) builds on this formulation.

## What to extract when reading

- The exact form of the forward (noise-adding) process — the Markov chain over noise levels.
- The reverse process parameterization — what the model actually predicts (noise ε vs clean image x₀ vs velocity).
- The training objective — why it simplifies to MSE on predicted noise.
- The connection to score-based models and Langevin dynamics (Song et al. — separate paper).
- Hyperparameters used in the original experiments (number of steps T = 1000, noise schedule shape).

## Open questions to resolve from this paper

- Why does the linear vs cosine noise schedule matter so much in practice?
- How does training-time T (often 1000) relate to inference-time step count (often 20-50 with modern [[scheduler|schedulers]])?

## See Also

- [[diffusion-models]]
- [[scheduler]]
- [[foundations]]
