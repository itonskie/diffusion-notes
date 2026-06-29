---
type: paper
tags:
  - foundations
  - diffusion
  - ddpm
  - paper
date: 2026-06-29
updated: 2026-06-29
status: exploring
---

# DDPM — Denoising Diffusion Probabilistic Models

> Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
> https://arxiv.org/abs/2006.11239

## Status

**Exploring** — primary content via Lilian Weng's [survey post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Original paper not yet read end-to-end. The math below is from that survey, which derives DDPM's contribution rigorously.

## Why it matters

DDPM is the paper that formalized the forward/reverse noising process for [[diffusion-models|diffusion models]] and made them competitive with GANs on image quality. Almost every later diffusion paper (Stable Diffusion, Flux, SD3, [[dit|DiT]]) builds on this formulation.

## Forward process

Add Gaussian noise over T steps with variance schedule `{β_t}`:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

With `α_t = 1 - β_t` and `ᾱ_t = ∏_{i=1}^t α_i`, you can jump directly to any noise level:

```
x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

This closed-form jump is what makes training tractable — every minibatch samples a random t and gets the noisy version in O(1).

The original schedule is **linear**: `β_1 = 10⁻⁴`, `β_T = 0.02`, `T = 1000`. See [[scheduler]] for the cosine improvement.

## Reverse process

Generate by reversing the chain:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

DDPM's key parameterization choice: predict noise `ε`, not means. The mean is then:

```
μ_θ(x_t, t) = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t))
```

Variance is **fixed, not learned** — set to either `β_t` or `β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t`. Learning the variance was found to destabilize training; [[papers/improved-ddpm|Improved DDPM]] later fixed this.

## The simplified training objective

The full variational lower bound has many KL terms. The paper's key empirical finding: **dropping the weighting and training a plain L2 on noise prediction works better**:

```
L_simple = E_{t, x_0, ε} [ ||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||² ]
```

That's it. Sample a clean image, sample a random t, sample noise, add it, ask the model to predict the noise, take MSE. The whole forward and reverse process formalism collapses to a one-line MSE loss in practice.

## Training algorithm (Algorithm 1 in paper)

```
repeat
  x_0 ~ q(x_0)                   # sample clean image
  t ~ Uniform({1, ..., T})       # random noise level
  ε ~ N(0, I)                    # random noise
  Take gradient step on ||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||²
until converged
```

## Sampling algorithm (Algorithm 2)

```
x_T ~ N(0, I)
for t = T, ..., 1:
  z ~ N(0, I) if t > 1 else 0
  x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t)) + √(β̃_t) · z
return x_0
```

The `z` term is the **stochastic** part of "ancestral sampling." [[ddim|DDIM]] later showed you can set `z = 0` (deterministic) and get equivalent quality with fewer steps.

## Connection to score matching

Define the score function `s(x) = ∇_x log p(x)`. The noise prediction relates directly:

```
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)
```

So DDPM is (up to a reweighting) equivalent to training a score model at multiple noise levels. This connects to Song & Ermon's NCSN work and to the broader **score-based / SDE** framework. Sampling becomes a Langevin-dynamics-like procedure.

## What to extract when reading the original paper

- The full derivation of the variational lower bound and why each term is what it is — Lilian Weng's post sketches this; the paper does it fully.
- The empirical comparison of ε-prediction vs μ-prediction vs x₀-prediction.
- The choice of `T = 1000` and why — practical compute vs convergence trade-off.
- Architectural details of the [[u-net|U-Net]] used (channel counts, time embedding, attention placement).

## Open questions

- Why does ε-prediction work better than x₀-prediction empirically? (Theoretical answer: noise has unit variance regardless of `t`, so the regression target is well-scaled; predicting x₀ from noisy x_t requires huge dynamic range at high t.)
- What's lost by dropping the VLB weighting? (Answer: tight likelihood bounds, but sample quality is better with the simplified loss.)

## See Also

- [[diffusion-models]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[ddim]]
- [[improved-ddpm]]
- [[u-net]]
- [[foundations]]
