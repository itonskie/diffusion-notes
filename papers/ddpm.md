---
type: paper
tags:
  - foundations
  - diffusion
  - ddpm
  - paper
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# DDPM — Denoising Diffusion Probabilistic Models

> Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
> https://arxiv.org/abs/2006.11239 • [PDF](https://arxiv.org/pdf/2006.11239) • [code](https://github.com/hojonathanho/diffusion)

## Status

**Developing** — abstract and key results from arxiv + ar5iv HTML version; math and intuition from Lilian Weng's [survey post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Have not deep-read the appendix yet (missing: full U-Net config, optimizer/LR, EMA decay).

## Abstract (verbatim)

> "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding."

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

## Architecture (from Appendix B references)

- **Backbone**: [[u-net|U-Net]] "similar to an unmasked PixelCNN++" with group normalization throughout
- **Time embedding**: Transformer sinusoidal position embedding
- **Self-attention**: at the 16×16 feature map resolution
- **Residual blocks**: PixelCNN++ style

Full channel counts and depth are in Appendix B (not extracted here).

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| T (timesteps) | 1000 |
| β₁ | 10⁻⁴ |
| β_T | 0.02 |
| Schedule | linear interpolation between β₁ and β_T |

Batch size, optimizer, learning rate, EMA decay — not yet extracted.

## The key ablation (Table 2)

Comparing parameterization and loss on CIFAR-10:

| Approach | IS ↑ | FID ↓ |
|----------|------|-------|
| μ̃ prediction + L bound, fixed Σ | 8.06 ± 0.09 | 13.22 |
| ε prediction + L bound, fixed Σ | 7.67 ± 0.13 | 13.51 |
| ε prediction + L_simple objective | **9.46 ± 0.11** | **3.17** |

**The headline takeaway**: predicting noise ε with the simplified L2 objective beats predicting means *or* using the full weighted variational bound, by a large margin. This single result is the practical contribution that made diffusion competitive.

## Headline results

| Dataset | Resolution | Metric | Value |
|---------|------------|--------|-------|
| CIFAR-10 (train) | 32×32 | FID | 3.17 |
| CIFAR-10 (train) | 32×32 | IS | 9.46 |
| CIFAR-10 (test) | 32×32 | FID | 5.24 |
| LSUN Church | 256×256 | FID | 7.89 |
| LSUN Bedroom | 256×256 | FID | 4.90 |

The CIFAR FID was state-of-the-art at the time and matched ProgressiveGAN on LSUN 256×256.

## Progressive lossy decompression

The paper frames diffusion sampling as a kind of progressive decoding: at any intermediate `t`, the receiver has `x_t` (a noisy partial estimate) and can keep refining it. This connects to **rate-distortion** analysis — and the paper shows that "the majority of the bits are indeed allocated to imperceptible distortions." That is, most of the sampling steps refine perceptually-small details; the broad structure is locked in early.

This is a useful intuition for why early steps matter more than late ones, and why techniques like CFG-skipping (only apply CFG on certain steps) can save compute without much quality loss.

## Score matching connection (Section 3.2)

The paper makes the equivalence explicit: training with the ε-prediction objective "resembles denoising score matching over multiple noise scales indexed by t," and sampling "resembles Langevin dynamics with ε_θ as a learned gradient of the data density." So DDPM and the score-based / SDE framework (Song et al.) are mathematically equivalent — they're two perspectives on the same algorithm.

This is why later papers (Song et al. "Score-Based Generative Modeling Through Stochastic Differential Equations," 2020) could unify them into a single continuous-time SDE framework.

## What to still extract (when re-reading appendix)

- Full U-Net config (channel multipliers, depth, where attention is exactly).
- Optimizer (Adam? lr? warmup? weight decay?).
- EMA decay rate (these papers almost always use EMA for sampling).
- Total training iterations and batch size.

## Open questions

- Why does ε-prediction work better than x₀-prediction empirically? (Likely: noise has unit variance regardless of `t`, so the regression target is well-scaled; predicting x₀ from noisy x_t requires huge dynamic range at high t.)
- What's lost by dropping the VLB weighting? (Answer: tight likelihood bounds, but sample quality is better with the simplified loss — confirmed by Table 2.)

## See Also

- [[diffusion-models]]
- [[scheduler]]
- [[classifier-free-guidance]]
- [[ddim]]
- [[improved-ddpm]]
- [[u-net]]
- [[foundations]]
