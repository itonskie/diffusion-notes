---
type: paper
tags:
  - foundations
  - diffusion
  - ddpm
  - paper
date: 2026-06-29
updated: 2026-06-29
status: summary
---

# DDPM — Denoising Diffusion Probabilistic Models

> Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
> https://arxiv.org/abs/2006.11239 • [PDF](https://arxiv.org/pdf/2006.11239) • [code](https://github.com/hojonathanho/diffusion)

## Status

**Developing** — abstract and key results from arxiv + ar5iv HTML version; math and intuition from Lilian Weng's [survey post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Have not deep-read the appendix yet (missing: full U-Net config, optimizer/LR, EMA decay).

## What this paper actually claims

Take a clear image. Gradually sprinkle Gaussian noise onto it, step after step, until after 1000 steps it's pure static — no signal left. Now train a neural network to do the *reverse*: given the static-looking input at step `t`, predict what noise was added so we can subtract it and get back to step `t-1`. Apply that 1000 times starting from random static and you have a brand-new image.

That's the whole technique. The paper's specific contributions:

1. A clean math framework (forward = adding noise, reverse = a learned denoiser) that you can derive from variational inference.
2. The empirical finding that **predicting the noise itself**, with a plain MSE loss, works better than the fancier objectives the framework suggests.
3. Sample quality competitive with GANs (the previous SOTA for image generation) at the time.

Almost every later diffusion paper — Stable Diffusion, Flux, SD3, [[dit|DiT]] — builds on this exact formulation.

> **Domain decode first.** A few words you need throughout: **Gaussian noise** = random numbers drawn from a normal (bell-curve) distribution; **diffusion model** = a generative model that learns to reverse a noising process; **timestep `t`** = how many noise-adding steps have happened, ranging from 0 (clean image) to T=1000 (pure noise); **variance schedule** = the recipe that decides how much noise to add at each step; **U-Net** = the workhorse convolutional architecture used to predict the noise — see [[u-net]]; **EMA** = exponential moving average of the model's weights, used at sampling time because it's smoother than the live weights.

## Abstract (verbatim)

> "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding."

## Why it matters

DDPM formalized the forward/reverse noising process for [[diffusion-models|diffusion models]] and made them competitive with GANs on image quality. Before this paper, diffusion existed as a niche idea from 2015; after it, diffusion swallowed image generation.

## The forward process: ruin an image on purpose

The forward process is not learned. It's a fixed recipe: at each step `t`, take whatever image you've got and mix in a little Gaussian noise. After enough steps, the image is indistinguishable from pure static.

The clever part is the math: because each step is Gaussian, you can jump from a clean image at step 0 directly to the noisy version at any step `t` in one shot — no need to actually loop. This is what makes training tractable, because every minibatch can sample a random `t` and get the noisy version in O(1) instead of running a 1000-step simulation.

The schedule in the paper is **linear**: noise is added gently at the start (very little change between adjacent steps near `t=0`) and harder near the end. See [[scheduler]] for the cosine improvement that came later.

## The reverse process: learn to walk back

The reverse process is what you actually train. Starting from pure noise at step `T`, you want a learned function that takes the noisy image at step `t` and produces a slightly-less-noisy version at step `t-1`. Do that 1000 times and you've reached step 0 — a generated image.

The paper's pivotal design choice: don't parametrize the network to predict the cleaner image directly. Predict **the noise that was added**. Then subtract a fraction of it to walk one step back. The math works out cleanly because noise has unit-ish variance regardless of `t`, so the regression target is always well-scaled — whereas predicting the clean image from very noisy input would force the network into a huge dynamic range.

The variance of each reverse step is **fixed, not learned**. The paper found that trying to learn it destabilized training; [[papers/improved-ddpm|Improved DDPM]] later fixed this.

## The simplified training objective

In theory, you derive the objective from a variational lower bound (VLB) — a standard recipe in latent-variable models that gives you a weighted sum of KL-divergence terms across all timesteps. In practice, the paper's headline empirical finding is: **drop the weighting, drop the KL terms, just take MSE between predicted noise and true noise. It works better.**

The whole training loop collapses to: sample a clean image, sample a random `t`, sample noise, add it, ask the model to predict the noise, take MSE. Done. The complicated formalism is the justification; the actual code is a few lines.

## Math sketch

### Forward process

Add Gaussian noise over `T` steps with variance schedule `{β_t}`:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

- `x_t` — the noisy image at step `t`
- `q(x_t | x_{t-1})` — the distribution from which the next-step image is drawn
- `N(mean, variance)` — a Gaussian (normal) distribution
- `β_t` — how much noise to inject at step `t` (the variance schedule)
- `I` — the identity matrix; "noise added is independent per pixel"

With `α_t = 1 - β_t` (shorthand) and `ᾱ_t = ∏_{i=1}^t α_i` (the running product over all earlier steps), the closed-form jump to any noise level becomes:

```
x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

- `x_0` — the clean image
- `ε` — a single draw of standard Gaussian noise (mean 0, variance 1)
- The two `√` coefficients control the mix: as `t → T`, `ᾱ_t → 0`, so `x_t` is almost entirely noise

This closed-form jump is what makes training tractable — every minibatch samples a random `t` and gets the noisy version in O(1).

The original schedule is **linear**: `β_1 = 10⁻⁴`, `β_T = 0.02`, `T = 1000`. See [[scheduler]] for the cosine improvement.

### Reverse process

Generate by reversing the chain:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

- `p_θ` — the learned reverse distribution (`θ` are the network's weights)
- `μ_θ(x_t, t)` — the mean, predicted by the network
- `Σ_θ(x_t, t)` — the variance (in DDPM, fixed not learned)

DDPM's key parameterization choice: have the network output `ε`, not means. Then derive the mean from `ε` algebraically:

```
μ_θ(x_t, t) = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t))
```

- `ε_θ(x_t, t)` — the network's noise prediction
- The rest is just algebra: "given my noise estimate, what mean is implied?"

Variance is **fixed, not learned** — set to either `β_t` or `β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t`. Learning the variance was found to destabilize training; [[papers/improved-ddpm|Improved DDPM]] later fixed this.

### The simplified training objective

The full variational lower bound has many KL terms. The paper's key empirical finding: **dropping the weighting and training a plain L2 on noise prediction works better**:

```
L_simple = E_{t, x_0, ε} [ ||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||² ]
```

- `E_{t, x_0, ε}` — "expectation over random samples of t, clean image, and noise"
- `|| · ||²` — squared L2 norm, i.e. mean squared error
- The thing inside the prediction is just `x_t` written out

That's it. Sample a clean image, sample a random t, sample noise, add it, ask the model to predict the noise, take MSE. The whole forward and reverse process formalism collapses to a one-line MSE loss in practice.

### Training algorithm (Algorithm 1 in paper)

```
repeat
  x_0 ~ q(x_0)                   # sample clean image
  t ~ Uniform({1, ..., T})       # random noise level
  ε ~ N(0, I)                    # random noise
  Take gradient step on ||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||²
until converged
```

### Sampling algorithm (Algorithm 2)

```
x_T ~ N(0, I)
for t = T, ..., 1:
  z ~ N(0, I) if t > 1 else 0
  x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t)) + √(β̃_t) · z
return x_0
```

The `z` term is the **stochastic** part of "ancestral sampling" — at each reverse step you don't just subtract noise, you also add a small fresh noise kick. This is what makes the process probabilistic instead of deterministic. [[ddim|DDIM]] later showed you can set `z = 0` (deterministic) and get equivalent quality with fewer steps.

## Connection to score matching

There's a parallel formulation of generative modeling called **score matching**, where you train a network to estimate the score function — `s(x) = ∇_x log p(x)`, which roughly means "for any point in image-space, which direction would make the data density there higher." Sample by walking up the score (Langevin dynamics).

It turns out DDPM is equivalent to score matching at multiple noise levels:

```
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)
```

So the noise prediction *is* (up to a known rescaling) a score estimate. This means DDPM and the score-based / SDE framework (Song et al.) are two perspectives on the same algorithm — which is why a year later Song et al. could unify them into a single continuous-time **SDE** (stochastic differential equation) framework.

> Decode: **Langevin dynamics** = a way to sample from a distribution by repeatedly taking a small step in the direction of higher density plus a random noise kick. **Score** = the gradient of log-density with respect to input; tells you which way to move to make the input "more likely."

## Architecture (from Appendix B references)

- **Backbone**: [[u-net|U-Net]] "similar to an unmasked PixelCNN++" with group normalization throughout
- **Time embedding**: Transformer sinusoidal position embedding (the same trick transformers use to encode position; here it's used to tell the network which timestep `t` it's currently denoising)
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

**The headline takeaway:** predicting noise `ε` with the simplified L2 objective beats predicting means *or* using the full weighted variational bound, by a large margin. This single result is the practical contribution that made diffusion competitive.

> Decode: **IS** (Inception Score) measures sample quality and class confidence — higher is better. **FID** (Fréchet Inception Distance) measures how close the distribution of generated samples is to real data — lower is better. Both use a pretrained Inception classifier as the judge.

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

The paper makes the equivalence explicit: training with the ε-prediction objective "resembles denoising score matching over multiple noise scales indexed by t," and sampling "resembles Langevin dynamics with `ε_θ` as a learned gradient of the data density." So DDPM and the score-based / SDE framework (Song et al.) are mathematically equivalent — they're two perspectives on the same algorithm.

This is why later papers (Song et al. "Score-Based Generative Modeling Through Stochastic Differential Equations," 2020) could unify them into a single continuous-time SDE framework.

## What this paper changed downstream

Without DDPM there's no Stable Diffusion, no Flux, no SD3. The "predict noise, MSE loss, U-Net backbone" recipe is the spine of every major image diffusion model. Specific things that came directly from this paper:

- The noise-prediction parameterization (still used by Flux, SD3, SDXL)
- The ε / x₀ / v parameterization debate downstream (everyone starts from "what DDPM did")
- The training loop most fine-tuning tools use (sample t, add noise, MSE)
- The intuition that early sampling steps matter more than late ones

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
