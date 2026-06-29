---
type: paper
tags:
  - foundations
  - cfg
  - guidance
  - paper
date: 2026-06-29
updated: 2026-06-29
status: summary
---

# Classifier-Free Diffusion Guidance

> Ho & Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop on Score-Based Methods. (Posted on arxiv as 2207.12598 in 2022.)
> https://arxiv.org/abs/2207.12598 • [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2207.12598)

## Status

**Developing** — content from ar5iv HTML render.

## What this paper actually claims

You have a text-to-image diffusion model. You give it a prompt. By default it tries to generate an image that's *consistent* with the prompt — but "consistent" is a soft constraint, so the model often wanders and produces something only loosely related. You want a knob that says "really lean into the prompt" without having to retrain the model.

The earlier solution (**classifier guidance**, Dhariwal & Nichol 2021) added that knob by training a *separate* classifier `f(y|x_t, t)` that could score noisy images for class-membership, then using its gradient to push samples toward the desired class. That worked but required training and shipping an extra model, and the classifier had to handle noisy inputs at every timestep.

This paper's bet: you don't need a separate classifier at all. Just train *one* diffusion model that sometimes sees the prompt and sometimes sees nothing (a special "empty" token). At sampling time, run it twice per step — once with the prompt, once with nothing — and combine the two predictions to push toward the prompt. That difference acts like the classifier gradient, for free.

This is **classifier-free guidance** (CFG). It's the knob behind every "CFG scale" slider you've ever touched in ComfyUI or A1111.

> **Domain decode first.** Words you need: **conditioning** = feeding the model extra info (a class label, a text prompt, a class embedding) so it generates something specific instead of something random; **unconditional generation** = same model, but with no condition given (or a null/empty placeholder); **null token `∅`** = a specific learned embedding the model uses when "no condition" is given — it's not zero, it's a real learned vector; **score** = the gradient of log-density (which direction makes the data more likely); **log-SNR** = log signal-to-noise ratio, an alternate way of indexing the timestep `t` that's monotone and well-behaved.

## Abstract (verbatim)

> "Classifier guidance is a recently introduced method to trade off mode coverage and sample fidelity in conditional diffusion models post training... We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity."

## The picture, step by step

At each reverse-diffusion step, the model normally produces a single noise prediction `ε(prompt)` — "given this noisy image and this prompt, here's the noise I'd subtract." CFG runs the model twice instead:

1. With the prompt → `ε_cond`
2. With the null token (no prompt) → `ε_uncond`

Then it constructs the actual noise estimate it'll use as a linear combination:

```
ε̃ = ε_uncond + (CFG scale) · (ε_cond - ε_uncond)
```

Read it as: "start from the unconditional prediction, then add an exaggerated push in the direction the prompt is asking for." If CFG scale = 1, that exaggeration factor is 1 and you get back the conditional prediction. If CFG scale > 1, you're amplifying — going *further* in the prompt's direction than the model would on its own. That amplification is what gets the prompt to actually take effect.

The cost: you run the model twice per step instead of once. So sampling takes ~2× longer with CFG on than off — but that's why nobody turns it off, because the quality gap is huge.

## The contribution

Classifier guidance (Dhariwal & Nichol 2021) needs a separately-trained classifier `f(y|x_t, t)` that can score noisy images. That's an extra training step nobody wants to do.

CFG replaces it with a clever trick: train one diffusion model that handles both conditional and unconditional generation (by randomly dropping the condition with some probability during training). At inference, combine the two predictions linearly to push samples toward the conditional distribution.

This **eliminates the need for a separate classifier** and is the technique used by every modern text-to-image model.

## Training procedure (Algorithm 1)

Single network, conditioned via class embedding (or null token `∅` when dropping):

```python
for batch in data:
    if uniform_random() < p_uncond:
        condition = ∅          # null/unconditional token
    else:
        condition = batch.class
    loss = mse(model(x_t, t, condition), epsilon)
```

The unconditional case is **not zero**; it's a specific learned token (often the embedding of `∅`). The whole trick rests on the fact that one network learns to do both jobs (conditional and unconditional generation) and only differs in which embedding gets fed in.

### Dropout probability `p_uncond`

The paper tested `p_uncond ∈ {0.1, 0.2, 0.5}`:

- **0.1 and 0.2 perform equally well**
- **0.5 underperforms**

Intuition: if you drop the condition 50% of the time you're starving the conditional pathway of training signal. 10-20% leaves enough signal for both modes to be learned well.

Modern convention: 10-15% dropout during text-to-image training.

## Math sketch

The paper writes the combined prediction as:

```
ε̃_θ(z_λ, c) = (1 + w) · ε_θ(z_λ, c) - w · ε_θ(z_λ)
```

Symbols decoded:
- `ε_θ(z_λ, c)` — conditional noise prediction (model run with prompt `c`)
- `ε_θ(z_λ)` — unconditional noise prediction (model run with null prompt)
- `w` — **guidance strength** (paper's convention; not the same number as the ComfyUI CFG slider, see below)
- `λ` — **log-SNR**, a reparameterization of timestep `t`
- `z_λ` — the noisy image at log-SNR level `λ`

If you expand the algebra: `ε̃ = ε_uncond + (1+w) · (ε_cond - ε_uncond)`. Same form as the intuition picture above, where the "CFG scale" you see in tools is `1 + w`.

### Convention warning

The paper's `w` is **not** the same as the "CFG scale" you set in ComfyUI / A1111. The relationship:

```
ComfyUI CFG scale  ≈  paper's w + 1
```

| Paper's w | ComfyUI CFG | Behavior |
|-----------|-------------|----------|
| 0 | 1 | Pure conditional, no amplification |
| 0.3 | 1.3 | Slight guidance |
| 1 | 2 | Moderate guidance |
| 6 | 7 | Typical SDXL setting |
| 9 | 10 | Strong guidance |

When reading papers, subtract 1 from ComfyUI numbers to get paper-style `w`. When reading code, look at the exact formula being used.

## The implicit-classifier trick

Why does taking the difference of two noise predictions act like a classifier gradient? Bayes' rule on the score functions.

```
∇_x log p(c | x_t) ∝ ∇_x log p(x_t | c) - ∇_x log p(x_t)
                  ∝ ε_θ(x_t, c) - ε_θ(x_t)
```

Read in words: "the gradient that pushes `x_t` toward higher probability under class `c`" equals "the gradient of the class-conditional density" minus "the gradient of the marginal density." And both of those gradients are exactly what the diffusion model's two noise predictions estimate (via the score-matching connection from DDPM — noise prediction is a rescaled score estimate).

So `ε(cond) - ε(uncond)` plays the role of `∇ log f(c|x)` from classifier guidance — but it's free because both terms come from the same network.

**Caveat**: this only holds if the network's score estimates are exact (in math terms, conservative vector fields). In practice neural networks don't satisfy this exactly, so CFG is an *approximation* of true classifier guidance, not a strict equivalence. The fact that it works so well empirically is somewhat magical.

## The empirical tradeoff: FID vs IS vs `w`

CFG trades **diversity (FID-favoring)** for **fidelity (IS-favoring)**. The headline numbers from class-conditional ImageNet 128×128 (T=256 sampling steps):

| Paper's w | ComfyUI ~CFG | FID ↓ | IS ↑ |
|-----------|--------------|-------|------|
| 0.0 | 1.0 | 7.27 | 82.45 |
| 0.2 | 1.2 | 3.03 | 132.54 |
| 0.3 | 1.3 | **2.43** | 158.47 |
| 0.4 | 1.4 | 2.49 | 183.41 |
| 1.0 | 2.0 | 7.86 | 297.98 |
| 4.0 | 5.0 | 21.53 | **421.03** |

**Reading the table:**
- FID optimum is at `w ≈ 0.3` (CFG ≈ 1.3) — too little or too much guidance hurts FID
- IS keeps rising with `w` — strong guidance produces "more confidently classifiable" samples
- At `w = 4` (CFG ≈ 5) FID is **8.8× worse** than at `w = 0.3` but IS is **2.7× better**

This is the visual artifact people complain about: high CFG produces over-saturated, "burnt" colors and reduced diversity. Low CFG is more natural but prompts get ignored.

> Decode: **FID** (Fréchet Inception Distance) measures how close generated samples are to the real data distribution — lower is better, rewards diversity and naturalness. **IS** (Inception Score) measures how confidently a classifier can identify each sample's class — higher is better, rewards "looks like an obvious instance of its class." They reward different things; that's why one improves while the other gets worse.

### Why the optimal `w` differs from text-to-image practice

ImageNet has 1000 classes — relatively simple conditioning. Text-to-image conditioning is harder (millions of possible prompts), so models like SDXL need stronger guidance (CFG 7) to lock onto a specific prompt. Flux was specifically distilled to need less guidance (CFG ~3.5).

## Other findings

- **64×64 ImageNet, FID 1.55** at `w=0.1` (`v=0.3` log-SNR interpolation), 400k training steps
- The paper notes that `p_uncond` hyperparameters were inherited from classifier-guidance experiments and "may be suboptimal" for CFG
- Naïve alternatives (just truncating noise variance) cause blurry samples — CFG specifically operates on score combinations, which is what makes it work where simpler tricks fail

## What this paper changed downstream

Every text-to-image model in ComfyUI uses CFG at inference. The CFG slider IS this paper's `w + 1`. Understanding the FID/IS curve tells you:

- Why prompt adherence goes up with higher CFG (more IS bias)
- Why diversity goes down with higher CFG (FID degrades)
- Why "burnt" / over-saturated outputs come from CFG too high
- Why some workflows use higher CFG early in sampling and lower later (CFG scheduling)

Flux's lower-than-SD CFG default (3.5 vs 7) is partly because of post-training distillation that reduced the model's reliance on strong guidance, and partly because Flux was trained with an architecture (MMDiT — multi-modal DiT, image and text tokens share attention) that produces stronger conditioning signal natively.

## See Also

- [[classifier-free-guidance]]
- [[guidance]]
- [[diffusion-models]]
- [[scheduler]]
- [[ddpm]]
- [[foundations]]
