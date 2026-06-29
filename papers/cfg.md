---
type: paper
tags:
  - foundations
  - cfg
  - guidance
  - paper
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Classifier-Free Diffusion Guidance

> Ho & Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop on Score-Based Methods. (Posted on arxiv as 2207.12598 in 2022.)
> https://arxiv.org/abs/2207.12598 • [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2207.12598)

## Status

**Developing** — content from ar5iv HTML render.

## Abstract (verbatim)

> "Classifier guidance is a recently introduced method to trade off mode coverage and sample fidelity in conditional diffusion models post training... We show that guidance can be indeed performed by a pure generative model without such a classifier: in what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity."

## The contribution

Classifier guidance (Dhariwal & Nichol 2021) needs a separately-trained classifier `f(y|x_t, t)` that can score noisy images. That's an extra training step nobody wants to do.

CFG replaces it with a clever trick: train one diffusion model that handles both conditional and unconditional generation (by dropping the condition with some probability during training). At inference, combine the two predictions linearly to push samples toward the conditional distribution.

This **eliminates the need for a separate classifier** and is the technique used by every modern text-to-image model.

## Math (paper convention)

```
ε̃_θ(z_λ, c) = (1 + w) · ε_θ(z_λ, c) - w · ε_θ(z_λ)
```

- `ε_θ(z_λ, c)` = conditional noise prediction
- `ε_θ(z_λ)` = unconditional noise prediction
- `w` = guidance strength (paper's convention)
- `λ` = log-SNR (a reparameterization of timestep `t`)

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

When reading papers, divide ComfyUI numbers by ~1 to get paper-style `w`. When reading code, look at the exact formula being used.

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

The unconditional case is **not zero**; it's a specific learned token (often the embedding of `∅`).

### Dropout probability `p_uncond`

The paper tested `p_uncond ∈ {0.1, 0.2, 0.5}`:

- **0.1 and 0.2 perform equally well**
- **0.5 underperforms**

Modern convention: 10-15% dropout during text-to-image training.

## The implicit-classifier trick

CFG works because the difference of conditional and unconditional noise predictions approximates a classifier gradient (via Bayes' rule on the score functions):

```
∇_x log p(c | x_t) ∝ ∇_x log p(x_t | c) - ∇_x log p(x_t)
                  ∝ ε_θ(x_t, c) - ε_θ(x_t)
```

So `ε(cond) - ε(uncond)` plays the role of `∇ log f(c|x)` from classifier guidance — but it's free because both terms come from the same network.

**Caveat**: this only holds if the network's score estimates are exact (conservative vector fields). In practice neural networks don't satisfy this, so CFG is an *approximation* of true classifier guidance, not an equivalence. The fact that it works so well empirically is somewhat magical.

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

### Why the optimal `w` differs from text-to-image practice

ImageNet has 1000 classes — relatively simple conditioning. Text-to-image conditioning is harder (millions of possible prompts), so models like SDXL need stronger guidance (CFG 7) to lock onto a specific prompt. Flux was specifically distilled to need less guidance (CFG ~3.5).

## Other findings

- **64×64 ImageNet, FID 1.55** at `w=0.1` (`v=0.3` log-SNR interpolation), 400k training steps
- The paper notes that `p_uncond` hyperparameters were inherited from classifier-guidance experiments and "may be suboptimal" for CFG
- Naïve alternatives (just truncating noise variance) cause blurry samples — CFG specifically operates on score combinations, which is what makes it work where simpler tricks fail

## Why this matters for the sprint

Every text-to-image model in ComfyUI uses CFG at inference. The CFG slider IS this paper's `w + 1`. Understanding the FID/IS curve tells you:
- Why prompt adherence goes up with higher CFG (more IS bias)
- Why diversity goes down with higher CFG (FID degrades)
- Why "burnt" / over-saturated outputs come from CFG too high
- Why some workflows use higher CFG early in sampling and lower later (CFG scheduling)

Flux's lower-than-SD CFG default (3.5 vs 7) is partly because of post-training distillation that reduced the model's reliance on strong guidance, and partly because Flux was trained with an architecture (MMDiT) that produces stronger conditioning signal natively.

## See Also

- [[classifier-free-guidance]]
- [[guidance]]
- [[diffusion-models]]
- [[scheduler]]
- [[ddpm]]
- [[foundations]]
