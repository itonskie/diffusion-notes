---
type: concept
tags:
  - foundations
  - diffusion
  - cfg
  - conditioning
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Classifier-Free Guidance (CFG)

A trick for steering [[diffusion-models|diffusion models]] toward a condition (text prompt, class label) more strongly than the model would natively, **without** requiring a separate classifier network. The standard way text-to-image models like Flux, SDXL, and SD3 are controlled at inference.

## The core idea

Train a single conditional model `ε_θ(x_t, t, y)` where `y` is the condition. During training, randomly drop the condition with some probability (typically 10-20%) — when dropped, the model has to predict noise unconditionally. Now you have one network that can do both `ε_θ(x_t, t, y)` (conditional) and `ε_θ(x_t, t, ∅)` (unconditional).

At inference, use both predictions and **extrapolate** away from the unconditional toward the conditional:

```
ε̃_θ(x_t, t, y) = ε_θ(x_t, t, ∅) + w · (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

where `w` is the **guidance scale** (sometimes written as `w + 1` depending on convention).

- `w = 0` → pure unconditional generation (the prompt is ignored)
- `w = 1` → pure conditional generation (no guidance amplification)
- `w > 1` → amplified guidance — outputs adhere more strictly to the prompt
- Typical SDXL/SD: `w ≈ 7`. Typical Flux: `w ≈ 3.5`. Higher than 10 usually over-saturates.

## Why it works

The math derivation (from [[classifier-guidance|classifier guidance]]) gives:

```
∇ log p(y | x_t) = ∇ log p(x_t | y) - ∇ log p(x_t)
```

CFG approximates this difference using the conditional vs unconditional noise predictions, which (via the score-matching connection from [[diffusion-models]]) are proportional to `∇ log p(x_t | y)` and `∇ log p(x_t)` respectively. So CFG is implicitly using the model's own ability to predict noise with/without the condition as a stand-in for a classifier's gradient.

## What CFG costs

- **2× inference compute** — every denoising step requires *two* forward passes: one with the condition, one without.
- **Trade-off**: higher `w` improves prompt adherence but reduces diversity and can introduce artifacts (over-saturation, "burnt" colors at extreme values).

Some modern variants (perturbed-attention guidance, CFG++, autoguidance) try to reduce the artifact cost without losing prompt adherence.

## CFG and the "many denoising steps" question

CFG is applied at *every* denoising step. It is one of the reasons inference is slow — you're doing twice the model forward passes per step.

Some [[scheduler|samplers]] and recent techniques skip CFG on certain steps (early or late in the trajectory) to claw back speed. LCM and Flux Schnell were specifically distilled to need less or no CFG at inference.

## How it gets applied during training

In code (PyTorch-ish pseudocode):

```python
# During training
y = caption_embedding  # text embedding
if random.random() < 0.1:   # 10% drop probability
    y = unconditional_embedding  # learned token, or zeros
loss = mse(model(x_t, t, y), epsilon)
```

That's it. CFG is mostly an *inference-time* technique — training just needs to occasionally see the unconditional case so the model knows how to handle it.

## See Also

- [[diffusion-models]]
- [[scheduler]]
- [[ddpm]]
- [[u-net]]
- [[dit]]
- [[foundations]]
