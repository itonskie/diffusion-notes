---
type: concept
tags:
  - foundations
  - guidance
  - inference
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Guidance

A family of inference-time techniques that **modify the model's noise prediction at each denoising step** to steer generation toward some objective — without retraining the model.

[[classifier-free-guidance|CFG]] is the most common example, but guidance is more general: you can use *any* function of the partially-denoised image to push generation one way or another.

## The general form

At each [[scheduler|sampling]] step, the model gives you `ε_θ(x_t, t)`. Before using it to take the next denoising step, you modify it:

```
ε̃_θ(x_t, t) = ε_θ(x_t, t) + scale · g(x_t, t)
```

where `g(x_t, t)` is the guidance signal — the direction in which to push.

The "direction" is usually a gradient: `g = ∇_{x_t} L(x_t)` where `L` is some loss measuring how far `x_t` is from what you want. The gradient tells you which way to perturb `x_t` to reduce `L`.

## Examples

### Classifier guidance
Train a classifier `f(y | x_t, t)` that takes noisy images and predicts class labels. At inference, use `g = ∇_{x_t} log f(y | x_t)` to push toward class `y`.

- Pros: works with an unconditional base model.
- Cons: requires training a noise-aware classifier — extra work nobody wants to do.
- Mostly superseded by [[classifier-free-guidance|CFG]].

### Classifier-free guidance (CFG)
The dominant text-to-image technique. See [[classifier-free-guidance]].

### CLIP guidance
Use a pre-trained CLIP model. Compute `L = -similarity(CLIP_image(x_t), CLIP_text(prompt))`. Gradient pushes `x_t` to be more CLIP-similar to the prompt.

- Pros: no diffusion-side training needed; any prompt-able CLIP works.
- Cons: CLIP wasn't trained on noisy images, so its embedding quality degrades at high `t`. Often less reliable than CFG with a text-conditional model.
- Used as a fallback before CFG became standard; still useful for niche objectives.

### Color guidance / aesthetic guidance / etc.
Any loss function on `x_t` works. Examples from the HF course:
- **Color** — push average color of `x_t` toward a target RGB. Trivial differentiable function.
- **Aesthetic** — use a pre-trained aesthetic-scoring model.
- **Composition** — saliency maps, layout constraints.

The fact that *any differentiable function on `x_t`* can be used as guidance is why this technique is so flexible.

## What guidance costs

- **Compute** — every guidance step requires either an extra forward pass (CFG, classifier guidance) or a backward pass through the guidance function (CLIP, color, etc.). Roughly 2× inference cost.
- **Numerical instability** — guidance gradients on `x_t` can be large; needs careful scaling per step.
- **Quality** — too much guidance causes artifacts. Less guidance loses control. The sweet spot is task-dependent.

## Guidance vs fine-tuning

Both add control. Different trade-offs — see [[fine-tuning]] for the comparison.

## When you'd use guidance in practice

In 2025 ComfyUI workflows, guidance is mostly:
- **CFG** (always on, configurable scale)
- **Negative prompts** (technically a CFG variant — push away from the negative)
- **IPAdapter-style image guidance** (uses a pre-trained image encoder)
- **Region-specific guidance** (different prompts in different regions)

The raw "gradient of a loss function on x_t" approach (color guidance, CLIP guidance) is now mostly historical — it was a useful stepping stone but specific tools have largely replaced it.

## See Also

- [[classifier-free-guidance]]
- [[diffusion-models]]
- [[fine-tuning]]
- [[scheduler]]
- [[u-net]]
- [[foundations]]
