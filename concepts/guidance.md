---
type: concept
tags:
  - foundations
  - guidance
  - inference
date: 2026-06-29
updated: 2026-06-29
---

# Guidance

A trick for steering a [[diffusion-models|diffusion model]] toward what you want at *generation* time, without retraining anything. The bet: at each denoising step the model spits out a prediction of "here's the noise in this image, subtract it." Guidance grabs that prediction and tilts it in a direction you choose before letting the sampler use it. Same model, just a nudge on every step.

[[classifier-free-guidance|CFG]] is the famous example — it's what makes text-to-image models actually follow your prompt instead of generating random plausible images. But guidance is a general pattern: *any* function you can compute on the half-denoised image can be turned into a guidance signal.

## The general form

At each [[scheduler|sampling step]] (one tick of the reverse denoising walk), the model gives you `ε_θ(x_t, t)` — its prediction of the noise hiding in the current noisy image `x_t` at timestep `t`. The subscript `θ` (theta) just means "parameters of the trained network." Before the sampler uses this prediction to take the next step, you modify it:

```
ε̃_θ(x_t, t) = ε_θ(x_t, t) + scale · g(x_t, t)
```

In words: the *modified* noise prediction equals the original noise prediction plus some scaled push in a chosen direction. Symbols:

- `ε_θ(x_t, t)` — the raw noise prediction the model gave you.
- `ε̃_θ(x_t, t)` — the tilted version you actually use (the tilde over the epsilon means "modified").
- `g(x_t, t)` — the guidance signal, a vector pointing in the direction you want to push.
- `scale` — how hard to push. A knob you set.

The "direction" `g` is usually a **gradient** — the partial derivative of some loss with respect to the noisy image, telling you which pixel-changes would reduce the loss fastest. Written out: `g = ∇_{x_t} L(x_t)` where `L` is a loss measuring how far `x_t` is from what you want. Gradient = direction-of-steepest-descent. If your loss says "this image is too red," the gradient points to "less red here," and you take a step that way.

## Examples

### Classifier guidance
The original idea (pre-CFG). Train a separate classifier `f(y | x_t, t)` that takes a *noisy* image and predicts a class label `y` (e.g. "this noisy blob is a dog"). At inference, use `g = ∇_{x_t} log f(y | x_t)` to push toward class `y`. Plain English: "tilt every denoising step toward making the image look more like a dog to the classifier."

- Pros: works with an unconditional base model (the diffusion model doesn't have to know about classes at all).
- Cons: requires training a noise-aware classifier — a whole second model trained specifically to recognize *noisy* images, which is extra work nobody wants to do.
- Mostly superseded by [[classifier-free-guidance|CFG]], which avoids the extra classifier entirely.

### Classifier-free guidance (CFG)
The dominant text-to-image technique. See [[classifier-free-guidance]] for the full breakdown.

### CLIP guidance
**CLIP** is a pre-trained model from OpenAI that maps images and text into the same embedding space, so you can measure how well an image matches a caption. The guidance trick: compute `L = -similarity(CLIP_image(x_t), CLIP_text(prompt))`. Then the gradient of `L` pushes `x_t` to be more CLIP-similar to the prompt.

- Pros: no diffusion-side training needed; any prompt-able CLIP works off the shelf.
- Cons: CLIP was trained on *clean* images, not noisy ones. Its embeddings degrade when fed a half-denoised mess, especially at high `t` (very noisy). So the guidance signal is unreliable early in sampling. Usually less robust than CFG with a properly text-conditional diffusion model.
- Used as a fallback before CFG became standard; still useful for niche objectives.

### Color guidance / aesthetic guidance / etc.
Any **differentiable** function on `x_t` (one whose gradient you can compute) becomes a guidance signal. Examples from the HuggingFace course:
- **Color** — push average color of `x_t` toward a target RGB. Trivial differentiable function: "the difference between current average color and target."
- **Aesthetic** — use a pre-trained aesthetic-scoring model. Push toward higher scores.
- **Composition** — saliency maps (which parts of an image draw attention), layout constraints.

The fact that *any differentiable function on `x_t`* can be plugged in is why this technique is so flexible. The cost is that you need to be able to compute gradients through your loss, which means it has to be implemented in PyTorch (or whatever) — you can't guide on, say, "does a human like this image."

## What guidance costs

- **Compute.** Every guidance step requires either an extra forward pass through the model (CFG, classifier guidance) or a backward pass through the guidance function (CLIP, color, etc.). Roughly **2× inference cost** in the common case.
- **Numerical instability.** Gradients on `x_t` can blow up. Needs careful per-step scaling or the image goes haywire.
- **Quality.** Too much guidance causes artifacts (washed-out colors, weird oversaturation). Too little loses control. The sweet spot is task-dependent — you tune.

## Guidance vs fine-tuning

Both let you control what the model does. Different trade-offs — see [[fine-tuning]] for the comparison. Short version: guidance is free (no training) but slower per generation; fine-tuning is expensive once (training) but cheap forever after.

## When you'd use guidance in practice

In 2025 ComfyUI workflows, guidance is mostly:

- **CFG** (always on, configurable scale — it's how prompts work at all)
- **Negative prompts** (technically a CFG variant — push *away* from a "don't do this" prompt)
- **IPAdapter-style image guidance** (uses a pre-trained image encoder to inject a reference image's style)
- **Region-specific guidance** (different prompts in different regions of the canvas)

The raw "compute the gradient of a loss function on `x_t`" approach (color guidance, CLIP guidance) is now mostly historical. It was a useful stepping stone — proof that you could steer diffusion with arbitrary objectives — but in practice purpose-built tools like IPAdapter have replaced it for almost every use case.

## See Also

- [[classifier-free-guidance]]
- [[diffusion-models]]
- [[fine-tuning]]
- [[scheduler]]
- [[u-net]]
- [[foundations]]
