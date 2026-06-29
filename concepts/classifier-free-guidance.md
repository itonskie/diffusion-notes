---
type: concept
tags:
  - foundations
  - diffusion
  - cfg
  - conditioning
date: 2026-06-29
updated: 2026-06-29
---

# Classifier-Free Guidance (CFG)

The trick that makes a text-to-image model actually pay attention to your prompt. Without it, a text-conditional [[diffusion-models|diffusion model]] *technically* knows about the prompt but tends to wander — generating something plausible that only loosely matches what you asked for. CFG cranks the prompt's influence up to a chosen intensity, at the cost of extra compute per step. It's how Flux, SDXL, SD3, and pretty much every modern text-to-image model are controlled at inference time. The "C" stands for the fact that, unlike older approaches, you don't need a separate classifier network to do the steering.

## The picture

Train one model that can do two things: predict noise *given* your prompt, and predict noise *without any prompt at all*. At generation time, ask it both. The conditional answer says "this is what I'd produce for your prompt." The unconditional answer says "this is what I'd produce on average." Subtract them and you get a *direction* — "this is what's specifically prompt-driven, as opposed to generic." Then *exaggerate* that direction by some multiplier. The result: outputs that lean harder into the prompt than the model would on its own.

That exaggeration multiplier is the "CFG scale" slider you see in every UI.

## How the training works

Train a single conditional model `ε_θ(x_t, t, y)` where:

- `x_t` — the noisy image at timestep t.
- `t` — the timestep (how noisy x_t is on a scale of 1 to T, usually T=1000).
- `y` — the condition (the text prompt, encoded as embeddings).
- `ε_θ` — the noise the model predicts.

During training, **randomly drop the condition** with some probability (typically 10-20%) — when dropped, replace `y` with a learned "empty" token (sometimes literally `∅`, the null symbol) so the model has to predict noise *unconditionally*. Now you have one network that can run in two modes: conditional `ε_θ(x_t, t, y)` and unconditional `ε_θ(x_t, t, ∅)`. Same weights, just different inputs.

In code:

```python
# During training
y = caption_embedding  # text embedding
if random.random() < 0.1:   # 10% drop probability
    y = unconditional_embedding  # learned token, or zeros
loss = mse(model(x_t, t, y), epsilon)
```

That's it. CFG is mostly an *inference-time* technique — training just needs to occasionally see the unconditional case so the model knows how to handle it when asked.

## How the inference works

At inference (generation), run the model twice per denoising step — once with the prompt, once without — and combine the two predictions. Two equivalent formulas, with different conventions floating around in the wild (see [[papers/cfg]]):

```
ε̃_θ(x_t, t, y) = ε_θ(x_t, t, ∅) + s · (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))    [UI convention, s = CFG scale]
ε̃_θ(x_t, t, y) = (1 + w) · ε_θ(x_t, t, y) - w · ε_θ(x_t, t, ∅)             [paper convention, w = s - 1]
```

These two equations are algebraically identical, just written with a different "zero point" for the slider. The relationship: **`s = w + 1`**. A paper saying "w=6" is the same setting as a ComfyUI/A1111 slider showing "CFG=7." This off-by-one bites people constantly when porting recipes between papers and UIs. Keep both forms in your head.

What each symbol means:

- `ε_θ(x_t, t, y)` — the model's noise prediction *with* the prompt.
- `ε_θ(x_t, t, ∅)` — the model's noise prediction *without* the prompt.
- `ε̃_θ(x_t, t, y)` — the combined/modified prediction (tilde = modified) that gets fed to the sampler.
- `s` — the UI's "CFG scale" slider value.
- `w` — the paper's guidance weight.

### Reading the slider (using `s`)

- `s = 0` → pure unconditional generation (prompt is ignored entirely).
- `s = 1` → pure conditional generation (the model's natural conditional output, no extra amplification).
- `s > 1` → amplified guidance — output adheres more strictly to the prompt.
- Typical SDXL/SD: `s ≈ 7`. Typical Flux: `s ≈ 3.5`. Higher than 10 usually over-saturates (colors get crispy and unnatural, sometimes referred to as "burnt").

## Why it works (math note, optional)

The derivation comes from [[classifier-guidance|classifier guidance]] — the older approach CFG replaces. There's a Bayes-rule identity:

```
∇ log p(y | x_t) = ∇ log p(x_t | y) - ∇ log p(x_t)
```

In words: the gradient of "how much does this image match the label" equals the gradient of "how likely is this image given the label" minus the gradient of "how likely is this image overall." It says the *classifier signal* is exactly the difference between the conditional and unconditional generative signals.

CFG approximates this difference using the conditional vs unconditional noise predictions, which (via the score-matching connection from [[diffusion-models]] — diffusion models implicitly learn `∇ log p(x_t)`) are proportional to `∇ log p(x_t | y)` and `∇ log p(x_t)` respectively. So CFG uses the model's own ability to predict noise with/without the condition as a stand-in for the gradient that a separate classifier would have given you. No extra network needed.

This is why it's called *classifier-free*: the math doesn't need a real classifier, just the model's own dual ability.

## What CFG costs

- **2× inference compute.** Every denoising step requires two forward passes: one with the condition, one without. This is a big chunk of why image generation is slow.
- **Trade-off.** Higher `s` (or `w`) improves prompt adherence but reduces diversity (outputs collapse toward a stereotypical "best fit") and can introduce artifacts (over-saturation, burnt colors at extreme values).

Some modern variants — perturbed-attention guidance, CFG++, autoguidance — try to claw back the artifact cost without losing prompt adherence. They're newer recipes that tweak how the conditional/unconditional predictions are combined.

## CFG and the "many denoising steps" question

CFG is applied at *every* denoising step. That's a big reason inference is slow — you're doing twice the model forward passes per step, multiplied across 20-50 steps.

Some [[scheduler|samplers]] and recent techniques skip CFG on certain steps (early or late in the trajectory, where the contribution is small) to claw back speed. LCM and Flux Schnell were specifically distilled to need less or no CFG at inference — that's part of how they get away with only 4-8 steps.

## See Also

- [[papers/cfg]]
- [[diffusion-models]]
- [[guidance]]
- [[scheduler]]
- [[ddpm]]
- [[u-net]]
- [[dit]]
- [[foundations]]
