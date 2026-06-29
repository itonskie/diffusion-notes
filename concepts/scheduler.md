---
type: concept
tags:
  - foundations
  - diffusion
  - scheduler
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Scheduler

"Scheduler" is overloaded — it means two related-but-distinct things in diffusion:

1. **Noise schedule (training-time)** — how much noise is added at each step *during training*. Defined by the variance schedule `β_t`.
2. **Sampler (inference-time)** — how the denoising trajectory is traversed *during generation*. The choice of sampler determines how many steps you need and how good the output is.

The HuggingFace `diffusers` library confusingly calls both of these "schedulers." Most ComfyUI/UI tools split them — "scheduler" usually means the noise schedule, "sampler" means the inference-time integrator.

## Noise schedule (training)

The forward process adds Gaussian noise at level β_t at step t. As t increases from 1 to T, the image gets noisier; at t=T it's approximately pure isotropic Gaussian.

Define:
- `α_t = 1 - β_t`
- `ᾱ_t = ∏_{i=1}^t α_i`

Then the closed-form forward jump is:

```
x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

This means you can sample any noise level directly without iterating — critical for training, where each example draws a random t.

### Linear schedule (DDPM, Ho et al. 2020)
- `β_1 = 10⁻⁴`, `β_T = 0.02`, linear interpolation
- T = 1000 steps
- Simple, used in the original [[ddpm]] paper
- Weakness: drops `ᾱ_t` too fast near the end, wasting capacity on already-mostly-destroyed images

### Cosine schedule (Improved DDPM, Nichol & Dhariwal 2021)
```
ᾱ_t = f(t)/f(0),   f(t) = cos²((t/T + s)/(1+s) · π/2)
```
- Small offset `s` (0.008) keeps β near t=0 from being too small
- Near-linear drop in the middle; gentler at endpoints
- Better likelihoods, better samples in mid-range

The schedule choice is **load-bearing for training** but somewhat decoupled from inference — you can train with a cosine schedule and still pick whatever sampler you want at inference.

## Sampler (inference)

The trained model can be sampled in many ways. Different samplers integrate the reverse SDE/ODE differently, trading speed vs quality.

| Sampler | Stochastic? | Typical steps | Notes |
|---------|-------------|---------------|-------|
| DDPM (ancestral) | Yes | 1000 | Original, slow, baseline quality |
| DDIM | Deterministic (η=0) | 50-250 | Subsample timesteps; enables image-to-image interpolation |
| DPM++ 2M | Either | 20-50 | High-order ODE solver, current default for SD/Flux |
| Euler | Deterministic | 20-50 | Simple, robust, slightly worse than DPM++ |
| Euler Ancestral | Stochastic | 20-50 | More variety, can introduce subtle noise |
| LCM | Deterministic | 4-8 | Requires LCM-distilled model — radically fewer steps |
| Heun | Deterministic | 20-50 | 2nd-order, smoother but ~2x cost per step |

### Why does sampler choice matter?

- **Speed** — DDPM needs ~1000 steps; DPM++ gets equivalent quality in 20-50.
- **Determinism** — deterministic samplers (DDIM, DPM++, Euler) give the same output for the same seed and let you do latent-space interpolation.
- **Compatibility** — LCM and other distilled-step samplers require a model trained or fine-tuned for them.

[[ddim|DDIM]] was the first major sampler to demonstrate that fewer steps could match DDPM quality, by reframing the reverse process as a non-Markovian ODE.

## Training-vs-inference decoupling

You train with noise schedule β_t at T=1000 steps. At inference, the sampler picks an arbitrary subset of timesteps — e.g., 50 evenly spaced from 1 to 1000 — and integrates the reverse process over just those. The model never saw "step 20 out of 50" during training; it always saw "step t out of 1000." The sampler's job is to translate between the two.

This is why the same trained checkpoint can be sampled with DDPM at 1000 steps, DDIM at 50 steps, or DPM++ at 25 steps with very different speed/quality profiles.

## For the sprint

When training a [[lora|LoRA]] with AI-Toolkit or similar, you generally do not change the noise schedule — it's a property of the base model you're fine-tuning. You DO pick samplers at inference, in ComfyUI. The default modern choice for Flux is **dpmpp_2m + sgm_uniform** or **euler + simple** with 20-30 steps.

## See Also

- [[diffusion-models]]
- [[ddpm]]
- [[ddim]]
- [[classifier-free-guidance]]
- [[foundations]]
