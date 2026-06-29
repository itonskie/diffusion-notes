---
type: concept
tags:
  - foundations
  - diffusion
  - scheduler
date: 2026-06-29
updated: 2026-06-29
---

# Scheduler

The word "scheduler" gets used for two completely different things in diffusion. That's the first thing to know, because conflating them will trip you up reading any tutorial or library doc.

The two things:

1. **Noise schedule** — used during *training*. A recipe for how much noise to add to a clean image at each step of the destruction process. "At step 1 add this much noise, at step 1000 the image is basically pure static."
2. **Sampler** — used during *inference* (generation). A recipe for walking *back* from pure static to a clean image. "Take 20 steps, here's how big each step is and how to nudge the prediction."

Same word, very different jobs. HuggingFace's `diffusers` library calls both "schedulers," which is where most of the confusion comes from. ComfyUI and similar UIs usually split them — "scheduler" means the noise schedule, "sampler" means the inference-time walker. This page follows the ComfyUI split because that's where you'll actually use this knowledge.

## Picture: what's happening

A [[diffusion-models|diffusion model]] is trained by taking a clean image, adding noise to it in stages until it's pure random static, and learning to predict the noise that was added so it can be subtracted off. The noise schedule tells you how much noise to add at each stage during training. The sampler tells you how to traverse those stages in reverse to generate a new image.

You train *once* with a fixed noise schedule. You sample *many times* with whatever sampler you want. The model never gets retrained when you switch samplers — that's the key decoupling.

## Noise schedule (training time)

The forward process — the destruction half — adds a small bit of Gaussian noise at each step `t`. **Gaussian noise** here just means random numbers pulled from a bell curve, the standard "white noise" of math. The amount added at step `t` is called `β_t` (Greek letter beta). As `t` climbs from 1 to T (typically T=1000), the image gets noisier; by `t=T` it's basically indistinguishable from pure random static.

### Math note: the closed-form jump

The honest version of "add noise step by step" would be: do it 1000 times in a loop. That's slow. The clever trick: because each step's noise is Gaussian and independent, you can collapse all 1000 steps into one formula and jump straight to any noise level in a single shot.

Define two helper quantities:

- `α_t = 1 - β_t` — how much of the *signal* survives one step of noising. If β_t is small, α_t is close to 1 (signal mostly preserved).
- `ᾱ_t = ∏_{i=1}^t α_i` — the running product of all α values up to step t. This is the fraction of original signal still present after t noising steps.

Then the closed-form jump is:

```
x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

In plain English: the noisy image at step t equals "some of the original image" plus "some random noise," with the mix controlled by `ᾱ_t`. Symbols:

- `x_0` — the original clean image.
- `x_t` — the image after t noising steps.
- `ε` (epsilon) — a fresh batch of Gaussian noise pulled from a standard normal distribution.
- `√ᾱ_t` — how much of the original to keep.
- `√(1 - ᾱ_t)` — how much noise to mix in.

This is **critical for training**, where every training example draws a *random* t and needs the noisy version instantly. Without the closed form you'd be running a 1000-step loop per training sample. With it you sample t, plug into one formula, done.

### Linear schedule (DDPM, Ho et al. 2020)
- `β_1 = 10⁻⁴`, `β_T = 0.02`, linear interpolation between them
- T = 1000 steps
- The original recipe from the [[ddpm]] paper. Simple and works.
- Weakness: `ᾱ_t` (signal-survival fraction) drops too fast near the end. You're spending lots of training capacity on images that are already 99% destroyed, which is wasted effort.

### Cosine schedule (Improved DDPM, Nichol & Dhariwal 2021)
```
ᾱ_t = f(t)/f(0),   f(t) = cos²((t/T + s)/(1+s) · π/2)
```

A different curve shape that fixes the linear schedule's late-stage problem. Symbols:

- `f(t)` — a cosine-squared function defining how `ᾱ_t` falls off.
- `s` (offset) — a small constant (0.008) that keeps β from getting absurdly tiny right at t=0.
- The shape is near-linear in the middle and gentler at the endpoints (less wasted capacity at both ends).

Empirically this gives better likelihoods and better samples in the mid-range of t, where most of the visually interesting denoising work happens.

The schedule choice is **load-bearing for training** but mostly invisible at inference. You can train with a cosine schedule and still pick whatever sampler you want at inference time — the sampler reads off the trained model regardless of how its noise schedule was shaped.

## Sampler (inference time)

Now flip directions. You have a trained model. You want to generate an image. The sampler is the algorithm that starts from pure random noise and walks back to a clean image, calling the model at each step to ask "how much noise is in this thing, what should I subtract?"

The fancy version of the question: this reverse walk is the numerical integration of a **reverse SDE/ODE**. **SDE** = stochastic differential equation, **ODE** = ordinary differential equation — these are math objects describing how something evolves continuously over time. Different samplers integrate them differently, with different trade-offs between speed and quality.

What surprised me: a single trained model can be sampled with wildly different samplers and step counts. The model doesn't care. The sampler does all the smarts about *how* to step.

| Sampler | Stochastic? | Typical steps | Notes |
|---------|-------------|---------------|-------|
| DDPM (ancestral) | Yes | 1000 | Original, slow, baseline quality |
| DDIM | Deterministic (η=0) | 50-250 | Subsample timesteps; enables image-to-image interpolation |
| DPM++ 2M | Either | 20-50 | High-order ODE solver, current default for SD/Flux |
| Euler | Deterministic | 20-50 | Simple, robust, slightly worse than DPM++ |
| Euler Ancestral | Stochastic | 20-50 | More variety, can introduce subtle noise |
| LCM | Deterministic | 4-8 | Requires LCM-distilled model — radically fewer steps |
| Heun | Deterministic | 20-50 | 2nd-order, smoother but ~2x cost per step |

Two terms in that table that need decoding:

- **Stochastic** — adds fresh random noise at each step. Same seed, same prompt, but the path varies because there's added randomness along the way. Often better diversity, sometimes grainier.
- **Deterministic** — no added randomness during sampling. Same seed and prompt = exactly the same output every time. Lets you do tricks like blending between two seeds.

### Why does sampler choice matter?

- **Speed.** DDPM needs ~1000 steps. DPM++ gets comparable quality in 20-50. That's a 20-50× speedup at inference — huge.
- **Determinism.** Deterministic samplers (DDIM, DPM++, Euler) give the same output for the same seed every time and let you do latent-space interpolation (smoothly blending between two outputs).
- **Compatibility.** LCM and other "distilled-step" samplers require a model that was specifically trained or fine-tuned for them. You can't just drop LCM onto a vanilla model and expect 4-step generation to work.

[[ddim|DDIM]] was the first sampler to show that you could get DDPM-quality output in way fewer steps, by reframing the reverse process as a non-Markovian ODE (math for "the next step doesn't have to only depend on the immediately previous one — it can use more history").

## Training-vs-inference decoupling

Worth saying again because it tripped me up. You train with noise schedule β_t at T=1000 steps. At inference, the sampler picks an arbitrary subset of timesteps — say 50 evenly spaced from 1 to 1000 — and walks the reverse process over just those.

The model itself **never saw** "step 20 out of 50" during training. It always saw "step t out of 1000." The sampler's job is to translate between the two: when it's at its own step 20 of 50, it tells the model "pretend you're at t=400 out of 1000" and the model happily denoises accordingly.

This is why the same trained checkpoint can be sampled with DDPM at 1000 steps, DDIM at 50 steps, or DPM++ at 25 steps with very different speed/quality profiles. You're trading off how coarsely you walk the reverse process.

## For the sprint

When training a [[lora|LoRA]] with AI-Toolkit or similar, you generally **do not change the noise schedule** — it's baked into the base model you're fine-tuning. Mess with it and you'll desync from the base. Leave it alone.

You DO pick samplers at inference, in ComfyUI. The default modern choice for Flux is **dpmpp_2m + sgm_uniform** or **euler + simple** with 20-30 steps. Those are sampler-name + schedule-name pairs ComfyUI uses; treat them as "known-good combos" until you have a reason to deviate.

## See Also

- [[reference/scheduler-comparison]] — full lookup table across tools (A1111 / k-diffusion / Diffusers / ComfyUI) with per-model recommendations
- [[diffusion-models]]
- [[ddpm]]
- [[ddim]]
- [[classifier-free-guidance]]
- [[foundations]]
