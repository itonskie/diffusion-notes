---
type: reference
tags:
  - foundations
  - scheduler
  - sampler
  - reference
date: 2026-06-29
updated: 2026-06-29
---

# Scheduler / Sampler Comparison

A diffusion model generates an image by starting from pure noise and **denoising it in steps** — 20, 50, sometimes 1000 little nudges from "static" toward "picture." The **sampler** (a.k.a. scheduler — the field uses both words for the same thing depending on which tool you're in) is the algorithm that decides how big each nudge is and in what direction. Different samplers take very different paths to a similar destination; some are faster, some are more stable, some produce slightly noisier or slightly cleaner outputs at the same step count.

This page is the lookup table for "which sampler do I use." The answer changes by model (Flux vs SDXL vs SD 1.5), by step budget, and sometimes by aesthetic. Source: [HF Diffusers scheduler overview](https://huggingface.co/docs/diffusers/api/schedulers/overview) cross-referenced with practitioner conventions from ComfyUI / A1111 / k-diffusion.

For the **concept** of what a scheduler is (training-time noise schedule vs inference-time sampler), see [[scheduler]]. This page is the lookup.

A few words you'll see repeatedly:

- **ODE solver** — Ordinary Differential Equation solver. The denoising trajectory is described as an ODE; samplers are different numerical methods for following that path. First-order = simple straight-line steps. Second-order = uses a correction.
- **SDE** — Stochastic Differential Equation. Same idea but with a small random kick on every step. More diverse outputs, sometimes a tiny bit noisier.
- **Multistep** — uses information from previous steps to predict the next one (like a momentum term). Usually more efficient.
- **Singlestep** — each step only uses the current state.
- **Ancestral** — adds fresh noise back in at each step. The "a" suffix in names like `Euler a` or `DPM2 a`.
- **Sigma** — the noise level at a given step. The schedule of sigmas (how noise level drops over time) is separate from the sampler that uses them.
- **Karras sigmas** — a specific clever spacing of noise levels from the EDM paper. Same sampler, better spacing, free quality bump.

## Naming map across tools

Same algorithm, three different names depending on which tool's UI you're staring at. Use this when reading between a HuggingFace tutorial, an A1111 forum post, and a ComfyUI workflow.

| A1111 / k-diffusion | 🤗 Diffusers | ComfyUI |
|---------------------|--------------|---------|
| DPM++ 2M | DPMSolverMultistepScheduler | `dpmpp_2m` |
| DPM++ 2M Karras | DPMSolverMultistepScheduler (use_karras_sigmas=True) | `dpmpp_2m` + `karras` sigmas |
| DPM++ 2M SDE | DPMSolverMultistepScheduler (algorithm_type="sde-dpmsolver++") | `dpmpp_2m_sde` |
| DPM++ 2M SDE Karras | same + use_karras_sigmas=True | `dpmpp_2m_sde` + karras |
| DPM++ SDE | DPMSolverSinglestepScheduler | `dpmpp_sde` |
| DPM++ SDE Karras | same + use_karras_sigmas=True | `dpmpp_sde` + karras |
| DPM2 | KDPM2DiscreteScheduler | `dpm_2` |
| DPM2 Karras | same + use_karras_sigmas=True | `dpm_2` + karras |
| DPM2 a | KDPM2AncestralDiscreteScheduler | `dpm_2_ancestral` |
| Euler | EulerDiscreteScheduler | `euler` |
| Euler a | EulerAncestralDiscreteScheduler | `euler_ancestral` |
| Heun | HeunDiscreteScheduler | `heun` |
| LMS | LMSDiscreteScheduler | `lms` |
| — | DEISMultistepScheduler | `deis` |
| — | UniPCMultistepScheduler | `uni_pc` |
| — | LCMScheduler | `lcm` (model-dependent) |
| — | DDIMScheduler | `ddim` |
| — | DDPMScheduler | `ddpm` |

## Sigma / noise schedule modifiers

These are *not* samplers — they're noise-spacing options applied on top of a sampler. A sampler is "how I take each step." A sigma schedule is "what noise level is each step *at*." You combine the two.

| A1111 / k-diffusion | 🤗 Diffusers init kwarg | What it does |
|---------------------|-------------------------|--------------|
| Karras | `use_karras_sigmas=True` | Sigma spacing from the EDM paper — better quality at low step counts |
| sgm_uniform | `timestep_spacing="trailing"` | SD3/Flux native spacing |
| simple | `timestep_spacing="trailing"` | Same as above, simpler implementation |
| exponential | `use_exponential_sigmas=True` | Exponential drop, less common |
| beta | `use_beta_sigmas=True` | Beta distribution shape, occasional artifact mitigation |

The Karras sigma schedule (Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022) noticeably improves quality at step counts under ~30. It's free — same sampler, different sigma schedule.

For Flux, the native `sgm_uniform` / `simple` spacing is generally recommended over `karras`. Use `karras` for SD 1.5 / SDXL. The reason: a model performs best with the schedule it was trained on, and Flux was trained on `sgm_uniform`.

## Pragmatic chooser

If you don't want to think about it, pick from these tables and move on.

### For Flux

| Goal | Recommended |
|------|-------------|
| Quality, slow | `dpmpp_2m` + `sgm_uniform`, 30-50 steps |
| Speed, decent quality | `euler` + `simple`, 20-25 steps |
| Fastest (with schnell distilled model) | `euler` + `simple`, **4 steps** |

### For SDXL

| Goal | Recommended |
|------|-------------|
| Quality, slow | `dpmpp_2m_sde` + `karras`, 30-40 steps |
| Quality, faster | `dpmpp_2m` + `karras`, 20-30 steps |
| Speed, balanced | `euler_ancestral`, 20 steps |
| Image-to-image | `dpmpp_2m` or `ddim`, 15-25 steps |

### For SD 1.5

| Goal | Recommended |
|------|-------------|
| Quality | `dpmpp_2m_sde` + `karras`, 25-35 steps |
| Speed | `dpmpp_2m` + `karras`, 20 steps |
| Stylized | `euler_ancestral`, 20-30 steps |
| With LCM-LoRA | `lcm`, 4-8 steps |

**LCM-LoRA** — a tiny LoRA you load on top of a normal model that lets it sample in 4–8 steps instead of 25+. Distillation packaged as a swap-in patch.

### Wan 2.2 / video diffusion

| Goal | Recommended |
|------|-------------|
| Standard | `unipc` or `dpmpp_2m`, 20-30 steps |
| Lower step count | `euler` + `simple`, 12-20 steps |

## What each sampler family actually does

The taxonomy below is "name of the algorithm → what it actually does → when you'd reach for it."

### Euler family
- **Euler** — first-order ODE solver. The simplest possible thing: at each step, predict the direction of travel and take one step that way. Robust, deterministic (same seed → same image). Slightly worse than DPM++ at the same step count but easier to reason about and basically never broken.
- **Euler ancestral** — same as Euler but adds a small dose of fresh noise at each step. Slightly more diverse outputs; deterministic-seed reproducibility still works (the noise is seeded too).

### Heun
2nd-order ODE solver. Instead of one prediction per step, it takes a tentative step, looks again from the new position, and averages. Better accuracy per step, but ~2× cost per step (two model evaluations). Rarely worth the extra time vs DPM++ 2M.

### DPM family
"DPM" = Diffusion Probabilistic Model — the same family as DDPM but with smarter math.
- **DPM2 / KDPM2** — 2nd-order solver. Mostly historical now.
- **DPM++ 2M** — multistep variant. The current default for most SDXL/Flux work. Good quality at 20-30 steps. The "2M" means 2nd-order, multistep.
- **DPM++ 2M SDE** — same but adds a stochastic term (random kick per step). Slightly more diverse, slightly noisier per step.
- **DPM++ SDE** (singlestep) — different per-step handling. Less commonly used in 2025.
- **DPM++ 2S a** — ancestral version. Listed in A1111 but not directly in diffusers.

### DDIM
**Deterministic** sampler from the [[ddim]] paper (Denoising Diffusion Implicit Models). Works at 20-50 steps; key advantage is **latent-space interpolation** — because it's fully deterministic, two seeds give two reproducible trajectories that you can blend between to morph one image into another. Useful for animation tricks.

### DDPM
The original ancestral sampler from the [[ddpm]] paper. Needs ~1000 steps for quality. Mostly historical now; rarely used at inference. Every other sampler on this page was invented to avoid this thing's step budget.

### LCM (Latent Consistency Model)
**Requires a model trained or fine-tuned for it** (either an LCM-LoRA loaded on top, or an LCM-distilled base). 4-8 steps total. Massive speed win when applicable; small quality regression vs full-step samplers. You can't just point LCM at a stock SDXL and expect it to work — the model has to be prepared for it.

### UniPC (Unified Predictor-Corrector)
A predictor-corrector samples in two phases per step: predict where you'd land, then correct using info from there. UniPC unifies that for diffusion. 2-3 step convergence on simple prompts; up to 10 steps for complex. A relatively recent solver, good for low-latency video/animation pipelines.

### DEIS
DEIS (Diffusion Exponential Integrator Sampler) — high-order solver. Less commonly defaulted-to than DPM++ but produces clean outputs at low step counts.

## Picking a step count

A rough heuristic: ask "what's the minimum step count that still looks OK to you?" and use that. More steps almost never *improves* quality past a model's sweet spot; it just costs more compute. The sweet spots roughly are:

- LCM-distilled models: 4-8 steps
- Flux Schnell (distilled): 4 steps
- Flux dev: 20-50 steps
- SDXL: 20-40 steps
- SD 1.5: 20-30 steps
- Native DDPM: 1000 steps (don't do this)

## What "Karras sigmas" buys you

Recall: a sigma is the noise level at a given step. The default schedule spaces them out uniformly. The Karras schedule spaces them *non*-uniformly — concentrating more steps in the noise-level ranges where the model is most sensitive (the high-frequency detail zones) and fewer where it isn't.

Free quality improvement of ~1 FID point at the same step count for SD/SDXL. (FID = Fréchet Inception Distance, the standard "how realistic does this batch of images look" metric — lower is better.)

For Flux, the model was trained with `sgm_uniform` and is most consistent with that schedule. Karras sigmas on Flux can still work but might subtly shift the output distribution.

## See Also

- [[scheduler]]
- [[ddim]]
- [[ddpm]]
- [[reference/diffusion-course]]
- [[reference/flux-vs-sdxl]]
- [[foundations]]
