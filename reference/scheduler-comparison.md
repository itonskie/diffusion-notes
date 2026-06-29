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

The lookup table for "which sampler do I use." Source: [HF Diffusers scheduler overview](https://huggingface.co/docs/diffusers/api/schedulers/overview) cross-referenced with practitioner conventions from ComfyUI / A1111 / k-diffusion.

For the **concept** of what a scheduler is (training-time noise schedule vs inference-time sampler), see [[scheduler]]. This page is the lookup.

## Naming map across tools

Same algorithm, three different names. Use this when reading between a HuggingFace tutorial, an A1111 forum post, and a ComfyUI workflow.

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

These are *not* samplers — they're noise-spacing options applied on top of a sampler.

| A1111 / k-diffusion | 🤗 Diffusers init kwarg | What it does |
|---------------------|-------------------------|--------------|
| Karras | `use_karras_sigmas=True` | Sigma spacing from the EDM paper — better quality at low step counts |
| sgm_uniform | `timestep_spacing="trailing"` | SD3/Flux native spacing |
| simple | `timestep_spacing="trailing"` | Same as above, simpler implementation |
| exponential | `use_exponential_sigmas=True` | Exponential drop, less common |
| beta | `use_beta_sigmas=True` | Beta distribution shape, occasional artifact mitigation |

The Karras sigma schedule (Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022) noticeably improves quality at step counts under ~30. It's free — same sampler, different sigma schedule.

For Flux, the native `sgm_uniform` / `simple` spacing is generally recommended over `karras`. Use `karras` for SD 1.5 / SDXL.

## Pragmatic chooser

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

### Wan 2.2 / video diffusion

| Goal | Recommended |
|------|-------------|
| Standard | `unipc` or `dpmpp_2m`, 20-30 steps |
| Lower step count | `euler` + `simple`, 12-20 steps |

## What each sampler family actually does

### Euler family
- **Euler** — first-order ODE solver. Simple, robust, deterministic. Slightly worse than DPM++ at the same step count but easier to reason about.
- **Euler ancestral** — same but adds noise at each step. Slightly more diverse outputs; deterministic-seed reproducibility still works.

### Heun
2nd-order ODE solver. Better accuracy per step, but ~2× cost per step. Rarely worth the extra time vs DPM++ 2M.

### DPM family
- **DPM2 / KDPM2** — 2nd-order solver. Mostly historical.
- **DPM++ 2M** — multistep variant. The current default for most SDXL/Flux work. Good quality at 20-30 steps.
- **DPM++ 2M SDE** — adds a stochastic term. Slightly more diverse, slightly noisier per step.
- **DPM++ SDE** (singlestep) — different multistep handling. Less commonly used in 2025.
- **DPM++ 2S a** — ancestral version. Listed in A1111 but not directly in diffusers.

### DDIM
Deterministic sampler from the [[ddim]] paper. Works at 20-50 steps; key advantage is **latent-space interpolation** between two generations (same seeds, deterministic trajectory).

### DDPM
Original ancestral sampler from [[ddpm]]. Needs ~1000 steps for quality. Mostly historical now; rarely used at inference.

### LCM (Latent Consistency Model)
**Requires a model trained or fine-tuned for it** (either an LCM-LoRA loaded on top, or an LCM-distilled base). 4-8 steps total. Massive speed win when applicable; small quality regression vs full-step samplers.

### UniPC (Unified Predictor-Corrector)
2-3 step convergence on simple prompts; up to 10 steps for complex. A relatively recent solver, good for low-latency video/animation pipelines.

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

A different spacing of the noise schedule timesteps — concentrating more steps where the model needs them (high-frequency detail zones) and fewer where it doesn't. Free quality improvement of ~1 FID point at the same step count for SD/SDXL.

For Flux, the model was trained with `sgm_uniform` and is most consistent with that schedule. Karras sigmas on Flux can still work but might subtly shift the output distribution.

## See Also

- [[scheduler]]
- [[ddim]]
- [[ddpm]]
- [[reference/diffusion-course]]
- [[reference/flux-vs-sdxl]]
- [[foundations]]
