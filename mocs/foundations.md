---
type: moc
tags:
  - foundations
  - week-1
date: 2026-06-29
updated: 2026-06-29
---

# Foundations

The reading path for diffusion fundamentals — Week 1 of the [diffusion sprint](https://github.com/itonskie/diffusion-sprint).

## What you're about to learn, in one paragraph

A **diffusion model** is a neural network that turns pure noise into a sensible image (or video, or audio) by *gradually denoising* it across many small steps. To train one, you take real images, add noise to them in known amounts, and teach a network to predict the noise. To generate one, you start with random noise and run the network in reverse, peeling noise off step by step until an image emerges. Modern image models like **Stable Diffusion** and **Flux** are this — plus tricks that make it cheap (run the diffusion in a compressed "latent" space, not raw pixels) and steerable (let a text prompt push the model toward what you want).

The goal of Week 1 is to be able to *explain* the four self-test questions at the bottom in plain English. Not derive every equation. Stop reading after Day 2 even if the math feels incomplete — you learn more by training a LoRA than by reading another paper.

## Read in this order

Each page below is a digested, Feynman-style explanation. One line per page tells you what you'll get out of it.

**Start here — the big picture**
- [[generative-models]] — what "generative modeling" means and where diffusion fits among GANs, VAEs, and autoregressive models
- [[diffusion-models]] — the core idea: iterative noise → image, how it's trained, how it samples, and why it works
- [[latent-diffusion]] — the trick that makes Stable Diffusion and Flux affordable: run diffusion on a small compressed version of the image, not pixels

**Architectures — the network shape that does the denoising**
- [[u-net]] — the older U-shaped network used in SD 1.5 and SDXL, with the three ways text gets injected
- [[dit]] — the transformer-shaped network used in Flux, SD3, and Wan 2.2; replaces U-Net for big modern models

**How a single denoising run goes**
- [[scheduler]] — the recipe for "how much noise at each step" — different recipes (DDPM, DDIM, DPM++, LCM) trade speed vs. quality
- [[guidance]] — the umbrella idea of "nudge the denoising toward what we want"
- [[classifier-free-guidance]] — the trick everyone actually uses to make a text prompt steer generation, without training a separate classifier

**Fine-tuning — how you teach an existing model new tricks**
- [[fine-tuning]] — the spectrum of options for adapting a pretrained model (full fine-tune, LoRA, textual inversion, ControlNet…)
- [[lora]] — the cheap default: bolt on two skinny matrices that act as a small patch. Train the patch, ship a 200 MB file.
- [[rank-decomposition]] — the linear-algebra reason LoRA is allowed to work (low-rank approximation, SVD)

**Papers** — original sources for the techniques above. Each page is "Feynman summary + verbatim abstract + math sketch."
- [[ddpm]] — the foundational diffusion paper (2020)
- [[papers/lora]] — the original LoRA paper (NLP, 2021); diffusion application followed
- [[papers/cfg]] — Classifier-Free Guidance (2022); two-pass trick that became the default
- [[papers/dit]] — Diffusion Transformer (2022); the architecture behind Flux and SD3

**Reference** — comparison tables and resource maps
- [[reference/diffusion-course]] — Hugging Face Diffusion Course, Units 1 + 2 (the gentle on-ramp)
- [[reference/flux-vs-sdxl]] — picking between Flux and SDXL: VRAM, license, settings, where each shines
- [[reference/scheduler-comparison]] — sampler name lookup across A1111 / Diffusers / ComfyUI plus per-model recommendations

## What was originally on the reading list

These are the sources that got ingested into the wiki above. You can read them directly if a wiki page feels thin, but the wiki *is* the synthesis — start with the pages, not the sources.

**Core**
- Hugging Face Diffusion Course, Units 1 + 2 — https://huggingface.co/learn/diffusion-course
- Lilian Weng, "What are Diffusion Models?" — https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

**Papers (skim — don't drown in math)**
- DDPM — https://arxiv.org/abs/2006.11239
- LoRA — https://arxiv.org/abs/2106.09685
- Classifier-Free Guidance — https://arxiv.org/abs/2207.12598
- DiT — https://arxiv.org/abs/2212.09748

**Architecture-specific**
- Flux model card on HF — distilled into [[reference/flux-vs-sdxl]]
- Scheduler comparison reads — distilled into [[reference/scheduler-comparison]] and [[scheduler]]

## Status: foundations reading complete

All Week 1 sources are ingested and digested. The wiki now covers the four self-test questions below. Next is *practice*, not more reading.

- [ ] Day 3: ComfyUI local install — 3 prebuilt workflows
- [ ] Day 4: Vast.ai + AI-Toolkit, tiny dataset, verify training pipeline runs end-to-end
- [ ] Day 5: First trained Flux LoRA + inference with/without comparison
- [ ] Day 6-7: Publish the notes repo + write the Week 2 LoRA subject decision

## Self-test (interview answers — write these out)

When you can answer all four in plain English without notes, Day 1-2 is done.

1. **Why does diffusion need many denoising steps, and how does CFG affect that?** → see [[diffusion-models]], [[classifier-free-guidance]]
2. **What does LoRA actually do to the weight matrices? Why rank 8 vs rank 32?** → see [[lora]], [[rank-decomposition]]
3. **What's the difference between U-Net (SD 1.5 / SDXL) and DiT (Flux / SD3 / Wan 2.2)?** → see [[u-net]], [[dit]]
4. **What does "scheduler" mean and why does it matter for training vs inference?** → see [[scheduler]]

## All foundations pages

```base
filters:
  and:
    - file.hasTag("foundations")
    - 'type != "moc"'
views:
  - type: table
    name: Pages
    order:
      - file.name
      - type
      - updated
```

## See Also

- [[index|Wiki Index]]
- [[schema|Wiki Schema]]
- [diffusion-sprint Week 1 README](https://github.com/itonskie/diffusion-sprint/blob/main/week-1-foundations/README.md)
