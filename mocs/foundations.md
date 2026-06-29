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

The goal of this week is to be able to *explain* the four self-test questions at the bottom in plain English, not to derive every equation. Stop reading after Day 2 even if the math feels incomplete — you learn more by training a LoRA than by reading another paper.

## Core path

Work through these in order. Each becomes a page in [[concepts/|concepts/]] as you go.

1. **Hugging Face Diffusion Course, Unit 1** — https://huggingface.co/learn/diffusion-course — the gentlest on-ramp; covers forward/reverse process intuition and the first end-to-end training loop.
2. **Hugging Face Diffusion Course, Unit 2** — same site — fine-tuning, guidance, and conditional generation.
3. **Lilian Weng, "What are Diffusion Models?"** — https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ — the canonical written summary; tighter than the course, more math.

## Papers (skim — don't drown in math)

One page per paper in [[papers/|papers/]].

- **DDPM** — https://arxiv.org/abs/2006.11239 — forward/reverse process formalism. Anchors [[ddpm]].
- **LoRA** — https://arxiv.org/abs/2106.09685 — why rank decomposition works for fine-tuning. Anchors [[lora]] and [[rank-decomposition]].
- **Classifier-Free Guidance** — https://arxiv.org/abs/2207.12598 — how CFG steers generation. Anchors [[classifier-free-guidance]].
- **DiT (Diffusion Transformers)** — https://arxiv.org/abs/2212.09748 — modern Flux / SD3 / Wan 2.2 use this, not U-Net. Anchors [[dit]] and contrasts with [[u-net]].

## Architecture-specific reads

- **Flux model card** on HF — what makes Flux different from SDXL. Capture in [[reference/flux-vs-sdxl]].
- **One scheduler comparison post** (DPM++ / Euler / LCM). Capture in [[reference/scheduler-comparison]] and concept page [[scheduler]].

## Self-test (interview answers — write these out)

When you can answer all four in plain English without notes, Day 1-2 is done.

1. Why does diffusion need many denoising steps and how does CFG affect that? → see [[denoising-steps]], [[classifier-free-guidance]]
2. What does LoRA actually do to the weight matrices? Why rank 8 vs rank 32? → see [[lora]], [[rank-decomposition]]
3. What's the difference between U-Net (SD 1.5 / SDXL) and DiT (Flux / SD3 / Wan 2.2)? → see [[u-net]], [[dit]]
4. What does "scheduler" mean and why does it matter for training vs inference? → see [[scheduler]]

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
