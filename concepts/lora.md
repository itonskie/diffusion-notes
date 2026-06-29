---
type: concept
tags:
  - foundations
  - lora
  - fine-tuning
date: 2026-06-29
updated: 2026-06-29
---

# LoRA (Low-Rank Adaptation)

LoRA is the cheap, default way to fine-tune a big model. Instead of nudging every weight in the model, you bolt on two tiny matrices that together act like a small patch on top of it. Train the patch. Keep the base model frozen and untouched. Ship the patch as a small file.

Why we care: Flux base is ~12 GB. A Flux LoRA is 100–300 MB. Training takes an hour on a rented GPU instead of days on a cluster. At image-generation time it adds essentially no cost.

For the original paper itself, see [[papers/lora]]. This page is the **working understanding for the sprint**.

## The picture

A neural network is a stack of layers. Each layer has a big matrix of numbers — its **weights** — that get multiplied with the layer's input to produce the output. When you "train" a model, you're nudging those numbers. When you "fine-tune," you're nudging them further on a more specific task.

Full fine-tuning means nudging *every* weight. For Flux that's about 12 billion numbers. The catch: most of them barely move, and the ones that do tend to move *together*. You're storing and computing a lot of zeros.

LoRA's bet: the *change* you want — the difference between the base model and your fine-tuned model — is itself simple. It lives in only a few directions. So instead of learning that change as one giant matrix `ΔW`, learn it as the product of two skinny ones:

```
ΔW  =  B · A
```

Shape-wise: if `W₀` is a big square (say 1024 × 1024 — about 1 million numbers), then `B` is a tall skinny rectangle (1024 × r) and `A` is a wide short rectangle (r × 1024). Pick r = 16. Now you have 1024·16 + 16·1024 ≈ 33,000 numbers. Multiply `B` and `A` together and you get something the right shape (1024 × 1024) to patch back onto `W₀`. That's a ~30× reduction in what you have to train and store.

That's the whole technique. Freeze the base. Train `A` and `B`. At runtime, add their product back on as a patch.

## Why "low rank"

**Rank** is a linear algebra term. It measures how independent a matrix's directions of behavior really are. A 1024 × 1024 matrix has at most rank 1024, but it might actually be lower — its columns might be repeating each other in subtle ways. A matrix that *looks* 1024 × 1024 but is actually `B·A` with r=16 has rank at most 16. It only has 16 independent directions dressed up in a big shape.

The bet of LoRA is that the update you need *naturally* has low rank — that fine-tuning a model on a new concept really only needs a few dozen new directions, not a thousand. Empirically this turns out to be true, which is why the technique works at all.

The math intuition for *why* this is true lives in [[rank-decomposition]].

## The arithmetic, for when you see it in configs

```
W = W₀ + (α / r) · B · A
```

- `W₀` — the base weight matrix. Frozen. Never changes during LoRA training.
- `B` — initialized to **zero**, so that when training starts the patch contributes nothing and the model behaves exactly like the base. Gradients have to discover what to put here.
- `A` — initialized random.
- `α` (alpha) — a single number that scales how strongly the patch contributes.
- `r` — the rank you picked.

You don't have to memorize this. It's here so you recognize `r`, `alpha`, `rank` when you see them in an AI-Toolkit config or a ComfyUI loader.

## Which layers in a diffusion model get LoRA-ed?

A diffusion model isn't one matrix — it's hundreds of them, stacked across [[u-net|U-Net]] blocks (older models like SDXL) or [[dit|DiT]] blocks (modern models like Flux). You don't have to put a LoRA on every single matrix. Different training tools pick different defaults.

The original LoRA paper, which was for language models, only patched the Q and V matrices inside the attention mechanism. Diffusion LoRAs go broader because they're not just nudging task behavior — they're teaching the model new *visual concepts*, which is harder.

| Tool | Default layer selection | Notes |
|------|-------------------------|-------|
| AI-Toolkit | All linear in attention + feed-forward, all conv in [[u-net]] | Most aggressive default |
| Kohya | Attention only by default | More conservative |
| Musubi Tuner | All linear in [[dit\|DiT]] blocks | Video diffusion |

Adapt too few layers → the LoRA can't fit the concept (especially characters). Adapt too many → bigger file than needed and easier to overfit on a small dataset.

## Picking the rank `r`

The bigger `r`, the more "room" the patch has to fit the concept. But bigger `r` also = bigger file, slower training, easier overfitting. Rough guide for Flux:

| Concept type | Typical `r` |
|--------------|-------------|
| Simple style transfer | 8–16 |
| Character or face | 16–32 |
| Complex style or multiple concepts | 32–64 |
| Very rich / photorealistic concept | 64–128 |

Language-model LoRAs use r = 1–8. Vision needs more because visual concepts are *richer* — more independent directions of variation than a text task.

Diminishing returns kick in fast. Going from r=32 to r=64 usually doubles the file size and barely improves quality. The right `r` is the smallest one that still captures what you want.

## Alpha and "strength" at inference

`α` is a knob on top of `r` that controls how strongly the patch contributes. The common convention now:

- **During training,** set `α = r`. That makes `α/r = 1` and the patch contributes at its natural learned scale.
- **At inference,** use the LoRA loader's **strength slider** to dial that up or down. Strength 1.0 = trained intensity. Strength 0.5 = half-effect. Strength 1.5 = pushed harder than it learned. You can stack multiple LoRAs at different strengths.

In ComfyUI, the strength slider is doing exactly that scaling under the hood.

## Training cost, single 4090 or A6000

| Concept | Steps | Wall time | Vast.ai cost |
|---------|-------|-----------|--------------|
| Small style (~15 images, r=16) | 1,000 | 30–60 min | $0.50–$1.50 |
| Character (~25 images, r=32) | 2,000 | 1–2 hr | $1–$3 |
| Rich style (~50 images, r=64) | 3,000 | 3–5 hr | $3–$10 |

Assumes Flux-dev at sane settings. Add 30–50% if training Flux-schnell or running at 512×512 instead of 1024×1024.

## Captions: where the quality is won or lost

Each training image needs a caption sitting next to it in a `.txt` file (same filename, different extension). The captions teach the model two things: what your concept *is*, and what's *not* part of your concept.

- **Trigger word.** Invent a unique token (e.g. `OHWX person`, `myartstyle`) and put it in every caption. At inference you prompt with that token to activate the LoRA.
- **Describe what varies, not what stays constant.** If every image is the same character in different scenes, caption the *scene*, not the character. The model figures out: "what's the same in every image is what this LoRA *is*; what changes is detail I should respect on demand."
- **Be wordy.** For Flux, longer descriptive captions tend to outperform short ones.

Bad captions are the single biggest cause of bad LoRAs. People usually auto-caption their dataset with a vision-language model (Florence-2, Joy-Caption, BLIP-3) and then edit by hand — fix the trigger word, fix anything over- or under-described.

## Merging vs runtime application

Once trained, two ways to use the LoRA at generation time:

- **Runtime.** Keep `A` and `B` separate from the base. Every forward pass computes `W₀·x + (α/r)·B·A·x` on the fly. Tiny extra cost per step, but you can hot-swap LoRAs and stack many.
- **Merge.** Compute `W₀ + (α/r)·B·A` once, save the merged model, throw away `A` and `B`. Inference is now identical to a fully fine-tuned model. Faster per step but you can't cheaply swap.

ComfyUI does runtime application by default. That's why you can chain three LoRAs in a single workflow without merging anything.

## What this looks like in the sprint

For Week 2: pick a concept, train one Flux LoRA, see it work in ComfyUI.

1. Pick concept (style, character, product) — Week 2 decision
2. Collect 15–50 images, write a caption `.txt` for each
3. Use AI-Toolkit's Flux LoRA example config; change only the dataset path and output name
4. Train on Vast.ai (A6000 or 4090), ~1–2 hours
5. Pull the `.safetensors` (the trained LoRA file format) to your Mac, load in ComfyUI, prompt with the trigger word
6. Side-by-side comparison: same seed, same prompt, with and without the LoRA

Successful configs go in `configs/`. The `.safetensors` files do not (already gitignored).

## See Also

- [[papers/lora]] — the original paper
- [[rank-decomposition]] — why low rank is enough
- [[fine-tuning]] — broader context, what other fine-tuning techniques exist
- [[diffusion-models]] — what the base model actually does
- [[u-net]], [[dit]] — the two architectures whose layers we're patching
- [[foundations]] — the Week 1 MOC
