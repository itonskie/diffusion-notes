---
type: concept
tags:
  - foundations
  - fine-tuning
  - training
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Fine-Tuning

Continuing training of a pre-trained diffusion model on a new dataset or task, instead of training from scratch. The default approach for adapting any modern image/video diffusion model — training a foundation model from scratch on a single dataset is rarely tractable.

## Why fine-tune

- **Compute economics** — Flux took thousands of GPU-days. Fine-tuning a Flux model on a 20-image dataset takes 30 minutes on a single 4090.
- **Better starting point** — the base model already knows what edges, textures, faces, and composition look like. You only have to teach it the delta.
- **Surprising generalization** — fine-tuning works even when the new domain is very different from the base. The HF course example fine-tunes a model trained on LSUN Bedrooms onto WikiArt paintings and gets coherent painterly bedrooms in 500 steps.

## Approaches

| Approach | What gets updated | Storage cost | Notes |
|----------|-------------------|--------------|-------|
| Full fine-tuning | All model weights | Full model size (e.g. 12 GB for Flux) | Best quality, slowest, easy to overfit |
| [[lora\|LoRA]] | Low-rank adapter on selected layers | 20-300 MB | Default modern choice for style/character/product |
| DreamBooth | Full weights + class-preservation loss | Full model size | Old technique, mostly superseded by LoRA |
| Textual Inversion | Single learned embedding vector | Few KB | Limited capacity; good for one concept |
| Hypernetworks | Small network that modulates main weights | Small | Mostly historical; LoRA replaced it |

For this sprint, the default is [[lora|LoRA]] on Flux. Other approaches show up in the literature but rarely justify themselves over LoRA in 2025.

## What changes vs from-scratch training

Mechanically, fine-tuning is the same as pre-training: forward pass, noise prediction loss, backward pass. Differences are practical:

- **Learning rate** — much smaller (often 1e-4 to 1e-5 vs 1e-3+ from-scratch) to avoid destroying what's already learned.
- **Steps** — often 500-5000 instead of millions.
- **Dataset size** — can work with 15-50 images vs millions for pre-training.
- **Risk** — catastrophic forgetting (loses the base model's general capability) and overfitting (memorizes training images instead of learning patterns) are the two main failure modes.

## When fine-tuning vs guidance

There's overlap between [[guidance]] and fine-tuning — both add control over the base model's outputs. Rough rule:

- **Guidance** is inference-time and free of training, but the strength of effect is limited and inference becomes slower (extra forward passes).
- **Fine-tuning** is training-time work but inference cost is unchanged. Effect can be much stronger and more reliable.

For "make Flux always generate in this specific style," fine-tune. For "make Flux generate slightly bluer images right now," guidance.

## Practical notes for the sprint

- Captions matter. For [[lora|LoRA]] on Flux, captions in `.txt` files alongside each image are how you tell the model what concept it's learning. Bad captions = bad LoRA.
- Dataset size: 15-25 images is enough for a character or style LoRA on Flux. More isn't always better — quality dominates quantity.
- Repeats × epochs ≈ total steps. AI-Toolkit and others let you set either explicitly. Target ~1000-3000 total steps for a Flux LoRA, more if it's a complex concept.

## See Also

- [[lora]]
- [[guidance]]
- [[classifier-free-guidance]]
- [[diffusion-models]]
- [[scheduler]]
- [[reference/diffusion-course]]
- [[foundations]]
