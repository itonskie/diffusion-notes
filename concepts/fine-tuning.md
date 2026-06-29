---
type: concept
tags:
  - foundations
  - fine-tuning
  - training
date: 2026-06-29
updated: 2026-06-29
---

# Fine-Tuning

The picture: someone else spent millions of dollars and thousands of GPU-days training a model that already knows what edges, faces, textures, lighting, and composition look like. You don't redo any of that. You take their finished model and run a small amount of extra training on a tiny dataset of your own to teach it one new thing — a specific character, a specific style, a specific product.

That's fine-tuning. Continuing training of a pre-trained [[diffusion-models|diffusion model]] on a new dataset, instead of starting from scratch. For modern image and video diffusion this is basically the only option — training a foundation model (a big general-purpose pre-trained model like Flux or SDXL) from zero on your own data is not financially or computationally tractable for anyone outside a handful of labs.

## Why fine-tune

- **Compute economics.** Flux took thousands of GPU-days to pre-train. Fine-tuning a Flux model on a 20-image dataset takes 30 minutes on a single 4090.
- **Better starting point.** The base model already knows what edges, textures, faces, and composition look like. You only have to teach it the **delta** — the difference between what it already does and what you want it to do.
- **Surprising generalization.** Fine-tuning works even when the new domain is very different from the base. The HF (Hugging Face) course example fine-tunes a model trained on LSUN Bedrooms (a dataset of bedroom photos) onto WikiArt paintings and gets coherent painterly bedrooms in 500 steps. The model isn't starting over — it's bending what it already knows.

## Approaches

Different ways to do that "extra training." They differ mainly in *which parts of the model you let move.*

| Approach | What gets updated | Storage cost | Notes |
|----------|-------------------|--------------|-------|
| Full fine-tuning | All model weights | Full model size (e.g. 12 GB for Flux) | Best quality, slowest, easy to overfit |
| [[lora\|LoRA]] | Low-rank adapter on selected layers | 20-300 MB | Default modern choice for style/character/product |
| DreamBooth | Full weights + class-preservation loss | Full model size | Old technique, mostly superseded by LoRA |
| Textual Inversion | Single learned embedding vector | Few KB | Limited capacity; good for one concept |
| Hypernetworks | Small network that modulates main weights | Small | Mostly historical; LoRA replaced it |

Plain-English decode of the techniques:

- **Full fine-tuning** — touch every weight in the model. Maximum flexibility, maximum risk of breaking the base model's general capabilities, full model size on disk for every fine-tune.
- **[[lora|LoRA]]** (Low-Rank Adaptation) — freeze the base model entirely. Bolt on tiny "patch" matrices to selected layers and only train those. Ship the patch as a small file. See [[lora]] for the full picture.
- **DreamBooth** — full fine-tuning plus an extra loss term that says "don't forget what the word 'dog' means in general while you're learning *my* dog." Heavy. Superseded by LoRA for most cases.
- **Textual Inversion** — don't touch the model at all. Instead, learn a single new word vector — a numerical representation of a word that the model can take as input — that, when used in a prompt, conjures your concept. Tiny file. Limited in what it can express.
- **Embedding** (the thing Textual Inversion produces) — a vector of numbers that represents a piece of text inside the model. Words are turned into embeddings before they're fed in.
- **Hypernetworks** — a small auxiliary network that produces multipliers for the main model's weights at runtime. Was popular briefly. Effectively replaced by LoRA, which is simpler and works as well or better.

For this sprint, the default is [[lora|LoRA]] on Flux. Other approaches show up in the literature but rarely justify themselves over LoRA in 2025.

## What changes vs from-scratch training

Mechanically nothing changes — it's the same training loop. Forward pass through the model, noise-prediction loss (the standard diffusion loss: predict the noise that was added to this training image), backward pass, weight update. The differences are all in the dials:

- **Learning rate** — how big each gradient step is. Much smaller than from-scratch (typically 1e-4 to 1e-5 instead of 1e-3 or higher). If you take big aggressive steps, you'll blow away what the base model already knows.
- **Steps** — number of training updates. 500–5,000 total instead of millions.
- **Dataset size** — 15–50 images can be enough instead of the millions used to pre-train.
- **Risk profile** — two main failure modes to watch for:
  - **Catastrophic forgetting** — the model loses general capability while learning your specific concept. It forgets how to draw, say, a horse because all your training images are cats.
  - **Overfitting** — the model memorizes the training images themselves instead of learning the *pattern* they share. Generates near-copies of your dataset on any prompt.

## When fine-tuning vs guidance

There's overlap with [[guidance]] — both add control over what the base model outputs. Quick decode: **guidance** is a family of techniques that nudge the model's behavior at *generation time* without retraining anything, usually by computing something extra and steering the denoising step with it. [[classifier-free-guidance|Classifier-free guidance]] is the most common form. Rough rule for when to use which:

- **Guidance** — inference-time, no training, but the strength of effect is limited and inference becomes slower because you run extra forward passes per denoising step.
- **Fine-tuning** — training-time work upfront, but inference cost is unchanged afterwards. The effect can be much stronger and more reliable.

Heuristic: "make Flux **always** generate in this specific style" → fine-tune. "Make Flux generate slightly bluer images on this one prompt right now" → guidance.

## Practical notes for the sprint

- **Captions matter.** For [[lora|LoRA]] on Flux, captions in `.txt` files alongside each image are how you tell the model what concept it's learning. Bad captions = bad LoRA. Details in [[lora]].
- **Dataset size.** 15–25 images is enough for a character or style LoRA on Flux. More isn't always better — quality dominates quantity. A dataset of 15 great images beats 100 mediocre ones.
- **Repeats × epochs ≈ total steps.** AI-Toolkit and similar training tools let you set either explicitly. (Decode: an **epoch** is one full pass through the dataset; **repeats** is how many times each image is seen per epoch.) Target ~1,000–3,000 total steps for a Flux LoRA, more if it's a complex concept.

## See Also

- [[lora]]
- [[guidance]]
- [[classifier-free-guidance]]
- [[diffusion-models]]
- [[scheduler]]
- [[reference/diffusion-course]]
- [[foundations]]
