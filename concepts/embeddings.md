---
type: concept
tags:
  - foundations
  - embeddings
  - vocabulary
date: 2026-07-01
updated: 2026-07-01
---

# Embeddings (the four kinds)

The word "embedding" is overloaded in this field. At least four distinct things go by the name and confusing them will tangle your mental model. This page is a one-stop disambiguation.

## The general idea (one paragraph)

"To embed something into a vector space" means to represent it as a list of numbers in such a way that meaningful relationships in the original thing show up as geometric relationships among the vectors. Similar inputs land near each other; different inputs land far apart. **Embedding = vector representation of a thing.** Everything below is a specific use of that idea.

## The four kinds you'll meet

| Kind | What it is | Where it lives | Where you'll see it |
|---|---|---|---|
| **Token embedding** | The vector for a single word (or sub-word). Just an entry in a big lookup table indexed by token ID. | Inside every transformer — the very first layer. | Anywhere there's a transformer (text encoders, LLMs). You don't usually touch this directly. |
| **Positional embedding** | A vector added to each token embedding to tell the transformer *where* that token sits in the sequence. Pure-architecture mechanism. | Also inside every transformer, right after the token embedding lookup. | Same as above — internal, not usually exposed. |
| **Sentence / prompt embedding** (also: "conditioning vector") | The pooled or per-token output of an entire text encoder after running a prompt through it. This is what conditions the diffusion model. | The output of [[clip|CLIP]] (or T5, or whatever text encoder the model uses). | The output of the **CLIP Text Encode (Prompt)** node in ComfyUI. Called CONDITIONING on the wire. |
| **Textual Inversion embedding** (often just "embedding" or "TI") | A small learned vector that *replaces a token* — so you can teach the model a new concept and trigger it with a special trigger word. Each one is a `.safetensors` file. | `~/Projects/ComfyUI/models/embeddings/` on disk. | Used inline in prompts via `embedding:my_concept_name` syntax. Predates LoRA and is much smaller (~5-30 KB) but less expressive. |

## A picture of where they sit in the stack

```
Your prompt: "a black coffee mug, embedding:my_lighting_style"
  ↓ tokenize
Token IDs [2056, 4097, 5234, 32001, ...]
  ↓ token embedding lookup            ← Token embeddings live here (lookup table)
  ↓ swap special trigger tokens       ← Textual Inversion embeddings substituted here
Per-token vectors (each = word identity, with TI overrides applied)
  + positional embedding               ← Positional embeddings added here
Per-token vectors (now carrying word + position info)
  ↓ N transformer layers
Contextualized per-token vectors
  ↓ pool / pass through
Final encoder output                   ← Sentence / prompt embedding (the conditioning vector)
  ↓
KSampler uses this to condition the U-Net at every denoising step
```

## Quick rules of thumb

- **If someone says "the embedding came from CLIP"** — they mean the *sentence embedding* (conditioning vector). Output of the whole text encoder.
- **If someone says "drop this embedding into ComfyUI"** — they mean a *textual inversion* file. You put it in `models/embeddings/` and reference it in a prompt.
- **If someone says "the model uses RoPE instead of positional embeddings"** — they mean a different way of injecting position info into the same transformer architecture. Internal architecture talk.
- **If someone says "the token embedding dimension is 768"** — they mean each word maps to a 768-dim vector at the very bottom of the transformer. Architecture talk.

## Note on naming in this wiki

When this wiki says "embedding" unqualified, it usually means a **sentence / prompt embedding** (the conditioning vector) because that's the one most relevant to a diffusion workflow. Other senses are spelled out.

## See Also

- [[clip]] — produces the sentence/conditioning embedding for SDXL and earlier SD models
- [[fine-tuning]] — Textual Inversion is the smallest entry in the fine-tuning options table
- [[diffusion-models]] — what the conditioning vector is *for*, in the denoising loop
- [[u-net]] — the architecture that consumes the conditioning vector
- [[reference/comfyui]] — where the CONDITIONING wire fits in a graph
- [[foundations]] — Week 1 MOC
