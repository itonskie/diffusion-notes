---
type: concept
tags:
  - foundations
  - clip
  - conditioning
  - text-encoder
date: 2026-07-01
updated: 2026-07-01
---

# CLIP

The **translator** that sits between your text prompt and the diffusion model. English in, a vector of numbers out — the "meaning" of the prompt in a form a neural network can actually condition on. Stable Diffusion 1.x, 2.x, SDXL, and basically every pre-Flux/SD3 image model uses CLIP to do this.

Full name: **Contrastive Language–Image Pre-training**. Released by OpenAI in 2021. The acronym tells you the whole training story.

## The picture

A diffusion model can't read your prompt. It operates on **conditioning vectors** — fixed-size lists of floating-point numbers that encode meaning in a high-dimensional space. Your prompt "a photo of a black coffee mug" needs to become one of those vectors before the model can use it.

That's CLIP's job. It's a neural network that takes a string of text (or an image) and outputs a vector. Two strings with similar meaning ("a black coffee mug" and "a dark ceramic cup") produce similar vectors. Two with different meaning ("a black coffee mug" and "a red sports car") produce very different vectors. Once you have that mapping, the diffusion model can be trained to make images that, when CLIP-encoded, produce vectors close to the prompt vector.

So generation works like this:

1. You type a prompt → CLIP text encoder → conditioning vector
2. KSampler starts from random noise
3. At each denoising step, the U-Net is given the conditioning vector and the current noisy latent, and asked: "what should I subtract to nudge this latent closer to something that CLIP would say matches that conditioning vector?"
4. After enough steps, the latent decodes into an image whose CLIP-encoded vector is close to the prompt's vector

The whole thing is a fixed-point dance between the U-Net (which generates) and CLIP (which judges meaning).

## How it was trained — and why "contrastive"

CLIP was trained on **400 million image-text pairs** scraped from the public web. Each pair = an image and its accompanying caption (alt text, surrounding text on the page, whatever the scraper grabbed).

The training objective: for each batch of N pairs, run all N images through the image encoder and all N captions through the text encoder. You now have N image-vectors and N text-vectors. The model is rewarded for making the matching pairs' vectors similar (high dot product) and the mismatched pairs' vectors dissimilar (low dot product).

That's the **contrastive** part. The model learns by contrast — it doesn't get told "this caption means X"; it gets told "this caption matches this image, but not those other 31 images in the batch." Over 400M examples, it learns a shared embedding space where semantic similarity in text and visual similarity in images line up.

## The two halves

CLIP outputs two trained components, both targeting the same vector space:

- **Text encoder** — a transformer that eats a tokenized string and outputs a vector. This is what every text-to-image model uses.
- **Image encoder** — a Vision Transformer (ViT) or ResNet that eats pixels and outputs a vector. Used for image search, classification, image-guided generation, and zero-shot tasks. Not used in txt2img.

The whole "you can search for a cat photo by typing 'cat'" capability that powers modern image search is built on CLIP — text query → vector → nearest-neighbor lookup in pre-computed image vectors.

## SDXL uses *two* CLIPs

Stable Diffusion 1.x and 2.x used a single CLIP text encoder. SDXL doubled down: it uses **two text encoders in parallel** and concatenates their outputs:

- **CLIP-ViT-L/14** — the original OpenAI CLIP, smaller (~123M params)
- **OpenCLIP-ViT-bigG/14** — a larger open-source CLIP trained by the LAION non-profit (~695M params)

Both encode your prompt independently; their outputs are concatenated into one richer conditioning tensor. SDXL was trained from scratch to consume that combined signal. Empirically the dual-encoder setup captures more nuance — short and long prompts both work better than they did under single-CLIP SD models.

The single **CLIP Text Encode (Prompt)** node in ComfyUI handles both encoders behind the scenes when the loaded model is SDXL. You don't see two text nodes; the loader's CLIP output already packages both inside.

## The 77-token limit (and the ComfyUI workaround)

CLIP's text encoder has a **fixed input length of 77 tokens** (~50–60 English words, roughly). Anything longer gets truncated.

ComfyUI gets around this by **chunking** long prompts at the CLIP Text Encode level — under the hood, prompts longer than 77 tokens are split into multiple 77-token windows, each encoded separately, and their conditioning tensors are concatenated. So you can paste a 300-word prompt and ComfyUI will Do The Right Thing, but the model still only attends to 77 tokens at a time. Quality starts degrading on very long prompts because the model can't relate words across chunks.

Practical implication: long prompts work but are less coherent than they appear. Front-load the important stuff.

## What came after CLIP

Newer diffusion models have moved off pure CLIP:

- **Stable Diffusion 3** uses three text encoders: CLIP-L + OpenCLIP-G + **T5-XXL** (a Google language model, much larger and more semantically capable).
- **Flux** drops CLIP almost entirely and uses **T5-XXL + a small CLIP-L** only for style cues.

T5 brings much richer language understanding because it's a full language model, not just a text-image embedder. The trade-off: T5-XXL is enormous (~5 GB of weights) which is part of why Flux is harder to run on consumer GPUs.

For this sprint's purposes (SDXL → Flux progression), you'll touch CLIP-only conditioning in SDXL and dual-encoder T5+CLIP conditioning when you move to Flux on Vast.ai.

## See Also

- [[embeddings]] — disambiguating CLIP's output from token, positional, and Textual Inversion embeddings
- [[diffusion-models]] — what KSampler is doing with CLIP's conditioning vector at each step
- [[classifier-free-guidance]] — the two-pass trick that exaggerates the influence of CLIP conditioning
- [[guidance]] — the umbrella idea CFG is one instance of
- [[u-net]] — the network on the receiving end of CLIP's conditioning
- [[latent-diffusion]] — what space the conditioning is being applied in
- [[reference/flux-vs-sdxl]] — how text-encoder choices differ across SDXL, SD3, and Flux
- [[reference/comfyui]] — where the CLIP Text Encode node fits in the graph
- [[foundations]] — Week 1 MOC
