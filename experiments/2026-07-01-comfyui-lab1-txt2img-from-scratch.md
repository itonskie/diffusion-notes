---
type: experiment
tags:
  - foundations
  - comfyui
  - lab
  - txt2img
  - hands-on
date: 2026-07-01
updated: 2026-07-01
status: done
---

# Lab 1 — Build a txt2img graph from scratch

The first ComfyUI experiment of Day 3 (redo). You've already seen txt2img *work* — the red ceramic teapot yesterday came from dragging in a pre-built JSON. This lab is the inverse: you build that exact graph from zero, one node at a time, so "what each node does" becomes muscle memory instead of mystery.

**Expected time: ~20 minutes.**

## Hypothesis

A [[diffusion-models|diffusion model]] takes "a text prompt + random noise" and turns it into "a coherent image." The minimum viable ComfyUI graph that does this has **7 nodes**:

1. A model loader — pulls SDXL into memory
2. Two text encoders — one for the positive prompt, one for the negative
3. A latent initializer — provides the starting random noise
4. A sampler — runs the denoising loop (the heart of the graph)
5. A latent decoder — turns the final cleaned latent into pixels
6. An image saver — writes the result to disk

If we wire those 7 correctly, we should get *some* image for *any* prompt. Quality depends on settings, but a graph this simple should never refuse to run.

A **latent** is the compressed numerical representation of an image that the diffusion model actually works on internally — much smaller than a real RGB image, encoded and decoded by the VAE (variational autoencoder) bundled inside the SDXL checkpoint. See [[latent-diffusion]] for the why. For txt2img there's no input image, so we skip the encoder side — but the decoder is the last node before saving.

## Setup

**1. Open ComfyUI in a fresh tab.**

http://127.0.0.1:8188 — make sure the server is running. If not: `cd ~/Projects/ComfyUI && source .venv/bin/activate && python main.py`

**2. Start an empty workflow.** Top menu → **Workflow → New** (or click the **+** next to the open tabs to start fresh).

**3. Add 7 nodes, one at a time.** Two ways to add a node:

- **Right-click empty canvas** → Add Node → \<category\> → \<node name\>
- OR **double-click empty canvas** → search box opens, type the node name, Enter

**Note on names:** every node has *two* names — a **display name** you see in the UI (e.g. "Load Checkpoint") and an **internal class name** that shows up in workflow JSON files (e.g. `CheckpointLoaderSimple`). The table below shows both; use the display name when adding nodes via menu or search.

Add these (don't wire yet — just place them on the canvas):

| # | Display name (UI) | Internal name (JSON) | Category | Why we need it |
|---|---|---|---|---|
| 1 | **Load Checkpoint** | `CheckpointLoaderSimple` | loaders | Loads the full SDXL checkpoint. Outputs MODEL (the U-Net denoiser), CLIP (the text encoder), VAE (the pixel↔latent translator). One file, three outputs. |
| 2 | **CLIP Text Encode (Prompt)** ×1 (positive) | `CLIPTextEncode` | conditioning | Runs the positive prompt through CLIP to produce a conditioning vector — what the sampler should lean *toward*. |
| 3 | **CLIP Text Encode (Prompt)** ×1 (negative) | `CLIPTextEncode` | conditioning | Same thing for the negative prompt — what to push *away* from. The two together drive [[classifier-free-guidance]]. |
| 4 | **Empty Latent Image** | `EmptyLatentImage` | latent | Creates a blank latent at a chosen resolution. This becomes the starting noise the sampler refines. |
| 5 | **KSampler** | `KSampler` | sampling | The denoising loop. Iteratively cleans noise out of the latent guided by the conditioning. The single most important node in the graph. |
| 6 | **VAE Decode** | `VAEDecode` | latent | Takes the cleaned latent and converts it back into a real RGB image. |
| 7 | **Save Image** | `SaveImage` | image | Writes the image to `~/Projects/ComfyUI/output/`. |

**4. Set widget values** (click into each node and edit the fields):

- **Load Checkpoint** — dropdown → `SDXL/sd_xl_base_1.0.safetensors`
- Positive **CLIP Text Encode (Prompt)** — type a positive prompt, e.g. `a photo of a black coffee mug on a wooden desk, morning light, sharp focus`
- Negative **CLIP Text Encode (Prompt)** — type `blurry, low quality, watermark, text`
- **Empty Latent Image** — width=`1024`, height=`1024`, batch_size=`1`
- **KSampler** — seed=any integer, control_after_generate=`randomize`, steps=`25`, cfg=`7.5`, sampler_name=`euler`, scheduler=`normal`, denoise=`1.0`
- **Save Image** — filename_prefix=`lab1` (so you can tell lab outputs apart from yesterday's `ComfyUI_*` ones)

**5. Wire the nodes.** Click and drag from an **output dot** (right side of a node) to an **input dot** (left side of another). Compatible types snap into place; incompatible types ghost out and refuse to connect.

Make these **9 connections** (the count is 9 not 7 because the CLIP and VAE outputs each branch to two places):

| From | → | To |
|---|---|---|
| **Load Checkpoint** MODEL | → | **KSampler** model |
| **Load Checkpoint** CLIP | → | Positive **CLIP Text Encode** clip |
| **Load Checkpoint** CLIP | → | Negative **CLIP Text Encode** clip |
| **Load Checkpoint** VAE | → | **VAE Decode** vae |
| Positive **CLIP Text Encode** CONDITIONING | → | **KSampler** positive |
| Negative **CLIP Text Encode** CONDITIONING | → | **KSampler** negative |
| **Empty Latent Image** LATENT | → | **KSampler** latent_image |
| **KSampler** LATENT | → | **VAE Decode** samples |
| **VAE Decode** IMAGE | → | **Save Image** images |

**6. Hit "Run" (top-right) or "Queue Prompt".**

## Observe

Three things to watch:

1. **Terminal logs** (the window where `python main.py` is running):
   - `got prompt` — ComfyUI received your graph
   - Model loading messages — only on the first run if SDXL isn't already in memory (should be cached from yesterday — if so, you'll skip this)
   - Sampling progress bar — `25/25 [00:XX<00:00, X.XXs/it]`
   - `Prompt executed in X seconds` — total wall time

2. **Canvas** — each node briefly outlines green as it executes, in dependency order: loader → text encoders → empty latent → KSampler (longest pause here, ~25 steps) → VAE decode → save image.

3. **The output image** — appears as a preview at the bottom of the **Save Image** node when generation finishes. Full file in `~/Projects/ComfyUI/output/lab1_00001_.png`.

## Findings

_Fill these in as you observe the run. Bullet points, not essays. Be concrete._

- **First-run wall time (this graph):** **80.26 seconds**. Cold MPS kernel compile for this specific graph layout — second run on the same graph should drop to ~50s.
- **Per-step time (steady state, from the progress bar):** **2.06 s/it** (the value at the end of `25/25 [00:51<00:00, 2.06s/it]`). I first wrote "51 seconds" — that turned out to be the elapsed sampling-loop time (`00:51`), not the per-step rate. The progress bar format is `current/total [elapsed<remaining, per-step-time]`; the `s/it` value at the end is the rate.
- **What the image looks like (one sentence):** two black cups on a wooden table.
- **Anything that surprised you:**
  - How smooth the images are when I zoom in — that's the SDXL VAE decoder doing the upsample from 128×128 latent to 1024×1024 pixels cleanly. Not all VAEs do this well; Flux's 16-channel VAE is even cleaner.
  - The VAE loads *separately and later* than the rest of the checkpoint. The log shows `Requested to load SDXL` (~4.9 GB, the U-Net + CLIPs) at the start, then `Requested to load AutoencoderKL` (~160 MB, the VAE) only AFTER sampling completes. ComfyUI lazy-loads the VAE just-in-time for `VAE Decode`, not upfront with the checkpoint. That means peak memory usage can spike at unexpected times in the graph.
- **Anything that broke or threw an error (and how you fixed it):** no error.

## What this lab teaches

- **The minimum viable diffusion pipeline is 7 nodes.** Every fancier workflow (img2img, inpaint, ControlNet, LoRA loaders, multi-prompt blends) is this graph with extra nodes inserted or substituted in the right place.
- **The data flow is fixed and learnable.** Load weights → encode prompt → start from noise → denoise → decode latent → save. Once that sequence clicks, ComfyUI stops looking like a maze of dots and lines.
- **KSampler is the heart.** It's where the actual generation happens — every other node exists to prepare its inputs or unpack its output. Most of the "interesting knobs" live here (steps, cfg, sampler, scheduler, denoise).
- **Load Checkpoint is misleadingly named.** It loads three completely separate things bundled into one file: the [[u-net|U-Net]] (the denoiser), the CLIP encoder (turns prompts into vectors), and the [[latent-diffusion|VAE]] (translates between pixels and latents). When more complex workflows need just one of these from a different model — that's why we sometimes mix loaders.
- **Every node has two names.** What you see in the UI menus ("Load Checkpoint") and what shows up in workflow JSON files (`CheckpointLoaderSimple`). The class names matter when reading code or community workflows; the display names matter when navigating the UI. Lab 5 will hit a sharper example: the UI calls it "Load Diffusion Model" but the JSON calls it `UNETLoader`.
- **First-run slowness on Mac is one-time.** PyTorch compiles Metal kernels on first use of each operation pattern. Subsequent runs reuse the compiled kernels and run 3-5× faster.

## See Also

- [[diffusion-models]] — the iterative denoising loop you just ran, from first principles
- [[u-net]] — the architecture inside CheckpointLoaderSimple's MODEL output
- [[scheduler]] — what `euler` and `normal` mean inside KSampler
- [[latent-diffusion]] — why VAE encode/decode bookends the diffusion model
- [[classifier-free-guidance]] — why we send a *negative* prompt as well as a positive one
- [[workflows/sdxl-txt2img]] — the pre-built reference graph (essentially what you just built, exported as JSON)
- [[foundations]] — Week 1 MOC
