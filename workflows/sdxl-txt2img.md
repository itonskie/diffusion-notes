---
type: workflow
tags:
  - foundations
  - sdxl
  - workflow
  - txt2img
date: 2026-06-29
updated: 2026-06-29
---

# SDXL txt2img

The simplest possible SDXL graph: type a prompt, get a 1024×1024 image. Every other ComfyUI workflow — img2img, inpaint, controlnet, LoRA stacks — is this graph with extra stuff bolted onto it. Get this one in your bones first.

## Purpose

Plain text-to-image. You give it a positive prompt (what you want), a negative prompt (what you don't want), and it returns a single image at SDXL's native resolution. Reach for this when you're sketching ideas, dialing in a prompt, or sanity-checking that your install actually works. It's also the right starting point any time you're about to build a more complex graph — copy it, then add nodes on top.

## How to load

1. Make sure ComfyUI is running. The launcher prints a URL — usually `http://127.0.0.1:8188`.
2. Open that URL in a browser.
3. Drag `sdxl-txt2img.json` (the file next to this page) onto the canvas. Alternatively: click the menu, **Load**, pick the file.
4. Hit **Queue Prompt**. First run is slow because the checkpoint has to load into memory; subsequent runs are fast.

## Node walkthrough

Seven nodes, left to right.

- **CheckpointLoaderSimple** — loads `sd_xl_base_1.0.safetensors` from `ComfyUI/models/checkpoints/`. A checkpoint is a single file holding three things stitched together: the [[u-net]] (the actual denoiser), the [[clip-text-encoder|CLIP text encoder]] (turns your prompt into numbers), and the [[vae|VAE]] (turns the model's compressed "latent" output back into a real RGB image). This one node hands those three pieces off to the rest of the graph.
- **CLIPTextEncode (positive)** — takes your prompt as a string and runs it through CLIP to produce **conditioning**: a chunk of numbers that tells the [[u-net]] "aim toward this." This is what makes the model paint what you asked for instead of random noise.
- **CLIPTextEncode (negative)** — same node type, second copy. The text here gets pushed *away* from. Empty string is fine; common practice is to dump generic junk you never want ("blurry, watermark, extra fingers"). How "push away" actually works lives in [[classifier-free-guidance]].
- **EmptyLatentImage** — generates a blank canvas in **latent space**. Latent space is SDXL's compressed internal representation: a 1024×1024 image is really a 128×128×4 tensor in there. The whole denoising loop happens at that smaller size, which is why diffusion is fast enough to be practical. See [[latent-diffusion]] for the why.
- **KSampler** — the engine. Starts with the random noise inside the latent image and runs the [[u-net]] for `steps` iterations, each one nudging the noise a little closer to a real image that matches the conditioning. The [[scheduler]] decides *how big each nudge is* across those steps. Output: a clean latent that no longer looks like noise.
- **VAEDecode** — takes that clean latent (128×128×4 numbers) and runs it through the [[vae]] decoder to produce an actual 1024×1024 RGB image you can look at.
- **SaveImage** — writes the PNG to `ComfyUI/output/`. The widget value (`ComfyUI`) is the filename prefix; the node auto-numbers from there.

The data flow: checkpoint splits three ways → text becomes conditioning → blank latent + conditioning + model go into the sampler → latent comes out → VAE decodes → save. That's it. The whole field of [[diffusion-models]] is mostly variations on this seven-node pattern.

## Defaults explained

**steps = 25.** Each step is one denoising iteration. More steps = more refinement, but with diminishing returns and a hard ceiling — past about 30 you're usually just spending compute for no visible quality gain. 25 is the comfortable middle for `euler` on SDXL. If you switch the sampler to `dpmpp_2m` you can drop to 20 and lose nothing.

**cfg = 7.5.** Short for **classifier-free guidance scale** — how hard the sampler pushes toward the positive prompt and away from the negative. cfg=1 ignores your prompt entirely. cfg=7-8 is the SDXL sweet spot. Crank it above 12 and images start to look fried (oversaturated, deep-fried-meme look). See [[classifier-free-guidance]] for what's happening under the hood.

**sampler = euler, scheduler = normal.** The sampler is the algorithm that takes one denoising step; the scheduler is the timetable that says how much noise to remove at each step. `euler + normal` is the boring reliable baseline that works on every model. It's a fine default. Faster/better choices (`dpmpp_2m + karras`, `dpmpp_sde + karras`) exist but they trade reliability for speed — start here, change later.

**denoise = 1.0.** "How much of the latent should I treat as noise to remove." 1.0 = start from pure noise (correct for txt2img — there's nothing in the canvas to preserve). For img2img this drops to 0.5–0.8 so the input image partially survives. Always 1.0 here.

**seed, control_after_generate = randomize.** Seed is the integer that determines the starting noise pattern. Same seed + same prompt + same settings = bit-identical image. `randomize` means each Queue Prompt rolls a fresh seed; switch it to `fixed` when you're iterating on a prompt and want to compare apples to apples.

## Gotchas

- **The KSampler "denoise" widget is the last one.** Don't confuse it with cfg. cfg dials prompt strength; denoise dials how much noise to start from.
- **1024×1024 is mandatory for SDXL base.** It was trained at 1024. Drop to 512 and you get garbage limbs and warped faces — the model literally cannot produce a coherent image at SD1.5 sizes. Other supported aspect-ratio buckets exist (1152×896, 896×1152, 1216×832, etc.) — keep the total pixel count near 1024².
- **First run is slow.** Checkpoint load on an M4 Max can take 30-60 seconds the first time. Subsequent runs reuse the loaded model and the sampling itself takes 15-30 seconds at these settings.
- **No refiner here.** This is base-only. SDXL was released as a two-stage pipeline (base + refiner) but in practice the base alone is plenty for most uses, and skipping the refiner halves your VRAM footprint and wall time. Add the refiner later if you actually need it.
- **The previewer (TAESDXL) is decorative.** If you installed it via ComfyUI-Manager, you'll see a tiny live preview while sampling. It uses a separate tiny VAE to decode intermediate latents fast. It doesn't affect the final image at all — that's still VAEDecode's job.
- **Filename in CheckpointLoaderSimple must match exactly.** If the model on disk is named differently (e.g. `sdxl_base_1.0.safetensors` without the underscore, or `.fp16.safetensors`), open the dropdown in the UI and pick the actual filename. The JSON value is just a default.

## Screenshot

First run, 2026-06-30. Prompt: `a photo of a red ceramic teapot on a wooden table, soft window light, sharp focus, detailed`. 25 steps, euler/normal, CFG 7.5, 1024×1024, seed `156680208700286`. Mac M4 Max, 36 GB unified memory.

![[assets/sdxl-txt2img-result.png]]

**Wall time: 61.81 s** — cold start including MPS kernel compile and first-load of SDXLClipModel + SDXL UNET + AutoencoderKL. Steady-state at 1.92 s/iteration. Warm runs (same checkpoint already in memory) should land at ~15-25 s.

## See Also

- [[mocs/foundations]] — Week 1 MOC
- [[sdxl-img2img]] — same graph, but seeded with an existing image instead of pure noise
- [[sdxl-inpaint]] — same graph, but only repaints inside a mask
- [[diffusion-models]] — what's actually happening inside the KSampler
- [[latent-diffusion]] — why everything runs in compressed latent space
- [[u-net]] — SDXL's denoiser architecture
- [[scheduler]] — how the per-step noise removal is scheduled
- [[classifier-free-guidance]] — what the `cfg` knob is really doing
