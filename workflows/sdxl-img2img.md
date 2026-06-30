---
type: workflow
tags:
  - foundations
  - sdxl
  - workflow
  - img2img
date: 2026-06-29
updated: 2026-06-29
---

# SDXL img2img

## Purpose

Same SDXL Base 1.0 model as [[sdxl-txt2img]], same prompt-encoding pipeline, but the starting point is *your* image instead of pure random noise. The model takes the image you hand it, scrambles it part-way back toward noise, then denoises from there — guided by your prompt. The result keeps the bones of the input (composition, rough shapes, lighting direction) but redresses it according to the prompt. Reach for this when you have a photo, sketch, or earlier generation you want to push in a new direction without losing what already works about it.

## How to load

1. Make sure ComfyUI is running (it should be at http://127.0.0.1:8188).
2. Drop your starting image into `ComfyUI/input/` (or use the **choose file to upload** button on the LoadImage node once the graph is on the canvas).
3. Drag `sdxl-img2img.json` onto the ComfyUI canvas — or click **Load** in the menu and pick the file.
4. In the LoadImage node, pick your image from the dropdown (the graph ships with the placeholder filename `example.png` — swap it).
5. Edit the positive prompt to describe what you want the output to *become*.
6. Hit **Queue Prompt**.

## Node walkthrough

The graph is a straight pipe from "your image" to "saved PNG." Each node hands off to the next.

- **Load Checkpoint (`CheckpointLoaderSimple`)** — loads `sd_xl_base_1.0.safetensors` and unpacks it into three outputs: `MODEL` (the [[u-net]] denoiser — the network that's been trained to remove a bit of noise at each step), `CLIP` (the text encoder that turns your prompt into a vector the model understands), and `VAE` (the **Variational Autoencoder** — a small network that compresses normal RGB images down into a smaller "latent" representation, and decompresses them back). SDXL is a [[latent-diffusion]] model, meaning all the heavy denoising work happens in that compressed latent space — 8× smaller per side than the pixel image, so it's faster.
- **CLIPTextEncode (positive)** — runs your prompt through the CLIP text encoder and produces a **conditioning** tensor: a numerical summary of "what we want the image to look like." This is the steering signal the sampler uses to nudge denoising in your direction.
- **CLIPTextEncode (negative)** — same thing, but for what you *don't* want. This isn't just "ignored words" — it's an actively-opposing steering signal. See [[classifier-free-guidance]] for why having a negative side matters.
- **LoadImage** — reads the file you picked from `ComfyUI/input/` and emits it as an `IMAGE` tensor (raw RGB pixels). The filename defaults to `example.png` so the graph loads cleanly before you swap in your own.
- **VAEEncode** — the new node compared to [[sdxl-txt2img]]. Takes your pixel image and runs the VAE's encoder side on it, producing a **latent** — the compressed representation the model actually thinks in. Without this step the sampler would have no "starting image" in the right space; the latent it expects looks nothing like raw RGB.
- **KSampler** — the engine. It takes your encoded image-latent and runs the denoising loop guided by the positive and negative conditioning. With `denoise=0.65` it first re-adds 65% of the noise budget back to your latent, then runs the back-half of the schedule to clean it up. The lower this number, the more your input survives. See [[diffusion-models]] for the noise-add-then-remove picture, [[scheduler]] for how the steps are spaced, and [[classifier-free-guidance]] for what CFG does.
- **VAEDecode** — runs the VAE in reverse: latent back into a normal RGB image you can look at.
- **SaveImage** — writes the result to `ComfyUI/output/` with filename prefix `ComfyUI`. Tag it however you like.

## Defaults explained

**denoise = 0.65** — this is the whole point of img2img. A denoise of 1.0 means "start from pure noise" — that's just [[sdxl-txt2img]] with a wasted latent input. A denoise of 0.0 means "do nothing, return the input untouched." Between those, the value sets how much of the original survives. Rough guide:

- **0.30–0.45** — subtle restyling, the original is clearly recognizable.
- **0.50–0.70** — the sweet spot for most "redress this image in a new style" use cases. Composition stays, surfaces and details get repainted. 0.65 is a sensible default.
- **0.75–0.90** — the prompt mostly takes over, the input becomes a loose sketch.

**steps = 25, sampler = euler, scheduler = normal** — same defaults as [[sdxl-txt2img]] and for the same reasons. Worth knowing: at `denoise=0.65` only ~16 of those 25 steps actually run (steps × denoise ≈ effective steps), because the sampler skips the early part of the schedule. So img2img is also a bit faster than txt2img with the same settings.

**cfg = 7.5** — **CFG = Classifier-Free Guidance scale.** This single number controls how hard the sampler pulls toward the positive prompt (and away from the negative one). Too low (~3) and the prompt barely matters; too high (~12+) and you get over-saturated, fried-looking output. 7.5 is the SDXL default and a fine starting point. Full intuition lives on [[classifier-free-guidance]].

**seed, control_after_generate = randomize** — different number every run, so you get a different result each time. Set `control_after_generate` to `fixed` if you want to keep the seed and only change the prompt or denoise value while you A/B test.

## Gotchas

- **Resolution matters.** SDXL was trained at ~1 megapixel (1024×1024 and a handful of other aspect ratios). The VAE will encode any size, but the model gets weird at sizes far from what it saw in training. If your input is much smaller than 1024 on the short side, upscale it first or expect mush. If it's far from a trained aspect ratio (long panoramic, ultra-tall portrait), expect duplicated subjects or stretched faces. The trained sizes to aim for: 1024×1024, 1152×896, 896×1152, 1216×832, 832×1216, 1344×768, 768×1344, 1536×640, 640×1536.
- **The input file has to be in `ComfyUI/input/`.** LoadImage shows a dropdown of files in that folder. If you don't see your image, drop it in there and refresh the browser, or use the **upload** button on the node directly.
- **`example.png` is a placeholder.** It ships with ComfyUI so the graph validates on load. You'll get *something* if you Queue Prompt without swapping it, but it won't be your image — pick your file from the dropdown first.
- **Denoise interacts with steps in a hidden way.** Effective denoising steps ≈ `steps × denoise`. At 25 steps and denoise 0.65 you only get ~16 real steps of cleanup. If you're cranking denoise way down (say 0.3) and results look noisy, bump `steps` to compensate — try 40.
- **CFG above ~10 burns img2img faster than txt2img.** Because you're already starting from a low-noise latent, the sampler has fewer steps to "soften" an aggressive guidance pull. Stay closer to 6–8.
- **MPS quirk on Mac.** First Queue Prompt with a fresh checkpoint will be slow (~30–60s) while the model warms up MPS kernels. Subsequent runs are normal.

## Screenshot

First run, 2026-06-30. Stock settings — JSON defaults unchanged: positive prompt `"oil painting of a red ceramic teapot on a wooden table, soft window light, painterly brush strokes, rich color"`, denoise 0.65, 25 steps euler/normal, CFG 7.5, seed `156680208700286`. Source image: the txt2img teapot result from this same session.

![[assets/sdxl-img2img-result.png]]

Confirms img2img is wired correctly: same teapot composition (preserved by the partial denoise), now in a painterly style (driven by the prompt change). The 0.65 denoise is the sweet spot — high enough that the brushwork lands, low enough that the composition stays put.

## See Also

- [[foundations]] — the Week 1 MOC
- [[sdxl-txt2img]] — same pipeline minus the LoadImage + VAEEncode pair
- [[sdxl-inpaint]] — img2img with a mask, so denoising only touches the masked region
- [[diffusion-models]] — what the back-and-forth between noise and image actually is
- [[latent-diffusion]] — why we're working in VAE-compressed space, not RGB
- [[scheduler]] — how the 25 (or 16 effective) steps are spaced through the noise schedule
- [[classifier-free-guidance]] — what CFG and the negative prompt are really doing
- [[u-net]] — the architecture inside the `MODEL` output that does the per-step denoising
