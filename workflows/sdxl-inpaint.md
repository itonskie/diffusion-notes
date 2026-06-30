---
type: workflow
tags:
  - foundations
  - sdxl
  - workflow
  - inpaint
date: 2026-06-29
updated: 2026-06-29
---

# SDXL inpaint

## Purpose

Take an image you already have, paint over a region you want changed, and let SDXL regenerate just that region while keeping the rest of the image untouched. This is the workflow you reach for when the picture is 95% right and one thing is wrong — wrong sky, awkward hand, distracting object in the background, T-shirt logo you want to swap. Outpainting (extending a canvas beyond its edges) is the same idea with a mask that covers the new empty area.

Inpainting needs a *different* model from base SDXL. The inpainting model is trained with extra input channels that tell the [[u-net]] which pixels are fixed context and which it needs to invent — base SDXL has no idea what to do with a mask.

**Important wrinkle:** the SDXL inpainting model from diffusers is a **UNET-only** file (just the denoising network weights). It does NOT bundle the text encoder (CLIP) or the [[latent-diffusion|VAE]] like a full checkpoint does. So this workflow uses TWO loaders side by side: a `UNETLoader` for the inpaint UNET, and a `CheckpointLoaderSimple` for base SDXL — but we only borrow base SDXL's CLIP and VAE, not its UNET. The UNET output from the base loader is unconnected on purpose.

## How to load

1. Make sure ComfyUI is running: `python main.py` from `/Users/markoesterdal/Projects/ComfyUI`, then open `http://127.0.0.1:8188`.
2. Drag `sdxl-inpaint.json` from this folder onto the canvas (or click **Load** in the right-hand menu and pick the file).
3. In the **LoadImage** node, click **choose file to upload** and pick the image you want to edit.
4. Right-click the loaded image, choose **Open in MaskEditor**, paint the area you want regenerated, click **Save to node**.
5. Edit the positive prompt to describe what should appear *in the masked area*. Edit the negative prompt to push the model away from things you don't want.
6. Hit **Queue Prompt**.

## Node walkthrough

A [[latent-diffusion]] model never works on pixels directly. It works in a compressed numerical space called a **latent** — basically a small grid of feature vectors that a separate network can decode back into a real image. The encoder squeezes pixels in, the model edits the latent, the decoder pushes it back out as pixels. That round-trip explains most of what's happening here.

- **UNETLoader** — loads just the inpaint UNET (the denoising network — see [[u-net]]) from `models/diffusion_models/xl-inpaint-0.1/diffusion_pytorch_model.fp16.safetensors`. This is the file that knows what to do with mask-channel inputs.
- **CheckpointLoaderSimple** — loads base SDXL from `models/checkpoints/SDXL/sd_xl_base_1.0.safetensors`. A full checkpoint bundles three things: the U-Net, the **CLIP** text encoder (turns your prompt into vectors the U-Net can condition on), and the **VAE** (variational autoencoder — the encoder/decoder pair that moves between pixels and latents). We only use CLIP and VAE from this loader; the UNET output is left unconnected on purpose.
- **CLIPTextEncode (positive)** — runs your prompt through CLIP to get a conditioning vector. This is what tells the model what to draw in the masked region.
- **CLIPTextEncode (negative)** — same thing for the *negative* prompt. Used by [[classifier-free-guidance]] (CFG) — the model is run twice per step, once leaning toward the positive prompt and once away from the negative, and the two are blended. Lets you steer with "what I don't want" as well as "what I want."
- **LoadImage** — loads your source image *and* the mask you paint in the MaskEditor. One node, two outputs: the IMAGE pixels and the MASK (a black-and-white image where white = "regenerate this").
- **VAEEncodeForInpaint** — the inpaint-specific node. It does three things in one shot: encodes the source image into a latent using the VAE, encodes the mask down to latent resolution (since the model works at 1/8 the spatial size of the image), and zeroes out the masked region of the latent so the sampler has to fill it in from pure noise. The `grow_mask_by` widget expands the mask by N pixels at the edges — feathers the boundary so seams blend instead of cliffing.
- **KSampler** — the actual denoising loop. Starts from noise inside the masked area, runs the [[u-net]] for `steps` iterations, each step nudging the latent closer to something that matches the prompt. The **scheduler** (see [[scheduler]]) decides the noise schedule — how aggressively to denoise at each step. **CFG** is the classifier-free-guidance strength knob mentioned above. **denoise=1.0** because inpaint regenerates the masked region from scratch (lower values would preserve some of the original noise in the masked area, which is what you want for img2img but not here).
- **VAEDecode** — turns the final latent back into actual pixels. Uses the VAE output from the checkpoint loader.
- **SaveImage** — writes the result to `ComfyUI/output/` with the `ComfyUI` filename prefix. The result you'll see has the unmasked region copied straight from your source and the masked region replaced with what the model invented.

## Defaults explained

- **steps = 30** — SDXL converges well between 25 and 35 steps with a modern sampler. Fewer than 20 and the masked region looks under-resolved; more than 40 is mostly wasted compute on inpaint because the unmasked context is already fixed.
- **cfg = 7.5** — standard SDXL inpaint guidance. Higher (10–12) sharpens prompt adherence but introduces artifacts at the mask seam. Lower (5–6) blends more naturally but the model wanders from the prompt.
- **sampler = dpmpp_2m**, **scheduler = karras** — the workhorse SDXL combination. Good quality at 30 steps, no weird artifacts, deterministic given the seed. The [[reference/scheduler-comparison]] page covers when to swap this out.
- **denoise = 1.0** — inpaint regenerates the masked region completely. Lower values are only meaningful when you keep some of the original latent (img2img territory).
- **grow_mask_by = 6** — feathers the mask edge by 6 pixels so the seam between original and regenerated blends. Bump to 12–16 if you still see a hard boundary.

## Gotchas

- **The inpainting model is a different checkpoint from base SDXL.** It was trained with extra mask-channel inputs to the U-Net. If you accidentally point the loader at base SDXL, inpainting will still "run" but the masked region will come back looking like noise or a totally unrelated image — the model has no idea what to do with the mask signal.
- **Two loaders look weird at first.** Yes, two. The inpaint UNET is a UNET-only file (no CLIP, no VAE), so we have to source CLIP and VAE from base SDXL. The base loader's MODEL output is intentionally left dangling — only its CLIP and VAE outputs feed downstream. If you wire base SDXL's MODEL into the KSampler by mistake, you'll get base SDXL behavior with a randomly-noisy masked region, since base SDXL has no mask channels.
- **Filename gotcha if you re-downloaded.** The widget values reference `SDXL/sd_xl_base_1.0.safetensors` and `xl-inpaint-0.1/diffusion_pytorch_model.fp16.safetensors`. If ComfyUI-Manager saved those under different subfolders, click each loader's filename dropdown in the UI and pick the actual entries — then File → Save to persist.
- **The mask is painted in the LoadImage node itself, not in a separate node.** Right-click the loaded image thumbnail → **Open in MaskEditor** → paint → **Save to node**. White = regenerate, transparent = keep. Easy to forget on the first run and end up regenerating the whole image (which is just txt2img with extra steps).
- **Prompt the masked region, not the whole scene.** The CLIP encoder doesn't know which part is masked. If you prompt "a red sports car on a wet street" but you only masked the sky, you'll get sky that looks like it wants to *contain* a red sports car. Prompt what you want in the masked area specifically.
- **First run on Mac M4 Max with MPS will be slow** — model loading, VAE encode of a real image, and the denoising loop will all hit fresh code paths. Subsequent runs reuse the loaded model and run in a fraction of the time.

## Screenshot

_Screenshot pending — add after first successful run. Save the rendered image to `assets/` and link with `![[assets/sdxl-inpaint-result.png]]`._

## See Also

- [[foundations]] — Week 1 MOC
- [[sdxl-txt2img]] — companion workflow, image from prompt alone
- [[sdxl-img2img]] — companion workflow, transform an existing image without a mask
- [[latent-diffusion]] — why everything goes through the VAE round-trip
- [[u-net]] — the denoising network inside the checkpoint
- [[scheduler]] — what `dpmpp_2m` and `karras` actually do
- [[classifier-free-guidance]] — what the CFG knob is steering
- [[reference/scheduler-comparison]] — sampler/scheduler picks per model family
- [[reference/flux-vs-sdxl]] — when to reach for SDXL inpaint vs Flux Fill
