---
type: reference
tags:
  - foundations
  - resources
date: 2026-06-29
updated: 2026-06-29
---

# Hugging Face Diffusion Models Course

The canonical free starting point for learning diffusion models — written by the people who maintain the `diffusers` library, which is the standard Python toolkit for running and training these things. If you've ever wondered "where do most people *start* with diffusion," it's here.

This page is a map: what each unit covers, which notebooks matter, and which wiki pages have already absorbed each piece so you don't have to re-derive things you already wrote up. Used as the primary text for Week 1 of the sprint — see [[foundations]].

- **Course URL**: https://huggingface.co/learn/diffusion-course
- **Repo with notebooks**: https://github.com/huggingface/diffusion-models-class

## Unit structure

A quick orientation. "Ingested" in the right column means the key ideas from that unit already live as wiki pages in this vault.

| Unit | Focus | Status in this wiki |
|------|-------|---------------------|
| 1 | Introduction to diffusion models, training a diffusion model from scratch | Intro page ingested |
| 2 | Conditioning, guidance, fine-tuning | Intro page ingested — see [[fine-tuning]], [[guidance]], [[u-net]] (conditioning patterns) |
| 3 | Stable Diffusion | not ingested |
| 4 | Stable Diffusion deep-dive | not ingested |

## Unit 1 — what's in it

The intro page covers what diffusion models are and the high-level training + sampling loop — captured in [[diffusion-models]] and [[generative-models]]. Short version: a diffusion model learns to undo noise, step by step, so that you can start from pure noise and end at a picture.

Then two hands-on notebooks:

- **Introduction to Diffusers** — uses building blocks from the `diffusers` library to train and sample. Practical, less theory. You import pieces, wire them up, push the button.
  - Notebook: https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb
- **Diffusion Models from Scratch** *(optional but recommended)* — same flow built up from scratch in PyTorch, then compared to the `diffusers` version. Better for understanding the components, because you see the noise-adding loop, the U-Net, and the sampling loop as plain code rather than library calls.
  - Notebook: https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb

Notebooks open in Colab, Kaggle, Paperspace Gradient, or SageMaker Studio Lab — all free-tier cloud notebook services with GPU access.

## Unit 2 — what's in it

Focus: adapting pre-trained models and adding control. In other words, "you have a base model. Now make it do *your* thing, and steer it at generation time." Three main ideas, all captured in concept pages:

- **Fine-tuning** — continue training a pre-existing model on new data instead of training from scratch. Example: take a model that was trained on bedroom photos (LSUN Bedrooms), run 500 more training steps on a painting dataset (WikiArt), get coherent painterly bedrooms out. See [[fine-tuning]].
- **Guidance** — inference-time control. While the model is denoising, you steer the trajectory using any differentiable function on the noisy latent `x_t` (the partially-denoised image at step `t`). CFG steers toward your prompt; CLIP guidance steers toward a CLIP embedding; color guidance steers toward an average color. The trick is the same — apply a gradient at each denoising step. See [[guidance]] and [[classifier-free-guidance]].
- **Conditioning** — architectural patterns for *injecting* condition info (like a text prompt) into the model itself. Three common ones: extra input channels, project-the-condition-and-add-it-to-features, or cross-attention (the layer where image features look at text features). Captured in [[u-net]] under "Three ways to inject conditioning."

Quick decode of the three keywords if you're new to them: **fine-tuning** = teach the existing model something new. **Guidance** = nudge the model at inference time without retraining. **Conditioning** = the model's built-in way to receive instructions (the prompt) during training and inference.

Notebooks:
- **Fine-tuning and Guidance** — https://github.com/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb
- **Class-conditioned Diffusion Model Example** — https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb

Unit 2 additional resources: DDIM paper (Song et al. 2020), GLIDE paper (Nichol et al. 2021), eDiffi paper (Balaji et al. 2022) — all listed at the bottom of the unit's intro page. DDIM is the deterministic-sampler paper that made fewer-step generation possible; GLIDE was OpenAI's early text-to-image diffusion model; eDiffi is NVIDIA's "use different experts at different noise levels" paper.

## Additional resources (cited from Unit 1 intro)

| Resource | Notes |
|----------|-------|
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | In-depth walk-through with math and code. Goes alongside [[ddpm]]. |
| [Unconditional Image-Generation docs](https://huggingface.co/docs/diffusers/training/unconditional_training) | Official training script and dataset-prep examples |
| [AI Coffee Break — Diffusion Models](https://www.youtube.com/watch?v=344w5h24-h8) | Video explainer |
| [Yannic Kilcher — DDPM walkthrough](https://www.youtube.com/watch?v=W-O7AZNzbzQ) | Paper walkthrough for [[ddpm]] |
| [Unit 1 informal video](https://www.youtube.com/watch?v=09o5cv6u76c) | Author run-through of Unit 1 material |

"Unconditional" in that second link means the model just generates a random image with no prompt — useful for understanding the bare-bones training loop before you add text conditioning on top.

## See Also

- [[foundations]]
- [[diffusion-models]]
- [[generative-models]]
- [[ddpm]]
