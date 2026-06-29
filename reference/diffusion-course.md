---
type: reference
tags:
  - foundations
  - resources
date: 2026-06-29
updated: 2026-06-29
---

# Hugging Face Diffusion Models Course

The canonical free starting point for learning diffusion models. Used as the primary text for Week 1 of the sprint — see [[foundations]].

- **Course URL**: https://huggingface.co/learn/diffusion-course
- **Repo with notebooks**: https://github.com/huggingface/diffusion-models-class

## Unit structure

| Unit | Focus | Status in this wiki |
|------|-------|---------------------|
| 1 | Introduction to diffusion models, training a diffusion model from scratch | Intro page ingested |
| 2 | Conditioning, guidance, fine-tuning | Intro page ingested — see [[fine-tuning]], [[guidance]], [[u-net]] (conditioning patterns) |
| 3 | Stable Diffusion | not ingested |
| 4 | Stable Diffusion deep-dive | not ingested |

## Unit 1 — what's in it

The intro page covers what diffusion models are and the high-level training + sampling loop — captured in [[diffusion-models]] and [[generative-models]].

Then two hands-on notebooks:

- **Introduction to Diffusers** — uses building blocks from the `diffusers` library to train and sample. Practical, less theory.
  - Notebook: https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb
- **Diffusion Models from Scratch** *(optional but recommended)* — same flow built up from scratch in PyTorch, then compared to the `diffusers` version. Better for understanding the components.
  - Notebook: https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb

Notebooks open in Colab, Kaggle, Paperspace Gradient, or SageMaker Studio Lab.

## Unit 2 — what's in it

Focus: adapting pre-trained models and adding control. Three main ideas, all captured in concept pages:

- **Fine-tuning** — continue training a pre-existing model on new data. Example: LSUN Bedrooms model → 500 steps on WikiArt → coherent painterly bedrooms. See [[fine-tuning]].
- **Guidance** — inference-time control via any differentiable function on `x_t`. CFG, CLIP guidance, color guidance, etc. See [[guidance]] and [[classifier-free-guidance]].
- **Conditioning** — three architectural patterns for injecting condition info into the U-Net (extra channels, project-and-add, cross-attention). Captured in [[u-net]] under "Three ways to inject conditioning."

Notebooks:
- **Fine-tuning and Guidance** — https://github.com/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb
- **Class-conditioned Diffusion Model Example** — https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb

Unit 2 additional resources: DDIM paper (Song et al. 2020), GLIDE paper (Nichol et al. 2021), eDiffi paper (Balaji et al. 2022) — all listed at the bottom of the unit's intro page.

## Additional resources (cited from Unit 1 intro)

| Resource | Notes |
|----------|-------|
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | In-depth walk-through with math and code. Goes alongside [[ddpm]]. |
| [Unconditional Image-Generation docs](https://huggingface.co/docs/diffusers/training/unconditional_training) | Official training script and dataset-prep examples |
| [AI Coffee Break — Diffusion Models](https://www.youtube.com/watch?v=344w5h24-h8) | Video explainer |
| [Yannic Kilcher — DDPM walkthrough](https://www.youtube.com/watch?v=W-O7AZNzbzQ) | Paper walkthrough for [[ddpm]] |
| [Unit 1 informal video](https://www.youtube.com/watch?v=09o5cv6u76c) | Author run-through of Unit 1 material |

## See Also

- [[foundations]]
- [[diffusion-models]]
- [[generative-models]]
- [[ddpm]]
