---
type: reference
tags:
  - foundations
  - comfyui
  - tools
date: 2026-07-01
updated: 2026-07-01
---

# ComfyUI

The tool you're spending Day 3 onward inside of. A local-first, node-based interface for running diffusion image and video models.

## The one-paragraph picture

ComfyUI is a desktop app that gives you a **graph editor** for diffusion model pipelines — like Blender's shader nodes, or n8n's workflow canvas. Every image generation is a graph: a checkpoint loader plugs into a text encoder plugs into a sampler plugs into a decoder plugs into a save node. You drag boxes onto a canvas, wire them up, hit Run, and the engine executes the graph and produces an image. Everything else — img2img, inpainting, ControlNet, LoRAs, video, animation, batch processing, custom samplers — is the same graph with extra nodes inserted or substituted in the right place.

It runs as a **local Python web server** (default port 8188) that you talk to via a browser. The server holds the models in GPU memory; the browser is just the editor UI. Nothing is sent to the cloud. Your models, your machine, your privacy.

## Why nodes instead of a form?

The major alternative — **AUTOMATIC1111 / A1111 / Stable Diffusion WebUI** — uses a traditional form UI: pick a model from a dropdown, type a prompt in a textbox, slide some sliders, click Generate. Easier for casual one-off generations.

ComfyUI bets on nodes because diffusion *is* a graph. The math is "load weights → encode prompt → noise → denoise loop → decode latent → save," and complex techniques (multi-stage refinement, regional prompting, ControlNet stacks, LoRA blending) work by extending or reshuffling that graph. A form UI hides the structure; a node UI exposes it. That's worse for someone who just wants a cat picture and better for someone who wants to understand or compose workflows.

That's why every serious diffusion shop — pro illustrators, animation studios, training pipelines, research labs — uses ComfyUI now. The investment in learning the node graph pays back when you start building your own workflows instead of running other people's.

## What's actually on your machine

After yesterday's install, you have:

```
~/Projects/ComfyUI/
├── main.py                    # the server entrypoint — `python main.py` launches it
├── .venv/                     # uv-managed Python 3.12 venv with PyTorch nightly + ComfyUI deps
├── comfy/                     # the ComfyUI engine — node classes, sampler implementations
├── custom_nodes/              # third-party node packs (ComfyUI-Manager lives here)
│   └── ComfyUI-Manager/       # installs more custom nodes and models from the UI
├── models/                    # everything ComfyUI loads at runtime
│   ├── checkpoints/           # full diffusion checkpoints (SDXL Base, SD 1.5, Flux, etc.)
│   ├── diffusion_models/      # UNET-only model weights (e.g. the diffusers SDXL inpaint)
│   ├── loras/                 # LoRA adapter files (you'll fill this in Week 2)
│   ├── vae/                   # standalone VAEs
│   ├── controlnet/            # ControlNet adapters
│   ├── clip/                  # standalone text encoders
│   └── vae_approx/            # tiny VAEs for live preview during sampling (TAESDXL)
├── input/                     # source images you drop here become selectable in LoadImage
├── output/                    # generated images land here
└── user/                      # your saved workflows, settings, comfyui-manager config
```

**The server logs everything to the terminal where you started `main.py`.** Read those logs — they tell you what loaded, what's executing, and what failed. Most "why isn't this working" questions are answered by the log.

## What it is NOT

- **Not a training tool.** ComfyUI generates images from already-trained models. To train a model or a LoRA, you use AI-Toolkit, Kohya, OneTrainer, or Musubi Tuner. You'll touch AI-Toolkit on Day 4.
- **Not a fine-tuning UI** in the model-weights sense. It can apply LoRAs at inference time (combining a base model with a trained adapter), but it doesn't update weights itself.
- **Not the only diffusion frontend.** A1111 (form UI), Forge (A1111 fork, faster), InvokeAI (more polished UX, less flexible) all exist. ComfyUI is the one this sprint targets because real production work happens here.
- **Not deterministic across versions.** A custom node pack update or a ComfyUI version bump can change graph behavior. Pin versions when reproducibility matters.

## Where it sits in the diffusion ecosystem

Picture three layers:

| Layer | What lives here | Examples |
|---|---|---|
| **Model weights** | The actual neural network files | SDXL Base, Flux.1-dev, SD 1.5 |
| **Inference engine** | Code that loads weights and runs them | PyTorch, Diffusers library, ComfyUI's engine |
| **Frontend / orchestrator** | UI that lets you compose and run pipelines | **ComfyUI**, A1111, InvokeAI |

ComfyUI sits at the top. It uses PyTorch under the hood (the same PyTorch you'd use from a Python script) and loads model files (the same `.safetensors` you'd download anywhere else). The value-add is the **graph editor + execution engine + thousands of community-built nodes**.

Production teams often build a ComfyUI workflow during exploration, then export it to the **API workflow format** (different from the UI workflow format you've been touching) and hit it programmatically from a server. So learning ComfyUI is also learning a deployable runtime, not just a desktop app.

## Vocabulary you'll keep hearing

- **Workflow** — a graph saved as JSON. Two JSON formats exist: **UI format** (full visual layout, what you save from the menu) and **API format** (stripped-down graph topology, what you POST to `/prompt`). The pre-built workflows in `diffusion-notes/workflows/` are UI format.
- **Node** — a single box on the canvas. Has inputs (left side), outputs (right side), widgets (settable fields inside), and a node type (the underlying Python class).
- **Custom node** — a node type added by a third-party package (installed into `custom_nodes/`). ComfyUI-Manager is itself a custom node.
- **Queue Prompt / Run** — submit the current graph for execution. Multiple submissions queue up and run in order.
- **Checkpoint** — a full model file containing U-Net + CLIP + VAE bundled together. Loaded with `CheckpointLoaderSimple`.
- **Latent** — see [[latent-diffusion]]. The compressed representation diffusion actually operates on.
- **Conditioning** — the output of a text encoder. What the sampler "leans toward" or "away from."
- **Sampler / Scheduler** — see [[scheduler]]. The two-part recipe for how denoising steps are taken.

## Useful keyboard / mouse

- **Double-click empty canvas** → search box for adding any node by name
- **Right-click empty canvas** → menu for adding nodes by category
- **Right-click a node** → node-specific actions (e.g. "Open in MaskEditor" on LoadImage)
- **Drag from output dot to input dot** → wire two nodes
- **Click a wire, press Delete** → remove it
- **Hold Ctrl (Cmd on Mac) and drag** → box-select multiple nodes
- **Ctrl/Cmd + Shift + drag selected nodes** → move them as a group
- **Ctrl/Cmd + B** → toggle the selected node's "bypass" mode (skip it during execution)
- **Ctrl/Cmd + S** → Save Workflow (UI format JSON)

## See Also

- [[foundations]] — Week 1 MOC
- [[workflows/sdxl-txt2img]] — the reference pre-built graph
- [[workflows/sdxl-img2img]] — same graph + image input
- [[workflows/sdxl-inpaint]] — same graph + image + mask + dual-loader
- [[experiments/2026-07-01-comfyui-lab1-txt2img-from-scratch]] — Lab 1, builds the txt2img graph node by node
- [[diffusion-models]] — what's actually running inside KSampler
- [[reference/flux-vs-sdxl]] — picking a model to feed into ComfyUI
- [[reference/scheduler-comparison]] — sampler/scheduler choices for KSampler
