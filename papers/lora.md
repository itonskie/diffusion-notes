---
type: paper
tags:
  - foundations
  - lora
  - fine-tuning
  - paper
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# LoRA — Low-Rank Adaptation of Large Language Models

> Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
> https://arxiv.org/abs/2106.09685 • [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2106.09685)

## Status

**Developing** — primary content from ar5iv HTML render. Originally an NLP paper (GPT-3, RoBERTa) but the technique has become the dominant approach for fine-tuning diffusion models too. Empirical results below are NLP; diffusion-specific results are in [[lora]] (the concept page).

## Abstract (verbatim)

> "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."

## The core technique

For a pre-trained weight matrix `W₀ ∈ ℝ^(d×k)`, the adapted weight is:

```
W = W₀ + ΔW = W₀ + BA
```

with:
- `B ∈ ℝ^(d×r)` and `A ∈ ℝ^(r×k)`, both trainable
- `W₀` is frozen
- `r ≪ min(d, k)` is the **rank** (typically 1-64 for NLP; 8-128 for diffusion)

The forward pass becomes:

```
h = W₀·x + (α/r) · B·A·x
```

where `α` is a scaling constant. Setting `α = r` makes the update unit-scale; doubling `α/r` doubles the LoRA's effect at inference (this is what ComfyUI's "LoRA strength" slider modifies).

Initialization: `A` random Gaussian, `B = 0`. This means `ΔW = 0` at the start of training — the adapter is a no-op until it learns.

## Why it works: low intrinsic rank

The paper's central claim: weight updates during task adaptation have **low intrinsic rank**, so the rank-`r` factor `BA` is a sufficient approximation of the true `ΔW`.

This is supported by their rank ablation — see [[rank-decomposition]] for the SVD intuition.

## The rank ablation (Table 6 on GPT-3, WikiSQL)

| Rank `r` | Adapts | Score |
|----------|--------|-------|
| 1 | Wq only | 68.8 |
| 2 | Wq only | 69.6 |
| 4 | Wq only | 70.5 |
| 8 | Wq only | 70.4 |
| 64 | Wq only | 70.0 |
| 1 | Wq + Wv | 73.4 |
| 2 | Wq + Wv | 73.3 |
| 4 | Wq + Wv | 73.7 |
| 8 | Wq + Wv | 73.8 |
| 64 | Wq + Wv | 73.5 |

**Striking result**: r=1 is competitive with r=64 when adapting both Q and V. r=8 is essentially the ceiling. This says the *intrinsic dimensionality of the task-adaptation update is tiny* — single digits.

## Which matrices to adapt (Table 5)

For an 18M parameter budget on GPT-3:

| Adapt | WikiSQL | MNLI |
|-------|---------|------|
| Wq only (r=8) | 70.4 | 91.0 |
| Wv only (r=8) | 73.0 | 91.0 |
| Wq + Wv (r=4) | **73.7** | 91.3 |
| All four QKVO (r=2) | 73.7 | **91.7** |

**Conclusion**: Q+V is the canonical default. Adding K and O barely helps. FFN matrices are frozen for simplicity — the paper does not LoRA them.

For **diffusion**, the convention diverged: most diffusion LoRA tools adapt many more layers (including FFN and sometimes conv layers in [[u-net]]) because diffusion fine-tuning needs to capture richer visual concepts than NLP task adaptation. See [[lora]] for diffusion-specific layer selection.

## Parameter and storage savings

| Approach | Trainable params on GPT-3 175B | Checkpoint size |
|----------|-------------------------------|-----------------|
| Full fine-tuning | 175,000,000,000 | 350 GB (FP16) |
| LoRA (r=4, Q+V) | 4,700,000 | 35 MB |

**~10,000× fewer trainable params, ~10,000× smaller checkpoint.** Optimizer state savings (Adam stores momentum + variance, doubling memory) drop the GPU memory requirement from ~1.2 TB to ~350 GB.

For diffusion, a Flux LoRA at rank 16 is typically 70-300 MB vs Flux base at 12 GB — same order-of-magnitude wins.

## No inference-time cost

At deployment, you can merge the LoRA into the base:

```
W_merged = W₀ + (α/r) · BA
```

Now `W_merged` is just a regular weight matrix; inference uses it directly with no latency overhead vs a fully fine-tuned model. **This is the key advantage over adapter-style methods**, which add layers and incur 20-30% per-token latency.

To swap LoRAs at inference, subtract one and add another:

```
W₀ ← W_merged - (α/r) · BA          # restore base
W_new_merged ← W₀ + (α/r) · B'A'    # apply new LoRA
```

Fast and memory-light.

## Hyperparameters they recommended

| Param | NLP value |
|-------|-----------|
| Learning rate | 1e-4 to 5e-4 (depends on base model) |
| Rank `r` | 4-8 |
| α | First-attempted r-equivalent (set once, then leave fixed) |
| Dropout | Typically 0.0-0.1 |

For diffusion the values shift — typically lr 1e-4, r 16-128, α=r (so `α/r = 1`). See [[lora]] and AI-Toolkit configs in `configs/`.

## What ΔW actually learns (Section 7)

The paper analyzes the learned `ΔW` and finds it **amplifies features not emphasized in `W₀`**. For r=4 on Wq, the amplification factor is ~21× — meaning LoRA learns directions roughly orthogonal to the dominant directions of the pre-trained weight. This is intuitive: there's no point re-learning what's already there; the task-specific update should be a *delta*.

Wq has higher intrinsic rank than Wv (Section 7.3) — implying Q transformations are inherently richer/more task-dependent than V's, which explains why adapting Q alone benefits more from larger r than adapting V alone.

## Why this matters for the sprint

LoRA is the default fine-tuning method for image and video diffusion models. AI-Toolkit, Kohya, Musubi Tuner — all train LoRAs. The concepts above transfer directly:
- Pick which layers to adapt (often "all attention" or "all linear" in diffusion)
- Pick `r` (16-64 for character/style; 128+ for complex concepts)
- Set `α = r` for unit scaling
- Save the result as a `.safetensors` file ~100 MB instead of redistributing the full base model

See [[lora]] for the diffusion-specific application.

## See Also

- [[lora]]
- [[rank-decomposition]]
- [[fine-tuning]]
- [[diffusion-models]]
- [[u-net]]
- [[dit]]
- [[foundations]]
