---
type: paper
tags:
  - foundations
  - lora
  - fine-tuning
  - paper
date: 2026-06-29
updated: 2026-06-29
status: summary
---

# LoRA — Low-Rank Adaptation of Large Language Models

> Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
> https://arxiv.org/abs/2106.09685 • [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2106.09685)

## Status

**Developing** — primary content from ar5iv HTML render. Originally an NLP paper (GPT-3, RoBERTa) but the technique has become the dominant approach for fine-tuning diffusion models too. Empirical results below are NLP; diffusion-specific results are in [[lora]] (the concept page).

## What this paper actually claims

GPT-3 has 175 billion weights. If you fine-tune it the obvious way — nudging every weight on your task data — you end up with another 175-billion-weight model to store, serve, and swap. Do that for ten downstream tasks and you're shipping 3.5 TB of model files.

LoRA's bet: the *change* you want from fine-tuning is much simpler than the base model itself. So instead of learning a full-size update, learn it as the product of two skinny matrices. Freeze the base. Train the two skinnies. The patch you ship is ~10,000× smaller than the base, and there's no extra cost when you run inference because you can fold the patch back into the base weights.

That's the whole paper. The rest is "we tried this, it works, and here's how small `r` can go before it stops working."

> **Domain decode first.** A few words you need: **fine-tuning** = continuing to train a pre-trained model on a narrower task; **weight matrix** = the big grid of numbers inside a layer that multiplies the layer's input; **rank** of a matrix = how many genuinely-independent directions of behavior it has (a 1024×1024 matrix can have rank as low as 1 if its columns all look alike); **adapter** = an extra small module bolted onto a frozen base model to make it do a new task.

## Abstract (verbatim)

> "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."

## The core idea, in pictures

Picture one weight matrix `W₀` inside the model — say it's 1024×1024, about a million numbers. Full fine-tuning would learn a same-shape update `ΔW` (also a million numbers) and use `W₀ + ΔW` going forward.

LoRA says: don't learn `ΔW` directly. Learn it as a product of two skinny rectangles:

- `B` — tall and thin (1024 rows, `r` columns)
- `A` — short and wide (`r` rows, 1024 columns)

Multiply them and you get something the right shape (1024×1024) to patch back onto `W₀`. But the *thing you trained* was just `B` and `A`, which is `2 · 1024 · r` numbers. Pick `r = 4` and you went from training a million numbers to training about 8,000. Same shape on the outside; a tiny fraction of the parameters on the inside.

At training time `B` starts at zero, so the patch contributes nothing on step one — the model behaves exactly like the base, and gradients have to *discover* what `B` should be. At inference time you can either (a) keep `A` and `B` as separate matrices and apply them as a runtime patch, or (b) compute `B·A` once, add it to `W₀`, and now you have a regular weight matrix with zero overhead.

## Why it works: low intrinsic rank

The claim, in one line: the update fine-tuning really wants is *intrinsically low-rank* — it only varies along a few independent directions, not a thousand.

The intuition: pre-training already taught the model most of what it needs. Fine-tuning is supposed to add a few new tricks for a specific task — it isn't supposed to rebuild the model. So the delta should be small and simple, in the linear-algebra sense.

For the deeper "why this is mathematically reasonable," see [[rank-decomposition]] — that page has the SVD picture.

## Math sketch

For a frozen pre-trained weight matrix `W₀ ∈ ℝ^(d×k)`, the adapted weight is:

```
W = W₀ + ΔW = W₀ + BA
```

Symbols decoded:
- `W₀` — the original weight matrix, frozen during LoRA training
- `d, k` — the matrix's two dimensions (rows, columns)
- `B` — a trainable tall-skinny matrix of shape `(d × r)`
- `A` — a trainable short-wide matrix of shape `(r × k)`
- `r` — the **rank**, the bottleneck width. Typically 1-64 for NLP, 8-128 for diffusion. The whole point of the technique is that this is small.
- `r ≪ min(d, k)` reads "r is much less than the smaller of d and k"

The forward pass through this layer becomes:

```
h = W₀·x + (α/r) · B·A·x
```

- `x` — the layer's input
- `h` — the layer's output
- `α` — a scaling constant. Setting `α = r` makes `α/r = 1`, the natural unit scale. Doubling `α/r` doubles how strongly the patch contributes at inference — this is what ComfyUI's "LoRA strength" slider modifies.

Initialization matters: `A` is random Gaussian, `B` is zero. That means `BA = 0` on step one — so `ΔW = 0` and the model is exactly the base until gradients teach `B` to do something useful.

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

**Striking result:** r=1 is competitive with r=64 when adapting both `Wq` and `Wv`. r=8 is essentially the ceiling. The "intrinsic dimensionality" of the task-adaptation update is *tiny* — single digits.

> Decoding the table column: `Wq` is the query weight matrix, `Wv` is the value weight matrix, both inside the attention mechanism of a transformer block. Attention is where the model decides which earlier tokens to focus on; Q, K, V are the three projections that drive that decision. The paper only LoRA-patches Q and V; the choice of *which matrices* gets a separate ablation below.

## Which matrices to adapt (Table 5)

For an 18M parameter budget on GPT-3:

| Adapt | WikiSQL | MNLI |
|-------|---------|------|
| Wq only (r=8) | 70.4 | 91.0 |
| Wv only (r=8) | 73.0 | 91.0 |
| Wq + Wv (r=4) | **73.7** | 91.3 |
| All four QKVO (r=2) | 73.7 | **91.7** |

**Conclusion:** Q+V is the canonical default. Adding K and O barely helps. The feed-forward matrices are left frozen for simplicity — the paper does not LoRA them.

For **diffusion**, the convention diverged: most diffusion LoRA tools adapt many more layers (including FFN and sometimes the conv layers inside [[u-net]]) because diffusion fine-tuning needs to capture richer visual concepts than NLP task adaptation. See [[lora]] for diffusion-specific layer selection.

## Parameter and storage savings

| Approach | Trainable params on GPT-3 175B | Checkpoint size |
|----------|-------------------------------|-----------------|
| Full fine-tuning | 175,000,000,000 | 350 GB (FP16) |
| LoRA (r=4, Q+V) | 4,700,000 | 35 MB |

**~10,000× fewer trainable params, ~10,000× smaller checkpoint.** And the optimizer-state savings are bigger than just the parameter savings — **Adam** (the standard optimizer, which stores running momentum and variance for every parameter) needs roughly 2-3× the parameter count in extra memory. That drops the GPU memory requirement from ~1.2 TB to ~350 GB.

For diffusion, a Flux LoRA at rank 16 is typically 70-300 MB vs Flux base at 12 GB — same order-of-magnitude wins.

## No inference-time cost

At deployment, you can merge the LoRA into the base:

```
W_merged = W₀ + (α/r) · BA
```

Now `W_merged` is just a regular weight matrix; inference uses it directly with no latency overhead vs a fully fine-tuned model. **This is the key advantage over adapter-style methods**, which insert extra layers into the network and pay 20-30% per-token latency forever.

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

## What `ΔW` actually learns (Section 7)

This part is the most surprising. The paper opens up the learned `ΔW` and asks: is it just re-learning what `W₀` already knows, or is it adding genuinely new directions?

Answer: it **amplifies features that `W₀` did not emphasize**. For r=4 on `Wq`, the amplification factor is ~21× — meaning LoRA learns directions roughly orthogonal to the dominant directions of the pre-trained weight. Which is intuitive once you say it out loud: there's no point re-learning what's already there; the task-specific update should be a *delta*, not a duplicate.

A related finding (Section 7.3): `Wq` has higher intrinsic rank than `Wv`. That implies Q transformations are inherently richer / more task-dependent than V's, which explains why adapting Q alone benefits more from larger `r` than adapting V alone does.

## What this paper changed downstream

LoRA is now the default fine-tuning method for image and video diffusion models. AI-Toolkit, Kohya, Musubi Tuner — all train LoRAs. The concepts above transfer directly:

- Pick which layers to adapt (often "all attention" or "all linear" in diffusion)
- Pick `r` (16-64 for character/style; 128+ for complex concepts)
- Set `α = r` for unit scaling
- Save the result as a `.safetensors` file (~100 MB) instead of redistributing the full base model

See [[lora]] for the diffusion-specific application.

## See Also

- [[lora]]
- [[rank-decomposition]]
- [[fine-tuning]]
- [[diffusion-models]]
- [[u-net]]
- [[dit]]
- [[foundations]]
