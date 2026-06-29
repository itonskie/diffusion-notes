---
type: concept
tags:
  - foundations
  - math
  - lora
date: 2026-06-29
updated: 2026-06-29
---

# Rank Decomposition

The linear algebra story behind why [[lora|LoRA]] works. Short version: most big matrices don't actually need to be that big. They look big but they're secretly simple, and you can rewrite them as a product of two much smaller matrices that say almost the same thing. LoRA is just betting that the *change* you want to make to a fine-tuned model is one of those secretly-simple matrices.

This is the math-deeper companion to the [[lora|LoRA]] page. Same intuition, but with the linear algebra unpacked.

## The picture

A matrix is a grid of numbers. Multiplying a vector by a matrix is "apply a transformation" — rotate it, stretch it, project it. The **rank** of a matrix tells you how many *truly independent* directions of transformation it has.

A 1024×1024 matrix has at most rank 1024 — at most 1024 independent directions. But it can be much less than that. Imagine a 1024×1024 matrix where every column is just a copy of the first column. It still *looks* 1024×1024, but it only has 1 independent direction. Its rank is 1.

Real matrices rarely repeat that obviously — but they often have a small number of *dominant* directions and a long tail of tiny ones that you could throw away without losing much. That's what "approximately low-rank" means. The matrix is *acting* like a small matrix wearing a big-matrix costume.

LoRA's bet, restated: the *update* you apply during fine-tuning — the difference between the original weights and the fine-tuned weights — is approximately low-rank. So instead of storing it as a giant grid, you store it as the product of two skinny grids, which costs way less.

## SVD — the universal "find the directions" tool

Every real matrix can be factored as a product of three matrices in a specific way called the **singular value decomposition** (SVD). This is the canonical way to extract the dominant directions of a matrix.

### Intuition first

Given any matrix `W`, SVD tells you:
1. A list of "input directions" (orthogonal axes in input space).
2. A list of "output directions" (orthogonal axes in output space).
3. A list of **singular values** — one positive number per matched pair of directions, telling you how strongly that input-to-output direction is amplified by `W`.

The singular values come out sorted, biggest first. If only the first few are big and the rest are nearly zero, the matrix is *effectively* low-rank — almost all of its action lives in those first few directions.

### The math

```
W = U Σ Vᵀ
```

- `W` — the matrix you're decomposing. Shape `d × k` (d rows, k columns).
- `U` — `d × d` orthogonal matrix (columns are unit vectors pointing in mutually perpendicular directions). Each column is one "output direction."
- `V` — `k × k` orthogonal matrix. Each column is one "input direction." The `ᵀ` (transpose) flips rows and columns; `Vᵀ` means we're using `V`'s rows in the product.
- `Σ` — `d × k` matrix that is zero everywhere except the diagonal, which holds the **singular values** `σ_1 ≥ σ_2 ≥ ... ≥ 0` in decreasing order. These say how much each direction-pair gets amplified.

The **rank** of `W` equals the number of *non-zero* singular values. If `σ_17` and beyond are all zero, the matrix's rank is 16 — only 16 input directions actually map to anything non-trivial.

### Truncating: keep only what matters

The big move: if `σ_17` and beyond are very small (but not exactly zero), throw them away. Keep only the top `r` singular values and their matching directions:

```
W ≈ W_r = U_r Σ_r V_rᵀ
```

- `W_r` — the rank-`r` approximation of `W`.
- `U_r` — first `r` columns of `U` (the top `r` output directions).
- `Σ_r` — `r × r` diagonal matrix with the top `r` singular values.
- `V_r` — first `r` columns of `V` (the top `r` input directions).

The **Eckart–Young theorem** says this `W_r` is provably the **best** rank-`r` approximation of `W` in **Frobenius norm** (a way to measure "size of a matrix" — square every entry, sum them up, take the square root; basically Pythagoras on all the entries). There is no other rank-`r` matrix that is closer to `W`. That's why SVD is *the* tool for low-rank approximation.

## Why this matters for LoRA

LoRA writes the update as `ΔW = BA` where:
- `ΔW` — the change to the weight matrix from fine-tuning.
- `B` — a tall skinny matrix, shape `d × r`.
- `A` — a wide short matrix, shape `r × k`.

Multiply them: `BA` is shape `d × k`, same as the original weight. And because of how matrix multiplication works, the product of a `d × r` matrix and an `r × k` matrix can have rank at most `r`. So LoRA is *forcing* the update to be at most rank `r` — it physically can't be anything else.

If the "true" update from full fine-tuning happens to have low intrinsic rank, then a rank-`r` LoRA can capture it almost perfectly. If not — if the task genuinely needs a richer update — LoRA undertrains and produces a worse result than full fine-tuning would.

The [[papers/lora|LoRA paper]]'s central empirical claim: task-adaptation updates **do** have low intrinsic rank. For NLP, `r=1` to `r=8` is enough. For diffusion (visual concepts are richer — more independent ways an image can vary than a text task), `r=16–128` is typical.

## Parameter count — the savings

- Full `ΔW`: `d × k` parameters.
- LoRA `BA`: `d × r + r × k = r(d + k)` parameters.

For a 4096 × 4096 weight matrix and `r = 16`:
- Full: 16,777,216 params
- LoRA: 131,072 params
- **Compression: 128×**

The savings grow with matrix size, which is why LoRA pays off most on the largest models — exactly the ones you can't afford to fine-tune fully.

## How to verify your `r` is enough

Two practical checks:

1. **Training loss plateau.** If your training loss bottoms out noticeably higher than full fine-tuning would have, `r` is probably too small — the patch can't physically express what the model needs to learn.
2. **Subspace overlap analysis** (Figure 3 in the LoRA paper). Train at two different ranks (say `r=4` and `r=64`), then compare the *dominant directions* (top singular vectors) of the learned `ΔW`. If `r=4` and `r=64` agree on their top-1 direction (overlap > 0.5), then the smaller rank is capturing the essential update — anything more is decoration.

In practice for diffusion: just train at a candidate `r`, eyeball the sample quality, and bump `r` if the LoRA doesn't capture the concept well. Nobody runs SVD overlap analysis on a Friday afternoon.

## When low-rank fails

LoRA's "low rank is enough" assumption can break:

- **Many disparate concepts in one LoRA** — multiple characters, styles, and objects all competing for the same rank budget. They interfere. Better: train separate LoRAs and combine them at inference.
- **Very far from the base distribution** — adapting Flux to a new anime style is close to what Flux already knows. Adapting Flux to medical scans is not. The further you are from base, the more independent directions you need, the higher rank required (or just full fine-tune).
- **Long-tailed datasets** — rare features in the training set get crowded out of the rank budget by frequent ones.

In these cases, the answer is usually "train a higher-rank LoRA, or fine-tune fully."

## Why initialization matters

LoRA initializes `A ~ N(0, σ²)` (random small numbers from a normal distribution) and `B = 0` (all zeros). So `BA = 0` at the start of training — the adapter is a no-op and the base model behaves exactly as it did before.

If both `A` and `B` were initialized randomly, then `BA` at step zero would be random noise getting added to the carefully-trained base weights — destroying useful behavior before training even begins. The asymmetric initialization (one random, one zero) is essential. The model has to *earn* its non-identity behavior, like adaLN-Zero in [[dit|DiT]] does for transformer blocks.

## See Also

- [[lora]]
- [[papers/lora]]
- [[fine-tuning]]
- [[foundations]]
