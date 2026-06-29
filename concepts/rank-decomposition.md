---
type: concept
tags:
  - foundations
  - math
  - lora
date: 2026-06-29
updated: 2026-06-29
status: developing
---

# Rank Decomposition

The mathematical foundation of [[lora|LoRA]]. Any matrix can be approximated by a product of two smaller matrices; how good the approximation is depends on the matrix's *intrinsic rank*.

## The math

A real matrix `W ∈ ℝ^(d×k)` has rank ≤ `min(d, k)`. The **singular value decomposition** factorizes it as:

```
W = U Σ Vᵀ
```

where `U` is `d×d` orthogonal, `V` is `k×k` orthogonal, and `Σ` is `d×k` with non-negative singular values `σ_1 ≥ σ_2 ≥ ... ≥ 0` on the diagonal.

The **rank** of `W` is the number of non-zero singular values. If most singular values are tiny, `W` is "approximately low-rank" — you can truncate to the top `r` singular values and lose little:

```
W ≈ W_r = U_r Σ_r V_rᵀ
```

where `U_r` keeps the first `r` columns of `U`, `Σ_r` keeps the top `r` singular values, `V_r` keeps the first `r` columns of `V`.

The Eckart-Young theorem says `W_r` is the **best** rank-`r` approximation in Frobenius norm.

## Why this matters for LoRA

LoRA writes `ΔW = BA` where `B ∈ ℝ^(d×r)` and `A ∈ ℝ^(r×k)`. The product `BA` always has rank ≤ `r`. So LoRA is constraining the update to be **at most rank `r`**.

If the "true" update from full fine-tuning happens to have low intrinsic rank, then a rank-`r` LoRA can capture it almost perfectly. If not, LoRA undertrains the task.

The [[papers/lora|LoRA paper]]'s central empirical claim is that task-adaptation updates **do** have low intrinsic rank — `r=1` to `r=8` is enough for most NLP tasks. For diffusion, `r=16-128` is typical because visual concepts have higher intrinsic dimensionality.

## Parameter count

- Full `ΔW`: `d × k` parameters
- LoRA `BA`: `d × r + r × k = r(d + k)` parameters

For a `4096 × 4096` matrix and `r = 16`:
- Full: 16,777,216 params
- LoRA: 131,072 params
- **Compression**: 128×

The savings grow with matrix size, which is why LoRA is most beneficial on the largest models.

## How to verify your `r` is enough

Two practical checks:

1. **Training loss plateau** — if loss bottoms out higher than full fine-tuning would have, your `r` is probably too small.
2. **Subspace overlap analysis** (Figure 3 in the LoRA paper) — train at two ranks, then compare the dominant singular vectors of the learned `ΔW`. If `r=4` and `r=64` agree on their top-1 direction (overlap > 0.5), the lower rank is capturing the essential update.

In practice for diffusion, you just train at a candidate `r`, evaluate sample quality, and bump `r` if the LoRA doesn't capture the concept well.

## When low-rank fails

LoRA's "low rank is enough" assumption can break:

- **Many disparate concepts in one LoRA** — multiple characters, styles, and objects compete for the same rank budget. Train separate LoRAs and combine at inference instead.
- **Very different from base** — adapting Flux to an anime style is much closer to the base distribution than adapting it to medical scans. The further from base, the higher rank needed (or just full fine-tune).
- **Long-tailed datasets** — rare features in the training set may not get enough rank budget.

In these cases, the answer is usually "train a higher-rank LoRA, or fine-tune fully."

## Why initialization matters

LoRA initializes `A ~ N(0, σ²)` and `B = 0`. So `BA = 0` at start of training, meaning the adapter is a no-op and the base model behavior is preserved.

If both `A` and `B` were initialized randomly, the model would start with random perturbations on top of the base weights — destroying useful behavior before training even begins. The asymmetric initialization is essential.

## See Also

- [[lora]]
- [[papers/lora]]
- [[fine-tuning]]
- [[foundations]]
