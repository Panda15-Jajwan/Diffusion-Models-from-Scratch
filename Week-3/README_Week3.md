# Week 3: The Forward Diffusion Process & Noise Scheduler

Diffusion Models from Scratch — Seasons of Code 2026

---

## Deliverables Checklist

- [x] `NoiseScheduler` class with linear and cosine schedule support
- [x] Closed-form `q(x_t | x_0)` sampling using the reparameterization trick
- [x] Visualization: one image noised at `t = 0, 100, 250, 500, 750, 999`
- [x] Unit tests verifying `x_T ≈ N(0, I)` (pure Gaussian noise)
- [x] Plot comparing linear vs cosine SNR curves
- [x] Code pushed to GitHub with updated README

---

## Core Concept

The forward process adds noise over T=1000 steps. Instead of stepping one-by-one, the **reparameterization trick** lets us jump to any timestep t directly:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

At `t = T`: `alpha_bar_T ≈ 0`, so `x_T ≈ N(0, I)` — pure Gaussian noise.

---

## NoiseScheduler API

```python
scheduler = NoiseScheduler(T=1000, schedule='linear')   # or 'cosine'
scheduler = NoiseScheduler(T=1000, schedule='cosine')

x_t, noise = scheduler.q_sample(x0, t)
snr_curve   = scheduler.snr()
```

---

## Linear vs Cosine Schedule

| Property | Linear | Cosine |
|---|---|---|
| `beta` range | `[1e-4, 0.02]`, uniform | Derived from cosine curve |
| Image destroyed at | `t ≈ 500` | `t ≈ 800` |
| SNR decay | Fast in early steps | Gradual throughout |
| Training signal | Wasted steps near end | Meaningful throughout |

The cosine schedule (Nichol & Dhariwal, 2021) fixes the linear schedule's tendency to destroy structure too early.

---

## Unit Tests

Four statistical checks on `x_T` (t=999):
1. `mean(x_T) ≈ 0`
2. `std(x_T) ≈ 1`
3. `t=0` barely adds noise (alpha_bar[0] ≈ 1)
4. `x_T` is uncorrelated with `x_0`

---

## Connection to Week 4

The `NoiseScheduler.q_sample()` is used directly during training: for each batch, sample a random `t`, corrupt the image, and train the UNet to predict the added noise `epsilon`. The reparameterization trick means we never need to iterate through all T steps during training.

---

## How to Run

1. Upload `Week3_NoiseScheduler.ipynb` to Google Colab
2. Runtime → Change runtime type → T4 GPU (CPU works fine too — no training here)
3. Runtime → Run all
