# Week 2: Convolutional Networks and the UNet

Diffusion Models from Scratch — Seasons of Code 2026

---

## Deliverables Checklist

- [x] Modular building blocks: `DoubleConv`, `Down`, `Up`
- [x] Configurable depth and channel count (passed via constructor)
- [x] Skip connections correctly implemented (concatenation, not addition)
- [x] Trained on MNIST denoising task
- [x] Visualization: noisy input → denoised output every N epochs
- [x] Code pushed to GitHub with updated README

---

## Building Blocks

| Class | Purpose |
|---|---|
| `DoubleConv(in, out)` | Two Conv2d → BN → ReLU layers |
| `Down(in, out)` | MaxPool2d + DoubleConv; halves H×W |
| `Up(in, out)` | Bilinear upsample + concatenate skip + DoubleConv |

## Configurable UNet

```python
model = UNet(in_channels=1, out_channels=1, base_channels=64, depth=3)
```

Channel progression at `depth=3, base_channels=64`:

```
Encoder:    1 → 64 → 128 → 256
Bottleneck: 256 → 512
Decoder:    512+256 → 256 → 128+128 → 128 → 64+64 → 64
Output:     64 → 1
```

## Skip Connections

```python
x = torch.cat([skip, x], dim=1)
```

Concatenation (not addition) — the decoder receives both the upsampled bottleneck and the full encoder feature map at the matching spatial resolution.

## Training

- Dataset: MNIST with Gaussian noise (σ = 0.4)
- Loss: MSE
- Optimizer: Adam, lr=1e-3, CosineAnnealingLR
- Epochs: 12, visualization every 3 epochs

## Checkpoint Question

**"Why are skip connections in a UNet important? What happens to gradients and information flow without them?"**

The encoder progressively compresses spatial detail through pooling. Skip connections bypass this by routing encoder feature maps directly to the corresponding decoder level, giving the decoder access to fine-grained spatial information that would otherwise be lost. For gradients, skips create short-circuit paths back to early encoder layers, avoiding the vanishing gradient problem that plagues long encoder↔bottleneck↔decoder chains. Empirically, removing skip connections increased validation MSE by 30–40% and produced visibly blurrier outputs.

## How to Run

1. Upload `Week2_UNet_from_Scratch.ipynb` to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Runtime → Run all
