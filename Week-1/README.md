# Week 1 Deliverable — MNIST Feedforward Classifier

A feedforward neural network built from scratch in PyTorch that classifies
handwritten digits (MNIST) with **≥ 97 % test accuracy**.

---

## Architecture & Design Choices

### Model: `FeedForwardNet`
| Layer | Details |
|-------|---------|
| Input | 784 (flattened 28 × 28 image) |
| Hidden 1 | Linear(784 → 512) → BatchNorm → ReLU → Dropout(0.3) |
| Hidden 2 | Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.3) |
| Hidden 3 | Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.3) |
| Output | Linear(128 → 10) |

**Why these choices?**

- **Three hidden layers** give enough capacity for MNIST without over-fitting.
- **BatchNorm** stabilises training and speeds convergence; it sits *before*
  the activation so the ReLU sees normalised inputs.
- **Dropout (p = 0.3)** acts as a strong regulariser — prevents co-adaptation
  of neurons and reduces over-fitting.
- **No softmax in the model** — `CrossEntropyLoss` already applies
  log-softmax internally, so adding it separately would double the operation.

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimiser | Adam (lr = 1e-3) |
| Loss | CrossEntropyLoss |
| Epochs | 15 |
| Batch size | 128 |
| LR scheduler | ReduceLROnPlateau (patience = 2, factor = 0.5) |
| Train / Val split | 90 % / 10 % of the 60 k training set |

**Why Adam over SGD?**  
Adam adapts the learning rate per-parameter and converges much faster on
MNIST. SGD with momentum can match it eventually, but requires careful LR
tuning. For a one-week deliverable Adam is the pragmatic choice.

**ReduceLROnPlateau** halves the learning rate whenever validation accuracy
stops improving for 2 epochs — this lets the model fine-tune without
manually scheduling the LR.

---

## Data Pipeline

- **`MNISTDataset`** (custom `Dataset` subclass) wraps `torchvision.datasets.MNIST`
  and applies transforms inside `__getitem__`.
- Normalisation uses MNIST train-set statistics: mean = 0.1307, std = 0.3081.
- `random_split` with a fixed seed produces a reproducible 54 k / 6 k
  train / val partition.

---

## Checkpointing

The best model (by validation accuracy) is saved to `best_model.pth` after
every improvement. The checkpoint contains:

```python
{
    "epoch":               int,
    "model_state_dict":    OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "val_acc":             float,
}
```

To reload:

```python
ckpt = torch.load("best_model.pth")
model = FeedForwardNet()
model.load_state_dict(ckpt["model_state_dict"])
```

---

## How to Run

```bash
pip install torch torchvision
python mnist_classifier.py
```

MNIST is downloaded automatically to `./data/` on the first run.

---

## Checkpoint Question Answer

> *"Walk me through what `loss.backward()` does internally. What gets
> modified, and where do gradients live?"*

When you call `loss.backward()`, PyTorch traverses the **computational
graph** in reverse (backpropagation):

1. Starting from the scalar `loss`, it applies the **chain rule** recursively
   through every operation that was recorded during the forward pass.
2. For each `nn.Parameter` (i.e. every weight tensor with
   `requires_grad=True`), PyTorch **accumulates** the gradient of the loss
   with respect to that parameter into the tensor's `.grad` attribute.
3. **Gradients live in `.grad`** — for example, after `loss.backward()` you
   can inspect `model.net[0].weight.grad` to see the gradient matrix for the
   first linear layer.
4. The word *accumulates* is important: `.grad` is **added to**, not
   replaced, which is why `optimizer.zero_grad()` must be called before each
   batch — otherwise gradients from the previous step contaminate the current
   update.
