import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import json

BATCH_SIZE   = 128
EPOCHS       = 15
LR           = 1e-3
HIDDEN_DIMS  = [512, 256, 128]
DROPOUT_P    = 0.3
CHECKPOINT   = "best_model.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT    = 0.1


class MNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, transform=None):
        self.data = datasets.MNIST(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class FeedForwardNet(nn.Module):
    def __init__(self, hidden_dims=HIDDEN_DIMS, dropout_p=DROPOUT_P):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)

    return total_loss / n, correct / n * 100


def main():
    print(f"Using device: {DEVICE}\n")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = MNISTDataset("./data", train=True,  transform=tf)
    test_set   = MNISTDataset("./data", train=False, transform=tf)

    val_size   = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_set)}\n")

    model     = FeedForwardNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion)

        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>2}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
            f"lr={lr_now:.6f}"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":    val_acc,
            }, CHECKPOINT)
            print(f"  Checkpoint saved  (best val acc = {best_val_acc:.2f}%)")

    print("\nLoading best checkpoint for final evaluation ...")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc = run_epoch(model, test_loader, criterion)
    print(f"\n{'='*55}")
    print(f"  Final test accuracy : {test_acc:.2f}%")
    print(f"  Best val  accuracy  : {best_val_acc:.2f}%")
    print(f"{'='*55}")

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training history saved to training_history.json")


if __name__ == "__main__":
    main()
