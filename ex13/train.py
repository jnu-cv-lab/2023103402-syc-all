"""
Step 2: Train Skeleton Transformer on badminton stroke classification.
Saves model checkpoint, training curves, and accuracy logs to ./results/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"D:\实验13_骨架Transformer羽毛球动作识别\data"
RESULTS_DIR = r"D:\实验13_骨架Transformer羽毛球动作识别\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3


class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, seq_len=TARGET_FRAMES):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: [B, T, input_dim]
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.input_proj(x) + self.pos_emb(pos)
        x = self.encoder(x)          # [B, T, d_model]
        x = x.mean(dim=1)            # mean pooling over time
        return self.classifier(x)    # [B, num_classes]


def train():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    train_loader = DataLoader(SkeletonDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SkeletonDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SkeletonTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
        scheduler.step()

        avg_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # Eval on test set
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                t_correct += (preds == y_batch).sum().item()
                t_total += len(y_batch)
        test_acc = t_correct / t_total
        test_accs.append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pth"))

    # Save training log
    log = {"train_loss": train_losses, "train_acc": train_accs, "test_acc": test_accs}
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Plot training curves
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, train_losses, color="black", linewidth=1.5)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, train_accs, color="black", linewidth=1.5, label="Train")
    axes[1].plot(epochs_range, test_accs, color="gray", linewidth=1.5,
                 linestyle="--", label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFinal Test Accuracy: {test_accs[-1]:.4f}")
    print(f"Best Test Accuracy:  {max(test_accs):.4f} at epoch {test_accs.index(max(test_accs)) + 1}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    train()
