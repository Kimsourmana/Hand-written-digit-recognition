import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data_prep import prepare_data

OUTPUT_DIR = Path("outputs")
torch.manual_seed(42)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()

    X_train, y_train, X_test, y_test = prepare_data()
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)), batch_size=args.batch_size)

    model = LogisticRegressionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    losses = []
    for epoch in range(1, args.epochs + 1):
        total, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
        losses.append(total / n)
        print(f"Epoch {epoch:2d}/{args.epochs} - avg training loss: {losses[-1]:.4f}")

    model.eval()
    correct, total_n = 0, 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item(); total_n += yb.size(0)
            for t, pr in zip(yb.numpy(), preds.numpy()):
                confusion[t, pr] += 1
    accuracy = correct / total_n
    print(f"\nTest accuracy: {accuracy*100:.2f}%")

    plt.figure(figsize=(6,4))
    plt.plot(range(1, args.epochs+1), losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Average training loss")
    plt.title("Training Loss over Epochs"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(6,5))
    plt.imshow(confusion, cmap="Blues"); plt.colorbar()
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title("Confusion Matrix (Test Set)")
    for i in range(10):
        for j in range(10):
            c = "white" if confusion[i,j] > confusion.max()/2 else "black"
            plt.text(j, i, str(confusion[i,j]), ha="center", va="center", color=c, fontsize=7)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    metrics = {
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "final_train_loss": losses[-1], "test_accuracy": accuracy,
        "per_class_accuracy": {str(i): float(confusion[i,i]/confusion[i].sum()) for i in range(10)},
        "epoch_losses": losses,
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model, metrics, and figures to outputs/")

if __name__ == "__main__":
    main()
