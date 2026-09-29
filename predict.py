import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_prep import prepare_data
from train import LogisticRegressionModel

OUTPUT_DIR = Path("outputs")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None)
    args = p.parse_args()

    _, _, X_test, y_test = prepare_data()
    model = LogisticRegressionModel()
    model.load_state_dict(torch.load(OUTPUT_DIR / "model.pt", map_location="cpu"))
    model.eval()

    if args.index is not None:
        x = torch.tensor(X_test[args.index], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze().numpy()
        pred = int(np.argmax(probs))
        true_label = int(y_test[args.index])
        print(f"Test image #{args.index}: true label = {true_label}, predicted = {pred}")
        plt.imshow(X_test[args.index].reshape(28, 28), cmap="gray")
        plt.title(f"True: {true_label} | Predicted: {pred}")
        plt.axis("off")
        plt.savefig(OUTPUT_DIR / f"prediction_{args.index}.png", dpi=150)
        print(f"Saved visualization to outputs/prediction_{args.index}.png")
    else:
        X_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            preds = torch.argmax(model(X_t), dim=1).numpy()
        accuracy = (preds == y_test).mean()
        print(f"Accuracy on full test set: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
