from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_data_dir():
    import kagglehub
    return Path(kagglehub.dataset_download("oddrationale/mnist-in-csv"))

def load_csv(path):
    df = pd.read_csv(path)
    values = df.values.astype(np.float32)
    return values[:, 1:], values[:, 0].astype(np.int64)

def prepare_data():
    data_dir = get_data_dir()
    X_train, y_train = load_csv(data_dir / "mnist_train.csv")
    X_test, y_test = load_csv(data_dir / "mnist_test.csv")
    X_train, X_test = X_train / 255.0, X_test / 255.0

    print(f"Train images: {X_train.shape[0]}, Test images: {X_test.shape[0]}")
    print(f"Image dim: {X_train.shape[1]} (28x28 flattened), Classes: {sorted(np.unique(y_train).tolist())}")

    fig, axes = plt.subplots(2, 5, figsize=(10, 5.5))
    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        ax = axes[digit // 5, digit % 5]
        ax.imshow(X_train[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {digit}")
        ax.axis("off")
    plt.suptitle("One sample image per digit class (0-9)")
    plt.subplots_adjust(hspace=0.5, top=0.88)
    plt.savefig(OUTPUT_DIR / "sample_digits.png", dpi=150)
    plt.close()

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    prepare_data()
