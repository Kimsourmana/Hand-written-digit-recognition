# Handwritten Digit Recognition — MNIST Logistic Regression

Multiclass logistic regression (a single linear layer + softmax) trained to
classify handwritten digits 0–9 from the MNIST dataset, adapted from the
provided Fashion-MNIST logistic regression example.

## Repository Contents

| File | Purpose |
|---|---|
| `data_prep.py` | Downloads MNIST-in-CSV (via `kagglehub`) and preprocesses it |
| `train.py` | Builds, trains, and evaluates the logistic regression model |
| `predict.py` | Loads the saved model and runs predictions without retraining |
| `requirements.txt` | Python dependencies |
| `outputs/` | Saved model, metrics, and figures |

## How to Run

```bash
pip install -r requirements.txt
python train.py --epochs 20 --batch-size 64 --lr 0.01   # trains and evaluates
python predict.py                                        # accuracy on test set
python predict.py --index 17                              # single prediction
```

`data_prep.py` downloads the dataset with:

```python
DATA_DIR = Path(kagglehub.dataset_download("oddrationale/mnist-in-csv"))
```

---

## Part 1 — Dataset and Preprocessing

**How many training and test images are there?**
Training set: 60,000 images. Test set: 10,000 images.

**What are the image dimensions and class labels?**
Each image is 28 × 28 grayscale pixels (784 values flattened). Class labels are the 10 digits 0–9.

**Why do we convert each image into 784 values?**
A logistic regression model is a single linear layer, `y = Wx + b`, which needs a flat feature vector as input, not a 2D grid. Flattening the 28 × 28 image gives 784 values (28 × 28 = 784), one per input feature.

**Why do we divide pixel values by 255?**
Raw pixels range 0–255. Dividing by 255 rescales them to [0, 1], which keeps input magnitudes small and consistent, making gradient-based optimization (SGD) more stable and converge faster.

**Why must training and test data remain separate?**
The test set estimates how well the model generalizes to unseen data. If test data leaks into training, evaluation becomes overly optimistic — the model could just memorize instead of learning generalizable patterns.

**Dataset shapes:**

Training set shape: X=(60000, 784), y=(60000,)
Test set shape: X=(10000, 784), y=(10000,)


Sample image per digit class:

<img width="1788" height="878" alt="image" src="https://github.com/user-attachments/assets/d4ae4286-3e58-4d3b-af10-8a86c9830493" />


---

## Part 2 — Model and Training

**Why 784 inputs and 10 outputs?**
784 inputs = one per pixel. 10 outputs = one logit per digit class (0–9); the highest logit is the predicted digit.

**Model architecture:**
```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)
```

**Why CrossEntropyLoss?**
This is multiclass classification with 10 mutually exclusive classes. `CrossEntropyLoss` applies softmax internally and penalizes the model based on how much probability it puts on the wrong class — the standard loss for this setting.

**Training settings:**

| Setting | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 0.01 (SGD) |
| Epochs | 20 |
| Optimizer | SGD |
| Loss | CrossEntropyLoss |

**How did training loss change?**

Epoch 1/20 - avg training loss: 0.9763
Epoch 5/20 - avg training loss: 0.4091
Epoch 10/20 - avg training loss: 0.3550
Epoch 15/20 - avg training loss: 0.3323
Epoch 20/20 - avg training loss: 0.3188


<img width="1088" height="674" alt="image" src="https://github.com/user-attachments/assets/6944cdcc-7464-4f6d-b75d-588db334a699" />


Loss drops sharply in the first few epochs (0.98 → 0.41) as the model learns the coarse patterns separating digits, then decreases more slowly and flattens by epoch ~15–20. This is expected for logistic regression: the loss surface is convex, so gradient descent steadily converges toward the best linear solution, with diminishing returns as it approaches that limit.

---

## Results

**Final test accuracy: 91.64%**
**Final training loss: 0.3188**

<img width="1056" height="878" alt="image" src="https://github.com/user-attachments/assets/ee8928f8-2002-46ce-bc60-66e42cca6876" />


Digits `0`, `1`, and `6` classify most reliably — they have distinct, simple strokes. Digits `5`, `8`, `2`, `9`, and `3` are hardest, since they're visually similar to each other when handwritten and a linear model can only draw straight decision boundaries in pixel space. ~91.6% is a reasonable ceiling for plain logistic regression on MNIST; CNNs typically reach 98–99%+ by learning spatial features a flat linear model cannot.
