# AppliedML-Evaluation Lecture

| Field               | Value                                                        |
|---------------------|--------------------------------------------------------------|
| Activity Name       | AIML-LEC-AppliedML-Evaluation                                |
| Display Name        | AppliedML-Evaluation Lecture                                  |
| Duration            | 180 min (3 hours)                                            |
| Tech Stack          | Python 3.10+, numpy, pandas, scikit-learn, matplotlib, seaborn, shap |
| Unified Scenario    | Credit Card Fraud Detection (synthetic, 95/5 class imbalance)|

---

## Pre-Lecture Checklist (Instructor)

- [ ] Python 3.10+ environment with all packages installed
- [ ] Verify `pip install numpy pandas scikit-learn matplotlib seaborn shap` completes cleanly
- [ ] IDE or Jupyter Notebook open and tested
- [ ] Terminal ready for Git commands at end of session
- [ ] Display/projector configured with readable font size (14pt+ recommended)
- [ ] Associates have completed all 8 prerequisite readings/videos

---

## Pacing Legend

| Symbol              | Meaning                                                      |
|---------------------|--------------------------------------------------------------|
| `[LINE-BY-LINE]`    | Type each line live, explain as you go                       |
| `[BLOCK-UPDATE]`    | Paste or type the full block, then walk through it           |
| `[PAUSE FOR Q&A]`   | Stop for 3-5 min of questions                                |
| `[PAUSE FOR BREAK]` | 10 min break                                                 |
| `STEP N:`           | Incremental step marker inside code                          |

---

## Lecture Timeline Overview

| Time          | Duration | Content                                                    |
|---------------|----------|------------------------------------------------------------|
| 0:00 - 0:10  | 10 min   | Opening: Scenario introduction, module objectives          |
| 0:10 - 0:55  | 45 min   | **Stage 1** -- Splits, stratification, cross-val, loss fns |
| 0:55 - 1:00  | 5 min    | Q&A                                                        |
| 1:00 - 1:10  | 10 min   | Break                                                      |
| 1:10 - 1:55  | 45 min   | **Stage 2** -- Bias-variance, regularization, learning curves |
| 1:55 - 2:00  | 5 min    | Q&A                                                        |
| 2:00 - 2:10  | 10 min   | Break                                                      |
| 2:10 - 2:55  | 45 min   | **Stage 3** -- Metrics suite, SHAP, early stopping         |
| 2:55 - 3:00  | 5 min    | Wrap-up, Git activity, closing Q&A                         |

---

## Opening (0:00 - 0:10)

### Instructor Talking Points

Welcome the class. Frame the entire session around a single real-world problem: **detecting fraudulent credit card transactions**. Explain that fraud detection is the ideal scenario for this module because:

1. The dataset is naturally imbalanced -- only about 5% of transactions are fraudulent. This forces us to confront why accuracy alone is a terrible metric.
2. The cost of a false negative (missing fraud) is dramatically higher than a false positive (flagging a legitimate purchase), which motivates precision/recall tradeoffs.
3. The model must generalize to unseen transactions, which surfaces bias-variance and overfitting concerns directly.
4. Regulatory and business stakeholders need explanations for model decisions, motivating SHAP explainability.

State the exit criteria clearly. By the end of this session, Associates will be able to:

- Implement robust train/validation/test splitting with stratification
- Compare MSE and CrossEntropy loss mathematically and practically
- Diagnose bias-variance tradeoff from learning curves
- Apply L1, L2, and Dropout regularization
- Calculate precision, recall, F1 for imbalanced classes
- Generate and interpret AUC-ROC curves and confusion matrices
- (Preferred) Implement stratified k-fold, early stopping, and SHAP explanations

---

## Stage 1: Data Splits, Stratification, Cross-Validation, and Loss Functions (0:10 - 0:55)

### Instructor Notes

This stage builds the foundation. Associates need to see the data first, understand why naive splitting can destroy rare-class representation, and then connect loss functions to the classification vs. regression distinction. Type the dataset generation code line-by-line so Associates follow the construction logic. The splitting and cross-validation code can be done as block updates since the sklearn API is straightforward.

---

### STEP 1: Generate the Synthetic Fraud Dataset

`[LINE-BY-LINE]` -- Type each line and narrate. Explain the deliberate 95/5 imbalance.

```python
# STEP 1: Generate synthetic credit card fraud dataset
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_classes=2,
    weights=[0.95, 0.05],   # 95% legitimate, 5% fraud
    flip_y=0.02,
    random_state=42
)

feature_names = [f"txn_feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["is_fraud"] = y

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['is_fraud'].value_counts(normalize=True)}")
```

**Talking Point:** Ask the class -- "If we just predicted every transaction as legitimate, what accuracy would we get?" (Answer: ~95%). This is why accuracy is misleading and why we need everything we will cover today.

---

### STEP 2: Train / Validation / Test Split (Naive vs. Stratified)

`[LINE-BY-LINE]` for the naive split. `[BLOCK-UPDATE]` for the stratified split so Associates can compare outputs.

```python
# STEP 2a: Naive random split -- observe class distribution drift
from sklearn.model_selection import train_test_split

X_all = df.drop("is_fraud", axis=1).values
y_all = df["is_fraud"].values

X_temp, X_test_naive, y_temp, y_test_naive = train_test_split(
    X_all, y_all, test_size=0.15, random_state=0   # no stratify
)
X_train_naive, X_val_naive, y_train_naive, y_val_naive = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=0  # ~15% of original
)

print("=== NAIVE SPLIT (no stratification) ===")
print(f"Train fraud rate:  {y_train_naive.mean():.4f}")
print(f"Val fraud rate:    {y_val_naive.mean():.4f}")
print(f"Test fraud rate:   {y_test_naive.mean():.4f}")
```

**Talking Point:** Point out that the fraud rates may differ significantly across splits. In a dataset with only 500 fraud cases total, losing even a few dozen to an unlucky split materially changes what the model learns.

```python
# STEP 2b: Stratified split -- preserves class proportions
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print("\n=== STRATIFIED SPLIT ===")
print(f"Train fraud rate:  {y_train.mean():.4f}")
print(f"Val fraud rate:    {y_val.mean():.4f}")
print(f"Test fraud rate:   {y_test.mean():.4f}")
print(f"\nSplit sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

**Talking Point:** The fraud rate should be approximately 0.05 across all three sets. Emphasize that `stratify=y` is a single parameter but has enormous practical impact.

**Exit Criterion Addressed:** *Implement robust train, validation, and test dataset splitting algorithms.*

---

### STEP 3: Stratified K-Fold Cross-Validation (Preferred)

`[BLOCK-UPDATE]` -- paste and walk through.

```python
# STEP 3: Stratified K-Fold Cross-Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_model = LogisticRegression(max_iter=1000, random_state=42)

cv_scores = cross_val_score(
    baseline_model, X_train, y_train,
    cv=skf,
    scoring="f1"
)

print("=== Stratified 5-Fold CV (F1 Score) ===")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean:   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
```

**Talking Point:** Explain why we use F1 as the scoring metric here rather than accuracy. With 5% fraud, accuracy would be uniformly high and uninformative. Also clarify that cross-validation is used on the training set only -- the test set remains untouched until final evaluation.

**Exit Criterion Addressed:** *(Preferred) Implement stratified k-fold cross-validation.*

---

### STEP 4: Loss Functions -- MSE vs. CrossEntropy

`[LINE-BY-LINE]` for the math. `[BLOCK-UPDATE]` for the visualization.

```python
# STEP 4: MSE vs. CrossEntropy -- mathematical comparison
import matplotlib.pyplot as plt

y_true_example = 1.0  # true label: fraud

p = np.linspace(0.001, 0.999, 500)  # predicted probability

# MSE loss for a single sample: (y_true - p)^2
mse_loss = (y_true_example - p) ** 2

# Binary CrossEntropy for a single sample: -[y*log(p) + (1-y)*log(1-p)]
bce_loss = -(y_true_example * np.log(p) + (1 - y_true_example) * np.log(1 - p))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(p, mse_loss, color="steelblue", linewidth=2)
axes[0].set_title("MSE Loss (y_true = 1)")
axes[0].set_xlabel("Predicted Probability (p)")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(p, bce_loss, color="firebrick", linewidth=2)
axes[1].set_title("Binary CrossEntropy Loss (y_true = 1)")
axes[1].set_xlabel("Predicted Probability (p)")
axes[1].set_ylabel("Loss")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step4_loss_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Talking Points:**

- **MSE (Mean Squared Error):** The loss is the squared difference between the true label and the prediction. The gradient is proportional to `(p - y)`, which becomes very small when `p` is near 0 and `y=1`. This means the model receives a weak learning signal precisely when it is most wrong -- a critical problem for classification.
- **CrossEntropy:** The loss includes a `log(p)` term, which goes to infinity as `p` approaches 0. This creates a steep gradient when the model is confidently wrong, driving much faster and more reliable learning for classification tasks.
- **Practical rule:** Use MSE for regression (predicting house prices, temperatures). Use CrossEntropy for classification (predicting fraud, spam, disease).
- **Fraud context:** If our model predicts p=0.01 for a fraudulent transaction, CrossEntropy penalizes this far more harshly than MSE, which is exactly what we want.

```python
# Demonstrate the gradient difference at a specific wrong prediction
p_wrong = 0.05  # model predicts 5% fraud probability for an actual fraud case
print(f"At p = {p_wrong} (true label = 1):")
print(f"  MSE loss:          {(1 - p_wrong)**2:.4f}")
print(f"  CrossEntropy loss: {-np.log(p_wrong):.4f}")
print(f"  MSE gradient:      {2 * (p_wrong - 1):.4f}")
print(f"  CE gradient:       {-1/p_wrong:.4f}")
print("\nCrossEntropy gradient is ~10x stronger -- the model learns faster from mistakes.")
```

**Exit Criterion Addressed:** *Describe the mathematical difference and practical application of MSE vs. CrossEntropy.*

---

### [PAUSE FOR Q&A] (0:55 - 1:00)

Suggested prompts:
- "Why can't we use MSE for our fraud classifier?"
- "What would happen if we did a 50/50 random split on this dataset?"
- "Why do we stratify the k-fold and not just the train/test split?"

### [PAUSE FOR BREAK] (1:00 - 1:10)

---

## Stage 2: Bias-Variance Diagnostics, Regularization, and Learning Curves (1:10 - 1:55)

### Instructor Notes

This stage moves from data preparation to model behavior. The key pedagogical goal is to make bias-variance tangible through learning curves -- Associates should be able to look at a plot and diagnose the problem. Then regularization is presented as the direct remedy. Start with sklearn (Ridge/Lasso) for classical regularization, then use framework-agnostic pseudocode for Dropout since Dropout is inherently a neural network technique and the concept matters more than any specific implementation.

---

### STEP 5: Learning Curves -- Diagnosing Bias vs. Variance

`[BLOCK-UPDATE]` -- the learning_curve API produces a lot of output. Paste the block, then walk through the plot line by line.

```python
# STEP 5: Learning curves for bias-variance diagnosis
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, cv=5, scoring="f1"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="steelblue")
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color="firebrick")
    plt.plot(train_sizes, train_mean, "o-", color="steelblue",
             linewidth=2, label="Training score")
    plt.plot(train_sizes, val_mean, "o-", color="firebrick",
             linewidth=2, label="Validation score")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel(f"{scoring.upper()} Score")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"step5_{title.replace(' ', '_').lower()}.png", dpi=150,
                bbox_inches="tight")
    plt.show()
```

```python
# STEP 5a: High-bias model (underfitting example)
from sklearn.linear_model import LogisticRegression

simple_model = LogisticRegression(max_iter=1000, C=0.001, random_state=42)
plot_learning_curve(simple_model, "High Bias - Underfitting (C=0.001)",
                    X_train, y_train)
```

```python
# STEP 5b: High-variance model (overfitting example)
from sklearn.tree import DecisionTreeClassifier

complex_model = DecisionTreeClassifier(max_depth=None, random_state=42)
plot_learning_curve(complex_model, "High Variance - Overfitting (Unbounded Tree)",
                    X_train, y_train)
```

```python
# STEP 5c: Well-balanced model
balanced_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
plot_learning_curve(balanced_model, "Balanced - Good Fit (C=1.0)",
                    X_train, y_train)
```

**Talking Points:**

- **High Bias (Underfitting):** Both training and validation scores are low and converge early. The model is too simple to capture the fraud patterns. Adding more data will not help.
- **High Variance (Overfitting):** Training score is near-perfect but validation score is significantly lower. The gap between the curves is the hallmark of overfitting. The model memorizes training noise.
- **Good Fit:** Both curves converge to a reasonable level with a small gap. This is our target.
- **Diagnostic checklist:**
  - Large gap between curves = variance problem = need regularization or simpler model
  - Both curves low = bias problem = need more features or more complex model
  - Both curves high and close = good generalization

**Exit Criterion Addressed:** *Identify signs of bias-variance tradeoff and overfitting from learning curves.*

---

### STEP 6: L1 (Lasso) and L2 (Ridge) Regularization

`[LINE-BY-LINE]` for the model definitions. `[BLOCK-UPDATE]` for the coefficient comparison.

```python
# STEP 6: L1 (Lasso) and L2 (Ridge) Regularization
from sklearn.linear_model import LogisticRegression

# L2 Regularization (Ridge) -- penalty='l2' is the default
ridge_model = LogisticRegression(
    penalty="l2",
    C=1.0,         # C = 1/lambda; smaller C = stronger regularization
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
ridge_model.fit(X_train, y_train)

# L1 Regularization (Lasso) -- requires solver that supports L1
lasso_model = LogisticRegression(
    penalty="l1",
    C=1.0,
    max_iter=1000,
    solver="saga",  # saga supports L1
    random_state=42
)
lasso_model.fit(X_train, y_train)

print("=== Coefficient Comparison ===")
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Ridge (L2)": ridge_model.coef_[0],
    "Lasso (L1)": lasso_model.coef_[0]
})
coef_df["L1_is_zero"] = np.abs(coef_df["Lasso (L1)"]) < 1e-6
print(coef_df.to_string(index=False))
print(f"\nFeatures zeroed out by Lasso: {coef_df['L1_is_zero'].sum()} / {len(feature_names)}")
```

**Talking Points:**

- **L2 (Ridge)** adds `lambda * sum(w^2)` to the loss. It shrinks all coefficients toward zero but rarely makes them exactly zero. Think of it as "turn down the volume on all features."
- **L1 (Lasso)** adds `lambda * sum(|w|)` to the loss. It drives some coefficients to exactly zero, performing automatic feature selection. Think of it as "mute irrelevant features entirely."
- **`C` parameter:** In sklearn, `C = 1 / lambda`. Smaller C = stronger regularization. This is the inverse of the lambda in the mathematical formulation, which trips up many Associates.
- **Fraud context:** We generated 20 features but only 12 are informative. Lasso should zero out some of the 4 redundant and 4 noise features.

```python
# STEP 6b: Visualize coefficient magnitudes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(feature_names, ridge_model.coef_[0], color="steelblue")
axes[0].set_title("Ridge (L2) Coefficients")
axes[0].axvline(x=0, color="black", linewidth=0.5)

axes[1].barh(feature_names, lasso_model.coef_[0], color="firebrick")
axes[1].set_title("Lasso (L1) Coefficients")
axes[1].axvline(x=0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("step6_regularization_coefficients.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Exit Criterion Addressed:** *Implement L1 (Lasso) and L2 (Ridge) regularization.*

---

### STEP 7: Dropout Regularization (Conceptual, Framework-Agnostic)

`[LINE-BY-LINE]` for the pseudocode. Walk through the architecture and the training/inference distinction carefully.

**Instructor Note:** Dropout is inherently a neural network technique. The following pseudocode illustrates the concept without tying to any specific deep learning framework. Associates should focus on *what* Dropout does and *why*, not on framework syntax.

#### Network Architecture with Dropout

```
PSEUDOCODE -- Neural Network with Dropout

Architecture:
    Input Layer:    20 features (one per transaction feature)
    Hidden Layer 1: 64 neurons, activation = ReLU
    Dropout Layer:  drop rate = 0.3  (randomly zero out 30% of neurons)
    Hidden Layer 2: 32 neurons, activation = ReLU
    Dropout Layer:  drop rate = 0.3  (randomly zero out 30% of neurons)
    Output Layer:   1 neuron,  activation = Sigmoid (outputs fraud probability)
```

#### Training vs. Inference Behavior

```
PSEUDOCODE -- Training Loop with Dropout

FOR each epoch in 1..50:
    SET model to TRAINING mode
    # During training, Dropout layers randomly zero out 30% of neuron outputs.
    # Each mini-batch sees a different random subset of active neurons.

    FOR each mini-batch (X_batch, y_batch) in training_data:
        predictions = forward_pass(X_batch)       # Dropout is ACTIVE
        loss = binary_cross_entropy(predictions, y_batch)
        gradients = compute_gradients(loss)
        update_weights(gradients, learning_rate=0.001)

    SET model to EVALUATION mode
    # During evaluation, ALL neurons are active, but outputs are scaled
    # by (1 - drop_rate) to compensate for the neurons that were dropped
    # during training. This ensures consistent output magnitude.

    val_predictions = forward_pass(X_val)          # Dropout is INACTIVE
    val_loss = binary_cross_entropy(val_predictions, y_val)

    RECORD train_loss, val_loss
```

#### Comparing With and Without Dropout

```
PSEUDOCODE -- Comparing Dropout Effect

Model A: Network WITH Dropout (rate = 0.3)
Model B: Network WITHOUT Dropout (identical architecture, no Dropout layers)

Train both models on the same data for 50 epochs.
Record training loss and validation loss at each epoch.

Expected observation:
    Model B (no Dropout):
        - Training loss decreases steadily toward zero
        - Validation loss decreases initially, then INCREASES (overfitting)
        - Growing gap between training and validation loss

    Model A (with Dropout):
        - Training loss decreases more slowly (harder to memorize with random neuron masking)
        - Validation loss stays closer to training loss (better generalization)
        - Smaller gap between curves = less overfitting
```

After discussing the pseudocode, show the expected shape of the loss curves:

```python
# STEP 7: Illustrative plot of Dropout effect on training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = np.arange(1, 51)

# Simulated curves illustrating the expected behavior
dropout_train = 0.5 * np.exp(-0.04 * epochs) + 0.12 + np.random.normal(0, 0.005, 50)
dropout_val   = 0.5 * np.exp(-0.03 * epochs) + 0.15 + np.random.normal(0, 0.008, 50)

no_drop_train = 0.5 * np.exp(-0.06 * epochs) + 0.05 + np.random.normal(0, 0.005, 50)
no_drop_val   = 0.5 * np.exp(-0.03 * epochs) + 0.10 + 0.004 * epochs + np.random.normal(0, 0.008, 50)

axes[0].plot(epochs, dropout_train, label="Train", color="steelblue")
axes[0].plot(epochs, dropout_val, label="Validation", color="firebrick")
axes[0].set_title("WITH Dropout (p=0.3)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, no_drop_train, label="Train", color="steelblue")
axes[1].plot(epochs, no_drop_val, label="Validation", color="firebrick")
axes[1].set_title("WITHOUT Dropout")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("BCE Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step7_dropout_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Talking Points:**

- **How Dropout works:** During each training forward pass, each neuron in a Dropout layer is randomly "turned off" with probability `p`. This forces the network to not rely on any single neuron, building redundancy.
- **Training vs. Inference:** Dropout is ONLY active during training. At inference time, all neurons are active but their outputs are scaled by `(1-p)` to compensate. This is critical -- forgetting to switch modes is a common bug in any framework.
- **Why it works:** Dropout approximates training an ensemble of many sub-networks. Each forward pass trains a different sub-network. At inference, we effectively average their predictions.
- **The plot above uses simulated curves** to illustrate the expected pattern. In practice, you would observe this same pattern when training a real neural network with and without Dropout on this fraud dataset.

**Talking Point:** The model without Dropout should show a growing gap between train and validation loss (overfitting). The Dropout model should show the curves staying closer together.

**Exit Criterion Addressed:** *Understand and describe Dropout regularization in neural networks.*

---

### [PAUSE FOR Q&A] (1:55 - 2:00)

Suggested prompts:
- "When would you choose L1 over L2?"
- "What happens if you set Dropout to 0.9? To 0.0?"
- "How do you decide whether your model has a bias or variance problem?"

### [PAUSE FOR BREAK] (2:00 - 2:10)

---

## Stage 3: Full Evaluation Suite, SHAP Values, and Early Stopping (2:10 - 2:55)

### Instructor Notes

This is the culmination of the session. Everything built so far supports what happens here: we train a final model on the stratified data with appropriate regularization, then evaluate it with every metric the Associates need to master. The SHAP and early stopping demos are marked as preferred exit criteria but are critical for demonstrating professional ML practice. Pace the precision/recall/F1 section carefully -- Associates often confuse the denominators.

---

### STEP 8: Train the Final Evaluation Model

`[BLOCK-UPDATE]` -- straightforward sklearn fit.

```python
# STEP 8: Train the final model for evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

eval_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",  # upweight minority class
        random_state=42
    ))
])

eval_pipeline.fit(X_train, y_train)

y_val_pred = eval_pipeline.predict(X_val)
y_val_proba = eval_pipeline.predict_proba(X_val)[:, 1]

y_test_pred = eval_pipeline.predict(X_test)
y_test_proba = eval_pipeline.predict_proba(X_test)[:, 1]

print("Model trained. Predictions generated on validation and test sets.")
```

**Talking Point:** Note the use of `class_weight="balanced"`. This tells sklearn to automatically adjust weights inversely proportional to class frequencies, so the model pays more attention to the rare fraud class. Without this, the model might learn to predict "not fraud" for everything and achieve 95% accuracy.

---

### STEP 9: Precision, Recall, and F1 for Imbalanced Classification

`[LINE-BY-LINE]` -- this is the conceptual heart of the metrics section. Slow down.

```python
# STEP 9: Precision, Recall, F1
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("=== Test Set Metrics (Fraud = Positive Class) ===")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print()
print("Full Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Legitimate", "Fraud"]))
```

**Talking Points -- Write these formulas on the board or share screen:**

- **Precision = TP / (TP + FP):** "Of all transactions we FLAGGED as fraud, what fraction were actually fraud?" High precision = few false alarms.
- **Recall = TP / (TP + FN):** "Of all transactions that WERE fraud, what fraction did we catch?" High recall = we miss few frauds.
- **F1 = 2 * (Precision * Recall) / (Precision + Recall):** The harmonic mean. It penalizes extreme imbalances between precision and recall more than an arithmetic mean would.
- **Fraud context:** A fraud detection system with high precision but low recall catches fraud when it flags something, but misses many actual frauds. A system with high recall but low precision catches most frauds but also flags many legitimate transactions, annoying customers. The business must decide the tradeoff.

```python
# STEP 9b: Manual calculation to build intuition
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

print("=== Manual Calculation ===")
print(f"  True Positives  (caught fraud):         {tp}")
print(f"  False Positives (false alarms):          {fp}")
print(f"  True Negatives  (correct legitimate):    {tn}")
print(f"  False Negatives (missed fraud):           {fn}")
print()
manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
manual_f1 = 2 * manual_precision * manual_recall / (manual_precision + manual_recall) if (manual_precision + manual_recall) > 0 else 0
print(f"  Manual Precision: {manual_precision:.4f}")
print(f"  Manual Recall:    {manual_recall:.4f}")
print(f"  Manual F1:        {manual_f1:.4f}")
```

**Exit Criterion Addressed:** *Calculate and interpret Precision, Recall, and F1 scores for imbalanced classification.*

---

### STEP 10: Confusion Matrix Visualization

`[BLOCK-UPDATE]` -- paste and explain the heatmap.

```python
# STEP 10: Confusion Matrix heatmap
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    display_labels=["Legitimate", "Fraud"],
    cmap="Blues",
    ax=ax,
    values_format="d"
)
ax.set_title("Confusion Matrix -- Credit Card Fraud Detection")
plt.tight_layout()
plt.savefig("step10_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Talking Point:** Walk through each quadrant. Top-left (TN) should be the largest -- most transactions are legitimate and correctly classified. Bottom-right (TP) shows caught frauds. Top-right (FP) shows false alarms. Bottom-left (FN) shows missed frauds -- the most dangerous cell in fraud detection.

---

### STEP 11: AUC-ROC Curve

`[LINE-BY-LINE]` for the ROC calculation. `[BLOCK-UPDATE]` for the plot.

```python
# STEP 11: AUC-ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

auc_score = roc_auc_score(y_test, y_test_proba)
print(f"AUC-ROC Score: {auc_score:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    y_test, y_test_proba,
    name="Logistic Regression",
    ax=ax,
    color="steelblue",
    linewidth=2
)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC=0.5)")
ax.set_title(f"ROC Curve -- AUC = {auc_score:.4f}")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step11_roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Talking Points:**

- **ROC Curve:** Plots True Positive Rate (Recall) on the y-axis against False Positive Rate (FPR = FP / (FP + TN)) on the x-axis at every possible classification threshold.
- **AUC (Area Under the Curve):** A single number summarizing the ROC curve. AUC = 1.0 is perfect. AUC = 0.5 is random guessing. AUC < 0.5 means the model is worse than random (likely labels are inverted).
- **Threshold independence:** Unlike precision/recall/F1 which depend on a specific threshold (default 0.5), AUC evaluates the model across ALL thresholds. This makes it useful for comparing models before choosing an operating threshold.
- **Fraud context:** A high AUC means the model generally assigns higher probabilities to actual fraud cases than to legitimate ones, regardless of where we set the cutoff.

```python
# STEP 11b: Precision-Recall Curve (better for imbalanced data)
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

ap_score = average_precision_score(y_test, y_test_proba)

fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_predictions(
    y_test, y_test_proba,
    name="Logistic Regression",
    ax=ax,
    color="firebrick",
    linewidth=2
)
ax.set_title(f"Precision-Recall Curve -- AP = {ap_score:.4f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step11b_precision_recall_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Average Precision Score: {ap_score:.4f}")
```

**Talking Point:** For highly imbalanced datasets, the Precision-Recall curve is often more informative than ROC. ROC can look optimistic because the large number of true negatives inflates the FPR denominator. Precision-Recall focuses only on the positive class.

**Exit Criterion Addressed:** *Generate and analyze an AUC-ROC curve and Confusion Matrix.*

---

### STEP 12: SHAP Values for Model Explainability (Preferred)

`[BLOCK-UPDATE]` -- SHAP computation can take a moment. Paste and explain while it runs.

```python
# STEP 12: SHAP Values for Explainability
import shap

explainer = shap.LinearExplainer(
    eval_pipeline.named_steps["classifier"],
    eval_pipeline.named_steps["scaler"].transform(X_train),
    feature_names=feature_names
)

X_test_scaled = eval_pipeline.named_steps["scaler"].transform(X_test)
shap_values = explainer.shap_values(X_test_scaled)

print(f"SHAP values shape: {shap_values.shape}")
print("(Each test sample has one SHAP value per feature)")
```

```python
# STEP 12a: Summary plot -- global feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot -- Feature Impact on Fraud Prediction")
plt.tight_layout()
plt.savefig("step12_shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# STEP 12b: Force plot for a single prediction (first fraud case in test set)
fraud_indices = np.where(y_test == 1)[0]
if len(fraud_indices) > 0:
    sample_idx = fraud_indices[0]
    print(f"Explaining prediction for test sample {sample_idx}")
    print(f"  True label: {'Fraud' if y_test[sample_idx] == 1 else 'Legitimate'}")
    print(f"  Predicted probability: {y_test_proba[sample_idx]:.4f}")

    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X_test_scaled[sample_idx],
        feature_names=feature_names
    )
    shap.save_html("step12_shap_force_plot.html", force_plot)
    print("Force plot saved to step12_shap_force_plot.html")
```

```python
# STEP 12c: Bar plot -- mean absolute SHAP values (feature importance ranking)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("Mean |SHAP Value| -- Feature Importance Ranking")
plt.tight_layout()
plt.savefig("step12_shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Talking Points:**

- **What SHAP values are:** SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for each individual prediction. A positive SHAP value pushes the prediction toward fraud; a negative value pushes toward legitimate.
- **Summary plot:** Each dot is one test sample. The x-axis shows the SHAP value (impact on prediction). The color shows the feature value (red = high, blue = low). Features are sorted by overall importance.
- **Force plot:** Shows how features combine to push one specific prediction away from the baseline (average prediction). Red arrows push toward fraud, blue arrows push toward legitimate.
- **Fraud context:** A compliance officer can use these plots to understand WHY the model flagged a specific transaction, which may be a regulatory requirement.

**Exit Criterion Addressed:** *(Preferred) Generate SHAP values.*

---

### STEP 13: Early Stopping (Conceptual, Framework-Agnostic) (Preferred)

`[LINE-BY-LINE]` for the EarlyStopping pseudocode. This is a design pattern Associates should understand thoroughly, independent of any specific framework.

**Instructor Note:** Early Stopping is a general regularization technique applicable to any iterative learning algorithm (neural networks, gradient boosting, etc.). The following pseudocode presents the pattern in a framework-agnostic way so Associates understand the logic before encountering it in any specific library.

#### Early Stopping Logic

```
PSEUDOCODE -- Early Stopping Callback

CLASS EarlyStopping:
    PARAMETERS:
        patience        = 5       # How many epochs to wait after last improvement
        min_delta       = 0.001   # Minimum change to count as an improvement
        restore_best    = True    # Whether to restore weights from the best epoch

    STATE:
        best_loss       = infinity
        wait_counter    = 0
        best_weights    = None
        stopped_epoch   = 0

    METHOD check(current_val_loss, model, epoch):
        IF current_val_loss < best_loss - min_delta:
            # Improvement found -- reset counter, save weights
            best_loss = current_val_loss
            wait_counter = 0
            IF restore_best:
                best_weights = copy(model.weights)
        ELSE:
            # No improvement -- increment counter
            wait_counter += 1

        IF wait_counter >= patience:
            # Patience exhausted -- stop training
            stopped_epoch = epoch
            IF restore_best AND best_weights is not None:
                model.weights = best_weights
            RETURN True   # signal: stop training

        RETURN False      # signal: continue training
```

#### Training Loop with Early Stopping

```
PSEUDOCODE -- Training with Early Stopping

early_stopper = EarlyStopping(patience=7, min_delta=0.001)
max_epochs = 200

FOR epoch in 1..max_epochs:
    SET model to TRAINING mode

    FOR each mini-batch (X_batch, y_batch) in training_data:
        predictions = forward_pass(X_batch)
        loss = binary_cross_entropy(predictions, y_batch)
        gradients = compute_gradients(loss)
        update_weights(gradients, learning_rate=0.001)

    SET model to EVALUATION mode
    val_predictions = forward_pass(X_val)
    val_loss = binary_cross_entropy(val_predictions, y_val)

    RECORD train_loss, val_loss

    IF early_stopper.check(val_loss, model, epoch) == True:
        PRINT "Early stopping at epoch {epoch}"
        PRINT "Best validation loss: {early_stopper.best_loss}"
        PRINT "Weights restored from best epoch."
        BREAK

    IF epoch == max_epochs:
        PRINT "Reached max epochs without early stopping trigger."
```

After discussing the pseudocode, show the expected shape of early stopping on a loss plot:

```python
# STEP 13: Illustrative plot of Early Stopping behavior
np.random.seed(42)

n_epochs = 80
stopped_epoch = 55
best_epoch = 40

epochs = np.arange(1, n_epochs + 1)
train_loss = 0.6 * np.exp(-0.05 * epochs) + 0.04 + np.random.normal(0, 0.005, n_epochs)
val_loss = 0.6 * np.exp(-0.04 * epochs) + 0.08 + 0.002 * np.maximum(epochs - best_epoch, 0) + np.random.normal(0, 0.008, n_epochs)

# Only show up to the stopped epoch
train_loss_shown = train_loss[:stopped_epoch]
val_loss_shown = val_loss[:stopped_epoch]
epochs_shown = epochs[:stopped_epoch]

plt.figure(figsize=(10, 6))
plt.plot(epochs_shown, train_loss_shown, label="Train Loss", color="steelblue", linewidth=2)
plt.plot(epochs_shown, val_loss_shown, label="Val Loss", color="firebrick", linewidth=2)

plt.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
            label=f"Best epoch ({best_epoch})")
plt.axvline(x=stopped_epoch, color="red", linestyle="--",
            linewidth=1.5, label=f"Stopped epoch ({stopped_epoch})")

plt.title("Training with Early Stopping (Illustrative)")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step13_early_stopping.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Talking Points:**

- **Why Early Stopping:** It is the simplest form of regularization for iterative models. When validation loss stops improving, additional training only overfits to the training set.
- **Patience parameter:** We wait `patience` epochs without improvement before stopping. Too low = stop too early and underfit. Too high = waste compute and risk overfitting.
- **min_delta:** An improvement must be at least this large to reset the patience counter. Prevents stopping from being fooled by tiny fluctuations.
- **Weight restoration:** We save the model weights at the best epoch and restore them when stopping. The model you deploy is from the best epoch, not the last epoch.
- **Where you will encounter this:** Most deep learning frameworks (TensorFlow/Keras, PyTorch Lightning, etc.) and gradient boosting libraries (XGBoost, LightGBM) provide built-in early stopping callbacks that follow exactly this pattern.

**Exit Criterion Addressed:** *(Preferred) Build Early Stopping callback.*

---

### [PAUSE FOR Q&A] (Final, ~5 min)

Suggested prompts:
- "In what scenario would you prioritize recall over precision?"
- "What is the relationship between the confusion matrix and all the other metrics?"
- "How would SHAP explanations change your model selection process?"

---

## Exit Criteria Checklist

| # | Exit Criterion | Status | Step(s) |
|---|---------------|--------|---------|
| 1 | Implement robust train/val/test splitting | Required | 2 |
| 2 | Describe MSE vs. CrossEntropy mathematically and practically | Required | 4 |
| 3 | Identify bias-variance tradeoff and overfitting from learning curves | Required | 5 |
| 4 | Implement L1 (Lasso) and L2 (Ridge) regularization | Required | 6 |
| 5 | Understand and describe Dropout regularization | Required | 7 |
| 6 | Calculate and interpret Precision, Recall, F1 for imbalanced data | Required | 9 |
| 7 | Generate and analyze AUC-ROC curve and Confusion Matrix | Required | 10, 11 |
| 8 | Implement stratified k-fold cross-validation | Preferred | 3 |
| 9 | Build Early Stopping callback | Preferred | 13 |
| 10 | Generate SHAP values | Preferred | 12 |

---

## Closing Git Branch Activity (Final 5 min)

Instruct Associates to commit their work to a feature branch. Walk through these commands live:

```bash
# Navigate to the project directory
cd fraud-detection-evaluation

# Initialize a Git repo if not already done
git init

# Create and switch to a feature branch
git checkout -b feature/applied-ml-evaluation

# Stage all files
git add .

# Commit with a descriptive message
git commit -m "Add complete model evaluation pipeline for fraud detection

- Stratified train/val/test splits and 5-fold cross-validation
- MSE vs CrossEntropy loss comparison
- Learning curves for bias-variance diagnosis
- L1/L2 regularization with coefficient analysis
- Dropout regularization concepts
- Full metrics suite: precision, recall, F1, AUC-ROC, confusion matrix
- SHAP explainability plots
- Early stopping callback pattern"

# Verify the commit
git log --oneline -1
```

**Instructor Note:** Remind Associates that their assignment branch should branch from this lecture branch so they can build on the code demonstrated today.

---

## Appendix: Environment Setup Reference

If any student needs to set up their environment from scratch:

```bash
python -m venv fraud-eval-env
source fraud-eval-env/bin/activate    # Linux/Mac
# fraud-eval-env\Scripts\activate     # Windows

pip install numpy pandas scikit-learn matplotlib seaborn shap
```

Minimum package versions known to work:
- numpy >= 1.23
- pandas >= 1.5
- scikit-learn >= 1.2
- matplotlib >= 3.6
- seaborn >= 0.12
- shap >= 0.42

---

## Appendix: Key Formulas Quick Reference

| Formula | Expression |
|---------|-----------|
| MSE | `(1/n) * sum((y_true - y_pred)^2)` |
| Binary CrossEntropy | `-(1/n) * sum(y*log(p) + (1-y)*log(1-p))` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 Score | `2 * Precision * Recall / (Precision + Recall)` |
| L2 Penalty | `lambda * sum(w_i^2)` |
| L1 Penalty | `lambda * sum(abs(w_i))` |
| AUC-ROC | Area under the TPR vs. FPR curve |

---

*End of Lecture Guide -- AIML-LEC-AppliedML-Evaluation*
