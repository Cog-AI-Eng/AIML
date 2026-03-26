# AIML-LEC-AppliedML-NeuralNetworks-and-Evaluation

**Activity Type:** Lecture / Conceptual Live Coding
**Duration:** 195 minutes (~3 hours 15 min)
**Modules:** 003 -- Applied ML: Neural Networks + remaining 002 -- Applied ML: Evaluation
**Prerequisites:** Associates have completed pre-class readings on perceptrons, MLPs, backpropagation, gradient descent, optimizers, regularization, early stopping, classification metrics, AUC-ROC, and SHAP.

---

## Learning Objectives

By the end of this lecture, Associates will be able to:

1. Describe the biological inspiration behind the Perceptron model.
2. Explain the vanishing gradient problem and how modern activation functions address it.
3. Define the architecture of a multi-layer perceptron (MLP) using framework-agnostic pseudocode.
4. Explain backpropagation and gradient descent at a conceptual and computational level.
5. Implement L1 (Lasso), L2 (Ridge), and Dropout regularization and explain when to use each.
6. Differentiate SGD from Adam and articulate when each is preferable.
7. Build an Early Stopping callback and explain patience, min_delta, and weight restoration.
8. Calculate and interpret Precision, Recall, and F1 scores for imbalanced classification.
9. Generate and analyze an AUC-ROC curve and Confusion Matrix.
10. Generate SHAP values and interpret feature contributions for individual predictions.

---

## Tech Stack

| Tool | Version / Notes |
|------|-----------------|
| Python | 3.10+ |
| NumPy | latest stable |
| pandas | latest stable |
| matplotlib | latest stable |
| seaborn | latest stable |
| scikit-learn | latest stable |
| shap | latest stable |
| Dataset (Stages 1-2) | `sklearn.datasets.load_digits` (8x8 digit images, 1797 samples) |
| Dataset (Stages 3-4) | Synthetic fraud detection (10k samples, 95/5 class imbalance) |

> **Note:** This lecture is framework-agnostic for neural network concepts. All NN architecture and training logic is presented as pseudocode and mathematical notation. scikit-learn is used for regularization demos, metrics, and SHAP.

---

## Preparation Checklist (Instructor)

- [ ] Verify Python 3.10+, numpy, pandas, matplotlib, seaborn, scikit-learn, shap installed in demo environment.
- [ ] Clone the lecture repo; confirm the starter branch `lecture/stage-0-starter` exists.
- [ ] Pre-load `sklearn.datasets.load_digits` and test `sklearn.datasets.make_classification` to confirm availability.
- [ ] Have backup slides/diagrams ready for biological neuron, computation graph, optimizer comparison, and confusion matrix visuals.
- [ ] Test projector / screen-share for live coding.

---

## Lecture Timeline Overview

| Time          | Duration | Content                                                    |
|---------------|----------|------------------------------------------------------------|
| 0:00 - 0:35  | 35 min   | **Stage 1** -- Perceptrons and Activation Functions        |
| 0:35 - 1:10  | 35 min   | **Stage 2** -- MLPs, Backpropagation, and Gradient Descent |
| 1:10 - 1:20  | 10 min   | Break                                                      |
| 1:20 - 2:00  | 40 min   | **Stage 3** -- Regularization, Optimizers, Early Stopping  |
| 2:00 - 2:10  | 10 min   | Break                                                      |
| 2:10 - 2:50  | 40 min   | **Stage 4** -- Evaluation Metrics and Explainability       |
| 2:50 - 3:00  | 10 min   | Session Recap, Exit Criteria, Git Activity                 |
| 3:00 - 3:15  | 15 min   | Q&A Buffer                                                 |

---

## Scenario

This lecture spans two domains across four stages:

**Stages 1-2 (Neural Network Fundamentals):** Associates explore image digit classification using the sklearn digits dataset (1797 samples of 8x8 grayscale images, 10 classes). They design progressively more complex MLPs in pseudocode and visualize activation functions -- building a deep understanding of neural network fundamentals that transfers to any framework.

**Stages 3-4 (Regularization, Training Control, and Evaluation):** Associates switch to a fraud detection problem (synthetic, 95/5 class imbalance). This dataset makes regularization, evaluation metrics, and explainability tangible -- accuracy alone is meaningless when 95% of samples belong to one class.

---

## Git Branch Strategy

| Branch | Purpose |
|--------|---------|
| `lecture/stage-0-starter` | Empty scaffold with imports and data loading only |
| `lecture/stage-1-perceptron` | Completed Stage 1 code |
| `lecture/stage-2-mlp` | Completed Stage 2 code |
| `lecture/stage-3-regularization` | Completed Stage 3 code |
| `lecture/stage-4-evaluation` | Completed Stage 4 code (final) |

Associates should check out `lecture/stage-0-starter` at the start. At the end of each stage the instructor can show the corresponding branch as a reference checkpoint.

---

# STAGE 1 -- Perceptrons and Activation Functions (35 min)

> **Goal:** Connect biological neurons to the Perceptron model, define a single-layer perceptron mathematically, and explore activation functions.

## STEP 1.1 -- Biological Motivation (7 min)

**[PACING: Keep this conceptual; no code yet.]**

- Draw or display a diagram of a biological neuron: dendrites (inputs), soma (summation), axon hillock (threshold / activation), axon terminals (output).
- Map each biological component to the mathematical perceptron:
  - Inputs x_i --> dendrites
  - Weights w_i --> synaptic strengths
  - Weighted sum z = sum(w_i * x_i) + b --> soma integration
  - Activation function f(z) --> axon hillock firing rule
  - Output y = f(z) --> signal along axon
- Emphasize: the perceptron is a *drastic* simplification. The analogy is inspirational, not literal.

**Discussion Prompt:** "What biological properties does the perceptron completely ignore?"

---

## STEP 1.2 -- Load and Explore the Dataset (8 min)

**[PACING: Live code. Associates follow along.]**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target  # X: (1797, 64), y: (1797,)

print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.suptitle("Sample Digits from sklearn")
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

Key points to narrate:
- 64 features (8x8 pixels flattened).
- 10 classes (digits 0-9).
- Standardization centers features around 0 with unit variance -- important for gradient-based training.

---

## STEP 1.3 -- Single-Layer Perceptron (8 min)

**[PACING: Present mathematically and in pseudocode.]**

### Mathematical Definition

A single-layer perceptron for multi-class classification computes:

```
z = W . x + b
```

Where:
- **x** is the input vector (shape: 64 for our dataset)
- **W** is a weight matrix (shape: 10 x 64 -- one row per class)
- **b** is a bias vector (shape: 10)
- **z** is the vector of raw logits (shape: 10 -- one score per class)

The predicted class is: `y_hat = argmax(z)`

For training, logits are passed through a **softmax** function to get probabilities, then evaluated with **cross-entropy loss**.

### Pseudocode

```
CLASS Perceptron:
    PARAMETERS:
        W: matrix of shape (output_dim, input_dim), initialized randomly
        b: vector of shape (output_dim), initialized to zeros

    FUNCTION forward(x):
        z = W . x + b       # linear transformation
        RETURN z             # raw logits (no hidden activation)

model = Perceptron(input_dim=64, output_dim=10)
# Parameter count: 64 * 10 + 10 = 650
```

This is equivalent to multinomial logistic regression -- no hidden layers, so no ability to learn nonlinear decision boundaries.

---

## STEP 1.4 -- Activation Functions Deep Dive (12 min)

**[PACING: Conceptual + visualization code. Critical topic.]**

```python
z = np.linspace(-5, 5, 200)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

activations = {
    "Sigmoid": sigmoid(z),
    "Tanh": np.tanh(z),
    "ReLU": relu(z),
    "LeakyReLU (0.01)": leaky_relu(z, 0.01),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, (name, vals) in zip(axes, activations.items()):
    ax.plot(z, vals, linewidth=2)
    ax.set_title(name)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel("f(z)")
plt.tight_layout()
plt.show()
```

Narrate:
- **Sigmoid:** squashes to (0,1). Gradient approaches 0 at extremes --> **vanishing gradient** when stacking layers. Rarely used in hidden layers today.
- **Tanh:** squashes to (-1,1). Zero-centered but still saturates.
- **ReLU:** f(z) = max(0, z). No saturation for positive inputs. Default choice in most modern networks. Risk: "dying ReLU" (neuron stuck at 0).
- **LeakyReLU:** small slope for negative inputs avoids dying ReLU.

**Vanishing Gradient Explanation:**
- When sigmoid/tanh saturate, gradients are near zero.
- During backpropagation, gradients multiply across layers. If each factor is < 1, the product shrinks exponentially.
- ReLU has gradient = 1 for positive inputs, preventing this multiplicative decay.

**Q&A Checkpoint:** "Why not just use ReLU everywhere? What could go wrong?"

---

# STAGE 2 -- MLPs, Backpropagation, and Gradient Descent (35 min)

> **Goal:** Define multi-layer perceptrons, understand backpropagation, and build a complete training loop.

## STEP 2.1 -- Backpropagation Walkthrough (12 min)

**[PACING: Conceptual first, then connect to pseudocode. This is the hardest topic -- go slowly.]**

### Layer 1: The Idea (3 min)
- We have a loss function L that measures how wrong our predictions are.
- We need dL/dw for every weight w to know which direction to adjust it.
- Backpropagation is just the chain rule applied systematically from output to input.

### Layer 2: Chain Rule on a Computation Graph (5 min)

Draw a simple 2-layer network:

```
Input x --> [Linear + ReLU] --> h --> [Linear] --> logits --> CrossEntropyLoss --> L
```

Walk through the chain rule:
- dL/d(logits) -- provided by the loss function
- dL/dW2 = dL/d(logits) . h^T
- dL/dh = W2^T . dL/d(logits)
- dL/d(pre_relu) = dL/dh * ReLU'(pre_relu) -- element-wise; 0 or 1
- dL/dW1 = dL/d(pre_relu) . x^T

Key insight: each layer only needs the gradient from the layer above and its own local derivatives.

### Layer 3: Automatic Differentiation in Practice (4 min)

```
x = random_vector(64)
W = random_matrix(64, 10, track_gradients=True)
b = zeros(10, track_gradients=True)

logits = x @ W + b                         # forward pass (recorded by autograd)
loss = cross_entropy_loss(logits, target=3) # scalar loss

gradients = compute_gradients(loss)         # returns dL/dW and dL/db
# gradients[W].shape == (64, 10)
# gradients[b].shape == (10,)
```

In practice you never compute gradients by hand -- but understanding the mechanism prevents debugging nightmares. Every major framework (PyTorch, TensorFlow, JAX) implements autograd.

---

## STEP 2.2 -- Build an MLP (8 min)

**[PACING: Walk through pseudocode. Emphasize architecture concepts.]**

```
CLASS MLP:
    PARAMETERS:
        layers: list of (weight_matrix, bias_vector) pairs
        activation: chosen activation function (e.g., ReLU)

    CONSTRUCTOR(input_dim, hidden_dims, output_dim, activation="relu"):
        prev_dim = input_dim
        FOR EACH h_dim IN hidden_dims:
            ADD layer: W of shape (prev_dim, h_dim), b of shape (h_dim)
            prev_dim = h_dim
        ADD output layer: W of shape (prev_dim, output_dim), b of shape (output_dim)

    FUNCTION forward(x):
        FOR EACH hidden layer (W, b):
            x = activation(W . x + b)    # linear transform + nonlinearity
        x = W_output . x + b_output      # final layer: raw logits, no activation
        RETURN x

model = MLP(input_dim=64, hidden_dims=[128, 128], output_dim=10)
```

### Parameter Count Breakdown

| Layer | Input Dim | Output Dim | Weights | Biases | Total |
|-------|-----------|------------|---------|--------|-------|
| Hidden 1 | 64 | 128 | 8,192 | 128 | 8,320 |
| Hidden 2 | 128 | 128 | 16,384 | 128 | 16,512 |
| Output | 128 | 10 | 1,280 | 10 | 1,290 |
| **Total** | | | | | **26,122** |

The activation function between layers is what gives the network its ability to model nonlinear relationships. Without it, stacking linear layers would collapse into a single linear transformation.

### Width vs. Depth: Quick Reference

| Configuration | Hidden Layers | Approx. Params | Trade-off |
|---------------|---------------|-----------------|-----------|
| Wide-Shallow (1x512) | [512] | 38,410 | Learns quickly; all features at one abstraction level |
| Narrow-Deep (4x32) | [32, 32, 32, 32] | 3,594 | Hierarchical representations; harder to train |
| Balanced (2x128) | [128, 128] | 26,122 | Middle ground; good for small-medium datasets |
| Funnel (256-128-64) | [256, 128, 64] | ~42,000 | Progressively compresses representations |

Wider networks add capacity per layer (parameters grow quadratically between adjacent layers). Deeper networks learn hierarchical features but are harder to train (vanishing gradients). For small datasets, moderate architectures often win.

---

## STEP 2.3 -- Training Loop with Gradient Descent (10 min)

**[PACING: Walk through pseudocode carefully. This is the first complete training loop.]**

```
FUNCTION train_model(model, X_train, y_train, X_test, y_test,
                     optimizer, epochs=50, batch_size=64):

    train_losses = []
    test_accuracies = []

    FOR epoch = 1 TO epochs:
        epoch_loss = 0.0
        SHUFFLE training data
        SPLIT (X_train, y_train) INTO mini-batches of size batch_size

        FOR EACH (X_batch, y_batch) IN mini-batches:
            1. ZERO GRADIENTS       -- clear accumulated gradients from previous step
            2. logits = model.forward(X_batch)              -- forward pass
            3. loss = cross_entropy(logits, y_batch)        -- compute loss
            4. gradients = backpropagate(loss)               -- backward pass
            5. optimizer.update_weights(gradients)           -- apply gradient descent

            epoch_loss += loss * batch_size

        epoch_loss /= len(y_train)
        train_losses.append(epoch_loss)

        predictions = model.forward(X_test).argmax(axis=1)
        accuracy = mean(predictions == y_test)
        test_accuracies.append(accuracy)

        IF epoch % 10 == 0:
            PRINT "Epoch {epoch} | Loss: {epoch_loss} | Test Acc: {accuracy}"

    RETURN train_losses, test_accuracies
```

Walk through the five critical steps inside the batch loop:
1. **Zero gradients** -- most frameworks accumulate gradients by default.
2. **Forward pass** -- push data through the network to compute predictions.
3. **Compute loss** -- compare predictions to true labels (cross-entropy for classification).
4. **Backward pass** -- compute dL/dw for every weight using the chain rule.
5. **Optimizer step** -- apply the update rule (e.g., w <- w - lr * dL/dw).

**Mini-batch Gradient Descent:** Full-batch is stable but slow. Single-sample (stochastic) is fast but noisy. Mini-batch (32-128 samples) balances noise and stability -- this is what "SGD" usually means in practice.

**Q&A Checkpoint:** "What happens if you forget step 1 (zero gradients)?"

---

**[BREAK -- 10 min]**

> Associates should review the MLP pseudocode and training loop. Check out `lecture/stage-2-mlp` if behind.

---

# STAGE 3 -- Regularization, Optimizers, and Early Stopping (40 min)

> **Goal:** Address overfitting with regularization techniques, compare optimizers, and implement early stopping. We switch to a fraud detection dataset where class imbalance makes these problems visible.

## STEP 3.1 -- Introduce the Fraud Detection Dataset (5 min)

**[PACING: Live code. Quick setup for the rest of the lecture.]**

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

feature_names = [
    "txn_amount", "txn_hour", "merchant_risk", "distance_home",
    "avg_txn_30d", "txn_frequency", "card_age_days", "credit_util",
    "online_ratio", "foreign_txn", "amt_vs_avg", "velocity_1h",
    "prev_declines", "device_score", "geo_anomaly", "time_since_last",
    "weekend_flag", "high_risk_mcc", "ip_change_freq", "acct_balance",
]
df = pd.DataFrame(X, columns=feature_names)
df["is_fraud"] = y

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['is_fraud'].value_counts(normalize=True)}")

X_all = df.drop("is_fraud", axis=1).values
y_all = df["is_fraud"].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print(f"\nStratified split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Val fraud rate:   {y_val.mean():.4f}")
print(f"Test fraud rate:  {y_test.mean():.4f}")
```

**Talking Point:** "If we predicted every transaction as legitimate, we would get ~95% accuracy. This is why accuracy alone is worthless here -- and why we need everything we are about to cover."

---

## STEP 3.2 -- L1 (Lasso) and L2 (Ridge) Regularization (5 min)

**[PACING: Live code the models, narrate the coefficient output.]**

```python
from sklearn.linear_model import LogisticRegression

ridge_model = LogisticRegression(
    penalty="l2", C=1.0, max_iter=1000, solver="lbfgs", random_state=42
)
ridge_model.fit(X_train, y_train)

lasso_model = LogisticRegression(
    penalty="l1", C=1.0, max_iter=1000, solver="saga", random_state=42
)
lasso_model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Ridge (L2)": ridge_model.coef_[0],
    "Lasso (L1)": lasso_model.coef_[0]
})
coef_df["L1_is_zero"] = np.abs(coef_df["Lasso (L1)"]) < 1e-6
print(coef_df.to_string(index=False))
print(f"\nFeatures zeroed out by Lasso: {coef_df['L1_is_zero'].sum()} / {len(feature_names)}")
```

**Key Points:**

- **L2 (Ridge)** adds `lambda * sum(w^2)` to the loss. Shrinks all coefficients toward zero but rarely makes them exactly zero. "Turn down the volume on all features."
- **L1 (Lasso)** adds `lambda * sum(|w|)` to the loss. Drives some coefficients to exactly zero -- automatic feature selection. "Mute irrelevant features entirely."
- **`C` parameter:** In sklearn, `C = 1 / lambda`. Smaller C = stronger regularization.
- We generated 20 features but only 12 are informative. Lasso should zero out some of the redundant and noise features.

---

## STEP 3.3 -- Dropout Regularization (5 min)

**[PACING: Pseudocode walkthrough. Focus on the concept, not framework syntax.]**

### Architecture with Dropout

```
PSEUDOCODE -- Neural Network with Dropout

Architecture:
    Input Layer:    20 features
    Hidden Layer 1: 64 neurons, activation = ReLU
    Dropout Layer:  drop rate = 0.3
    Hidden Layer 2: 32 neurons, activation = ReLU
    Dropout Layer:  drop rate = 0.3
    Output Layer:   1 neuron,  activation = Sigmoid
```

### Training vs. Inference Behavior

```
PSEUDOCODE -- Training with Dropout

FOR each epoch in 1..50:
    SET model to TRAINING mode
    # Dropout ACTIVE: randomly zero out 30% of neuron outputs per mini-batch.
    # Each batch sees a different random subset of active neurons.

    FOR each mini-batch (X_batch, y_batch):
        predictions = forward_pass(X_batch)
        loss = binary_cross_entropy(predictions, y_batch)
        gradients = compute_gradients(loss)
        update_weights(gradients, learning_rate=0.001)

    SET model to EVALUATION mode
    # Dropout INACTIVE: all neurons active, outputs scaled by (1 - drop_rate).

    val_predictions = forward_pass(X_val)
    val_loss = binary_cross_entropy(val_predictions, y_val)
```

**Key Points:**

- Dropout forces the network to build redundancy -- no single neuron can dominate.
- It approximates training an ensemble of sub-networks. At inference, we average their predictions.
- Forgetting to switch from training mode to evaluation mode is a common bug in any framework.

### Expected Effect

| Model | Training Loss | Validation Loss | Gap |
|-------|--------------|-----------------|-----|
| Without Dropout | Decreases to near zero | Decreases then increases | Large (overfitting) |
| With Dropout (0.3) | Decreases more slowly | Stays close to training loss | Small (better generalization) |

---

## STEP 3.4 -- Optimizers: SGD vs. Adam (12 min)

**[PACING: Conceptual + math notation. Build intuition before formulas.]**

### Vanilla SGD

```
w <- w - lr * dL/dw
```

Problems: same learning rate for all parameters, oscillation in ravines, sensitive to learning rate choice.

### SGD with Momentum

```
v <- momentum * v + dL/dw
w <- w - lr * v
```

Dampens oscillations, accelerates in consistent gradient directions. Think of a ball rolling downhill: momentum carries it through small bumps. Typical momentum value: 0.9.

### Adam (Adaptive Moment Estimation)

```
m <- B1 * m + (1 - B1) * dL/dw          # first moment (direction)
v <- B2 * v + (1 - B2) * (dL/dw)^2      # second moment (magnitude)
m_hat = m / (1 - B1^t)                   # bias correction
v_hat = v / (1 - B2^t)                   # bias correction
w <- w - lr * m_hat / (sqrt(v_hat) + e)  # adaptive update
```

Effectively gives each parameter its own adaptive learning rate. Usually converges faster with less tuning. Default go-to optimizer for most deep learning tasks. Typical defaults: B1 = 0.9, B2 = 0.999, e = 1e-8.

### When to Use Which

| Criterion | SGD (+ Momentum) | Adam |
|-----------|-------------------|------|
| Convergence speed | Slower | Faster |
| Hyperparameter sensitivity | High (lr critical) | More forgiving |
| Generalization | Often slightly better final performance | Can overfit or find sharper minima |
| Memory | Lower (no moment buffers) | 2x parameter memory for moments |
| Common use | Large-scale vision (ResNets), when tuning budget available | NLP, GANs, prototyping, default choice |

**Rule of thumb:** Start with Adam for rapid prototyping; switch to SGD+momentum for final performance tuning if generalization matters.

> **Note:** Learning rate schedulers (step decay, cosine annealing, reduce-on-plateau) are covered in the concept thread readings and can be explored in the notebook. They automate the transition from high to low learning rates over the course of training.

**Discussion Prompt:** "You are training a model for production where every 0.1% accuracy matters. Which optimizer do you start with? Which do you finish with?"

---

## STEP 3.5 -- Early Stopping (8 min)

**[PACING: Pseudocode walkthrough + illustrative plot.]**

### Early Stopping Logic

```
CLASS EarlyStopping:
    PARAMETERS:
        patience        = 5
        min_delta       = 0.001
        restore_best    = True

    STATE:
        best_loss       = infinity
        wait_counter    = 0
        best_weights    = None

    METHOD check(current_val_loss, model, epoch):
        IF current_val_loss < best_loss - min_delta:
            best_loss = current_val_loss
            wait_counter = 0
            IF restore_best:
                best_weights = copy(model.weights)
        ELSE:
            wait_counter += 1

        IF wait_counter >= patience:
            IF restore_best AND best_weights is not None:
                model.weights = best_weights
            RETURN True   # stop training

        RETURN False      # continue training
```

### Training Loop with Early Stopping

```
early_stopper = EarlyStopping(patience=7, min_delta=0.001)
max_epochs = 200

FOR epoch in 1..max_epochs:
    # ... standard training loop ...
    val_loss = evaluate(model, X_val, y_val)

    IF early_stopper.check(val_loss, model, epoch) == True:
        PRINT "Early stopping at epoch {epoch}"
        PRINT "Weights restored from best epoch."
        BREAK
```

### Illustrative Plot

```python
import matplotlib.pyplot as plt

np.random.seed(42)

n_epochs = 50
best_epoch = 20
stopped_epoch = 30

epochs = np.arange(1, n_epochs + 1)

train_loss = 0.7 * np.exp(-0.15 * epochs) + 0.03 + np.random.normal(0, 0.004, n_epochs)

val_decay = 0.65 * np.exp(-0.12 * epochs)
val_overfit = 0.004 * np.maximum(epochs - best_epoch, 0) ** 1.5
val_loss = val_decay + 0.08 + val_overfit + np.random.normal(0, 0.006, n_epochs)

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
plt.title("Training with Early Stopping")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Key Points:**

- **Patience:** How many epochs to wait without improvement. Too low = stop too early. Too high = waste compute and risk overfitting.
- **min_delta:** An improvement must be at least this large to reset patience. Prevents stopping from being fooled by tiny fluctuations.
- **Weight restoration:** The deployed model comes from the best epoch, not the last epoch.
- Every major framework and gradient boosting library provides built-in early stopping callbacks that follow exactly this pattern.

---

**[BREAK -- 10 min]**

> Associates should review the regularization and optimizer notes. Check out `lecture/stage-3-regularization` if behind.

---

# STAGE 4 -- Evaluation Metrics and Explainability (40 min)

> **Goal:** Train a final model on the fraud dataset and evaluate it with precision, recall, F1, AUC-ROC, confusion matrix, and SHAP. We now have a trained model with proper regularization and early stopping -- the question is: how do we know if it is actually good?

## STEP 4.1 -- Train the Evaluation Model (3 min)

**[PACING: Block-update. Straightforward sklearn fit.]**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

eval_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight={0: 1, 1: 8},
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

**Talking Point:** `class_weight={0: 1, 1: 8}` tells the model to penalize misclassifying a fraud case 8x more than a legitimate one. This is a deliberate middle ground -- `"balanced"` would auto-scale to roughly 19:1 for this dataset, which catches more fraud but generates a flood of false alarms that tanks precision. Too low a weight (e.g., 3:1) and the model barely flags any fraud at all. Tuning this weight is itself a form of the precision-recall trade-off -- the exact number depends on how costly each type of error is to the business.

---

## STEP 4.2 -- Precision, Recall, and F1 (12 min)

**[PACING: Line-by-line. This is the conceptual heart of the metrics section.]**

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

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

**Write these formulas on the board:**

- **Precision = TP / (TP + FP):** "Of all transactions we FLAGGED as fraud, what fraction were actually fraud?" High precision = few false alarms.
- **Recall = TP / (TP + FN):** "Of all transactions that WERE fraud, what fraction did we catch?" High recall = we miss few frauds.
- **F1 = 2 * (Precision * Recall) / (Precision + Recall):** Harmonic mean that penalizes extreme imbalances between precision and recall.

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

print("=== Manual Calculation ===")
print(f"  True Positives  (caught fraud):       {tp}")
print(f"  False Positives (false alarms):        {fp}")
print(f"  True Negatives  (correct legitimate):  {tn}")
print(f"  False Negatives (missed fraud):         {fn}")
print()
manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
manual_f1 = (2 * manual_precision * manual_recall /
             (manual_precision + manual_recall)
             if (manual_precision + manual_recall) > 0 else 0)
print(f"  Manual Precision: {manual_precision:.4f}")
print(f"  Manual Recall:    {manual_recall:.4f}")
print(f"  Manual F1:        {manual_f1:.4f}")
```

**Fraud context:** A system with high precision but low recall catches fraud when it flags something, but misses many actual frauds. A system with high recall but low precision catches most frauds but also flags many legitimate transactions. The business decides the tradeoff.

---

## STEP 4.3 -- Confusion Matrix and AUC-ROC (13 min)

**[PACING: Block-update for plots, narrate as they render.]**

### Confusion Matrix

```python
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
plt.show()
```

Walk through each quadrant: top-left (TN) should be largest. Bottom-right (TP) shows caught frauds. Top-right (FP) shows false alarms. Bottom-left (FN) shows missed frauds -- the most dangerous cell in fraud detection.

### AUC-ROC Curve

```python
from sklearn.metrics import roc_auc_score, RocCurveDisplay

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
plt.show()
```

**Key Points:**
- ROC plots True Positive Rate (Recall) vs. False Positive Rate at every threshold. AUC = 1.0 is perfect; AUC = 0.5 is random.
- Unlike precision/recall/F1 which depend on a specific threshold, AUC evaluates across ALL thresholds -- useful for model comparison.
- A high AUC means the model generally assigns higher probabilities to actual fraud cases regardless of cutoff.

### Precision-Recall Curve

```python
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay

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
plt.show()

print(f"Average Precision Score: {ap_score:.4f}")
```

**Talking Point:** For highly imbalanced datasets, the PR curve is often more informative than ROC. ROC can look optimistic because the large number of true negatives inflates the FPR denominator.

---

## STEP 4.4 -- SHAP Values for Model Explainability (10 min)

**[PACING: Block-update. SHAP computation may take a moment.]**

```python
import shap

explainer = shap.LinearExplainer(
    eval_pipeline.named_steps["classifier"],
    eval_pipeline.named_steps["scaler"].transform(X_train),
    feature_names=feature_names
)

X_test_scaled = eval_pipeline.named_steps["scaler"].transform(X_test)
shap_values = explainer.shap_values(X_test_scaled)

print(f"SHAP values shape: {shap_values.shape}")
```

### Summary Plot -- Global Feature Importance

```python
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot -- Feature Impact on Fraud Prediction")
plt.tight_layout()
plt.show()
```

### Bar Plot -- Feature Importance Ranking

```python
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("Mean |SHAP Value| -- Feature Importance Ranking")
plt.tight_layout()
plt.show()
```

**Key Points:**

- SHAP assigns each feature a contribution value for each prediction. Positive SHAP pushes toward fraud; negative pushes toward legitimate.
- The summary plot shows one dot per test sample per feature. Color encodes feature value (red = high, blue = low). Features sorted by overall importance.
- A compliance officer can use these plots to understand WHY the model flagged a specific transaction -- a potential regulatory requirement.

---

## Session Recap

| # | Topic | Key Takeaway |
|---|-------|-------------|
| 1 | Perceptrons & Activation | Inspired by biological neurons; ReLU is the modern default |
| 2 | MLPs: Width vs. Depth | Width adds capacity per layer; depth enables hierarchical features |
| 3 | Backpropagation | Systematic chain rule from loss backward through layers |
| 4 | Regularization (L1, L2, Dropout) | L2 shrinks, L1 zeroes, Dropout builds redundancy |
| 5 | Optimizers (SGD vs. Adam) | Adam converges faster; SGD can generalize better with tuning |
| 6 | Early Stopping | Monitor val loss; restore best weights when patience exhausted |
| 7 | Precision, Recall, F1 | Precision = flagged correctly; Recall = caught; F1 = harmonic mean |
| 8 | AUC-ROC & Confusion Matrix | Threshold-independent model comparison; quadrant analysis |
| 9 | SHAP Explainability | Per-feature, per-prediction contribution scores |

---

## Exit Criteria Checklist

| # | Exit Criterion | Status | Step(s) |
|---|---------------|--------|---------|
| 1 | Describe the biological inspiration behind the Perceptron | Required | 1.1 |
| 2 | Explain vanishing gradient and how modern activations address it | Required | 1.4 |
| 3 | Define MLP architecture using framework-agnostic pseudocode | Required | 2.2 |
| 4 | Explain backpropagation and gradient descent conceptually and computationally | Required | 2.1, 2.3 |
| 5 | Implement L1, L2, and Dropout regularization | Required | 3.2, 3.3 |
| 6 | Differentiate SGD from Adam; articulate when each is preferable | Required | 3.4 |
| 7 | Build Early Stopping callback with patience, min_delta, weight restoration | Preferred | 3.5 |
| 8 | Calculate and interpret Precision, Recall, F1 for imbalanced classification | Required | 4.2 |
| 9 | Generate and analyze AUC-ROC curve and Confusion Matrix | Required | 4.3 |
| 10 | Generate SHAP values and interpret feature contributions | Preferred | 4.4 |

---

## Git Activity (Final 5 min)

```bash
cd neural-networks-evaluation

git init

git checkout -b feature/neural-networks-evaluation

git add .

git commit -m "Add combined neural networks and evaluation lecture

- Perceptrons, activation functions, vanishing gradient
- MLP architecture, backpropagation, training loop
- L1/L2/Dropout regularization on fraud dataset
- SGD vs Adam optimizer comparison
- Early stopping callback pattern
- Precision, recall, F1 for imbalanced classification
- AUC-ROC, confusion matrix, precision-recall curves
- SHAP explainability plots"

git log --oneline -1
```

## Additional Resources

- 3Blue1Brown neural network series (visual intuition for backpropagation)
- "Deep Learning" by Goodfellow, Bengio, Courville -- Chapters 6 and 8
- Stanford CS231n lecture notes (freely available)
- SHAP documentation: https://shap.readthedocs.io

## Appendix: Key Formulas Quick Reference

| Formula | Expression |
|---------|-----------|
| Perceptron | `z = W . x + b` |
| ReLU | `f(z) = max(0, z)` |
| Cross-Entropy | `-(1/n) * sum(y*log(p) + (1-y)*log(1-p))` |
| L2 Penalty | `lambda * sum(w_i^2)` |
| L1 Penalty | `lambda * sum(abs(w_i))` |
| SGD Update | `w <- w - lr * dL/dw` |
| Adam Update | `w <- w - lr * m_hat / (sqrt(v_hat) + e)` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 Score | `2 * Precision * Recall / (Precision + Recall)` |
| AUC-ROC | Area under the TPR vs. FPR curve |

---

*End of Lecture Guide -- AIML-LEC-AppliedML-NeuralNetworks-and-Evaluation*
