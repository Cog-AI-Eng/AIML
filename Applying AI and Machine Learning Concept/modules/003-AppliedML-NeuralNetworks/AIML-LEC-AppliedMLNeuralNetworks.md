# AIML-LEC-AppliedML-NeuralNetworks

**Activity Type:** Lecture / Conceptual Live Coding  
**Duration:** 180 minutes (3 hours)  
**Module:** 003 -- Applied ML: Neural Networks  
**Prerequisites:** Students have completed pre-class readings and videos on perceptrons, MLPs, backpropagation, gradient descent, and optimizers.

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Describe the biological inspiration behind the Perceptron model.
2. Explain the vanishing gradient problem and how modern activation functions address it.
3. Define the architecture of a multi-layer perceptron (MLP) using framework-agnostic pseudocode.
4. Explain backpropagation and gradient descent at a conceptual and computational level.
5. Differentiate SGD from Adam and articulate when each is preferable.
6. Analyze width-vs-depth trade-offs conceptually.
7. Describe the purpose and behavior of common learning rate schedulers.

---

## Tech Stack

| Tool | Version / Notes |
|------|-----------------|
| Python | 3.10+ |
| NumPy | latest stable |
| matplotlib | latest stable |
| scikit-learn | latest stable |
| Dataset | `sklearn.datasets.load_digits` (8x8 digit images, 1797 samples) |

> **Note:** This lecture is framework-agnostic. All neural network architecture and training logic is presented as pseudocode and mathematical notation. NumPy and matplotlib are used for data exploration and visualization only.

---

## Preparation Checklist (Instructor)

- [ ] Verify Python 3.10+, numpy, matplotlib, scikit-learn installed in demo environment.
- [ ] Clone the lecture repo; confirm the starter branch `lecture/stage-0-starter` exists.
- [ ] Pre-load `sklearn.datasets.load_digits` to confirm availability.
- [ ] Have backup slides/diagrams ready for biological neuron, computation graph, and optimizer comparison visuals.
- [ ] Test projector / screen-share for live coding.

---

## Scenario

Students will explore **image digit classification** using the sklearn digits dataset (1797 samples of 8x8 grayscale images, 10 classes). Across three stages they will design progressively more complex MLPs in pseudocode, compare optimizers conceptually, and visualize activation functions and learning dynamics -- building a deep understanding of neural network fundamentals that transfers to any framework.

---

## Git Branch Strategy

| Branch | Purpose |
|--------|---------|
| `lecture/stage-0-starter` | Empty scaffold with imports and data loading only |
| `lecture/stage-1-perceptron` | Completed Stage 1 code |
| `lecture/stage-2-mlp` | Completed Stage 2 code |
| `lecture/stage-3-optimizers` | Completed Stage 3 code (final) |

Students should check out `lecture/stage-0-starter` at the start. At the end of each stage the instructor can show the corresponding branch as a reference checkpoint.

---

# STAGE 1 -- Perceptrons and Activation Functions (45 min)

> **Goal:** Connect biological neurons to the Perceptron model, define a single-layer perceptron mathematically, and explore activation functions.

## STEP 1.1 -- Biological Motivation (10 min)

**[PACING: Keep this conceptual; no code yet.]**

- Draw or display a diagram of a biological neuron: dendrites (inputs), soma (summation), axon hillock (threshold / activation), axon terminals (output).
- Map each biological component to the mathematical perceptron:
  - Inputs x_i --> dendrites
  - Weights w_i --> synaptic strengths
  - Weighted sum z = sum(w_i * x_i) + b --> soma integration
  - Activation function f(z) --> axon hillock firing rule
  - Output y = f(z) --> signal along axon
- Emphasize: the perceptron is a *drastic* simplification. Real neurons have timing, plasticity, complex dendritic computation. The analogy is inspirational, not literal.

**Discussion Prompt:** "What biological properties does the perceptron completely ignore?" (timing, inhibition patterns, dendritic nonlinearities, neuromodulation)

---

## STEP 1.2 -- Load and Explore the Dataset (8 min)

**[PACING: Live code. Students follow along.]**

```python
# STEP 1.2 -- Load sklearn digits dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target  # X: (1797, 64), y: (1797,)

print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")

# Visualize a few samples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.suptitle("Sample Digits from sklearn")
plt.tight_layout()
plt.show()

# Normalize and split
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

## STEP 1.3 -- Single-Layer Perceptron (15 min)

**[PACING: Present mathematically and in pseudocode. Explain each piece.]**

### Mathematical Definition

A single-layer perceptron for multi-class classification computes:

```
z = W · x + b
```

Where:
- **x** is the input vector (shape: 64 for our dataset)
- **W** is a weight matrix (shape: 10 × 64 -- one row per class)
- **b** is a bias vector (shape: 10)
- **z** is the vector of raw logits (shape: 10 -- one score per class)

The predicted class is: `ŷ = argmax(z)`

For training, logits are passed through a **softmax** function to get probabilities, then evaluated with **cross-entropy loss**.

### Pseudocode

```
CLASS Perceptron:
    PARAMETERS:
        W: matrix of shape (output_dim, input_dim), initialized randomly
        b: vector of shape (output_dim), initialized to zeros

    FUNCTION forward(x):
        z = W · x + b       # linear transformation
        RETURN z             # raw logits (no hidden activation)

# Instantiate for our digit classification task
model = Perceptron(input_dim=64, output_dim=10)

# Parameter count: 64 * 10 (weights) + 10 (biases) = 650
```

Explain:
- The linear transformation `W · x + b` is the core operation.
- Output is 10 raw logits (one per class). The loss function applies softmax internally.
- Parameter count: 64 × 10 + 10 = 650.
- This is equivalent to multinomial logistic regression -- no hidden layers, so no ability to learn nonlinear decision boundaries.

---

## STEP 1.4 -- Activation Functions Deep Dive (12 min)

**[PACING: Conceptual + visualization code. Critical topic.]**

```python
# STEP 1.4 -- Visualize common activation functions
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
- **Sigmoid:** squashes to (0,1). Gradient approaches 0 at extremes --> **vanishing gradient** when stacking layers. Historically used; rarely used in hidden layers today.
- **Tanh:** squashes to (-1,1). Zero-centered (better than sigmoid) but still saturates.
- **ReLU:** f(z) = max(0, z). No saturation for positive inputs. Cheap to compute. Default choice in most modern networks. Risk: "dying ReLU" (neuron stuck at 0).
- **LeakyReLU:** small slope for negative inputs avoids dying ReLU.

**Vanishing Gradient Explanation:**
- When sigmoid/tanh saturate, gradients are near zero.
- During backpropagation, gradients multiply across layers. If each factor is < 1, the product shrinks exponentially.
- Deep networks with sigmoid activations learn extremely slowly in early layers.
- ReLU has gradient = 1 for positive inputs, preventing this multiplicative decay.

**Q&A Checkpoint (2-3 min):** "Why not just use ReLU everywhere? What could go wrong?" (dying ReLU, negative inputs always zeroed, LeakyReLU/ELU as alternatives)

---

**[BREAK -- 5 min]**

> Encourage students to run the data-loading and activation-function code, experiment with plotting different activation functions, and check out `lecture/stage-1-perceptron` if they fell behind.

---

# STAGE 2 -- MLPs: Backpropagation and Depth vs. Width (45 min)

> **Goal:** Define multi-layer perceptrons, understand backpropagation, and reason about network architecture.

## STEP 2.1 -- Backpropagation Walkthrough (15 min)

**[PACING: Conceptual first, then connect to pseudocode. This is the hardest topic -- go slowly.]**

Present backpropagation in three layers of abstraction:

### Layer 1: The Idea (3 min)
- We have a loss function L that measures how wrong our predictions are.
- We need ∂L/∂w for every weight w to know which direction to adjust it.
- Backpropagation is just the chain rule applied systematically from output to input.

### Layer 2: Chain Rule on a Computation Graph (7 min)

Draw a simple 2-layer network on the board/screen:

```
Input x --> [Linear + ReLU] --> h --> [Linear] --> logits --> CrossEntropyLoss --> L
```

Walk through the chain rule:
- ∂L/∂(logits) -- provided by the loss function
- ∂L/∂W₂ = ∂L/∂(logits) · hᵀ
- ∂L/∂h = W₂ᵀ · ∂L/∂(logits) -- the gradient "flows backward" through the weights
- ∂L/∂(pre_relu) = ∂L/∂h ⊙ ReLU'(pre_relu) -- element-wise; 0 or 1
- ∂L/∂W₁ = ∂L/∂(pre_relu) · xᵀ

Key insight: each layer only needs the gradient from the layer above and its own local derivatives. This is why it is called *back*-propagation -- the gradient signal propagates backward.

### Layer 3: Automatic Differentiation in Practice (5 min)

Modern deep learning frameworks provide **automatic differentiation** (autograd), which records operations on tensors and computes gradients automatically.

```
# Pseudocode: How autograd works conceptually

x = random_vector(64)                      # input
W = random_matrix(64, 10, track_gradients=True)
b = zeros(10, track_gradients=True)

logits = x @ W + b                         # forward pass (recorded by autograd)
loss = cross_entropy_loss(logits, target=3) # scalar loss

# Backward pass: autograd traverses the computation graph in reverse
gradients = compute_gradients(loss)         # returns ∂L/∂W and ∂L/∂b

# Result:
#   gradients[W].shape == (64, 10)
#   gradients[b].shape == (10,)
```

Emphasize:
- The backward pass traverses the computation graph in reverse, applying the chain rule at each node.
- In practice you never compute gradients by hand -- but understanding the mechanism prevents debugging nightmares.
- Every major framework (PyTorch, TensorFlow, JAX) implements autograd. The concept is universal.

---

## STEP 2.2 -- Build an MLP (10 min)

**[PACING: Walk through pseudocode. Emphasize the architecture concepts, not API calls.]**

### Architecture Pseudocode

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
            x = activation(W · x + b)    # linear transform + nonlinearity
        x = W_output · x + b_output      # final layer: raw logits, no activation
        RETURN x

# Example: 2 hidden layers of 128 neurons each
model = MLP(input_dim=64, hidden_dims=[128, 128], output_dim=10)
```

### Parameter Count Breakdown

| Layer | Input Dim | Output Dim | Weights | Biases | Total |
|-------|-----------|------------|---------|--------|-------|
| Hidden 1 | 64 | 128 | 64 × 128 = 8,192 | 128 | 8,320 |
| Hidden 2 | 128 | 128 | 128 × 128 = 16,384 | 128 | 16,512 |
| Output | 128 | 10 | 128 × 10 = 1,280 | 10 | 1,290 |
| **Total** | | | | | **26,122** |

Explain the design:
- `hidden_dims` is a list so we can easily experiment with width and depth.
- Layers are chained: each layer's output feeds into the next layer's input.
- Parameter count grows with width (quadratically between adjacent layers) and with depth (more layers).
- The activation function between layers is what gives the network its ability to model nonlinear relationships. Without it, stacking linear layers would collapse into a single linear transformation.

---

## STEP 2.3 -- Training Loop with Gradient Descent (12 min)

**[PACING: Walk through pseudocode carefully. This is the first complete training loop -- annotate each step.]**

### The Training Loop (Pseudocode)

```
FUNCTION train_model(model, X_train, y_train, X_test, y_test,
                     optimizer, epochs=50, batch_size=64):

    train_losses = []
    test_accuracies = []

    FOR epoch = 1 TO epochs:
        # --- Training phase ---
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

        # --- Evaluation phase ---
        predictions = model.forward(X_test).argmax(axis=1)
        accuracy = mean(predictions == y_test)
        test_accuracies.append(accuracy)

        IF epoch % 10 == 0:
            PRINT "Epoch {epoch} | Loss: {epoch_loss} | Test Acc: {accuracy}"

    RETURN train_losses, test_accuracies
```

Walk through the five critical steps inside the batch loop:
1. **Zero gradients** -- clear accumulated gradients from the previous iteration. Most frameworks accumulate gradients by default.
2. **Forward pass** -- push data through the network to compute predictions.
3. **Compute loss** -- compare predictions to true labels using a loss function (cross-entropy for classification).
4. **Backward pass (backpropagation)** -- compute ∂L/∂w for every weight using the chain rule.
5. **Optimizer step** -- apply the update rule (e.g., w ← w − lr · ∂L/∂w) to adjust weights.

### Mini-batch Gradient Descent

Explain why we use mini-batches rather than the full dataset:
- **Full-batch gradient descent:** Uses all samples to compute one gradient update. Stable but slow and memory-intensive.
- **Stochastic gradient descent (SGD):** Uses a single sample per update. Noisy but fast.
- **Mini-batch gradient descent:** Uses a small batch (e.g., 32-128 samples). Balances noise and stability. This is what "SGD" usually means in practice.

---

## STEP 2.4 -- Width vs. Depth Experiments (8 min)

**[PACING: Conceptual analysis with a summary table. Discuss trade-offs.]**

### Architectures to Compare

| Configuration | Hidden Layers | Parameter Count (approx.) |
|---------------|---------------|--------------------------|
| Wide-Shallow (1×512) | [512] | 64×512 + 512 + 512×10 + 10 = 38,410 |
| Narrow-Deep (4×32) | [32, 32, 32, 32] | ≈ 3,594 |
| Balanced (2×128) | [128, 128] | ≈ 26,122 |
| Deep (4×128) | [128, 128, 128, 128] | ≈ 59,402 |

### Expected Observations

Describe what students would typically see if they trained each configuration:

- **Wide-Shallow:** Learns quickly since one large layer has lots of capacity. But all features are learned at a single level of abstraction.
- **Narrow-Deep:** Fewer parameters but can learn hierarchical representations. May be harder to train (vanishing gradients) and slower to converge.
- **Balanced:** A reasonable middle ground. Often performs well on small-to-medium datasets.
- **Deep:** Highest capacity but highest risk of overfitting on small datasets. More parameters to tune.

### Visualization (Conceptual)

Describe to students what the loss and accuracy curves would look like:
- Plot training loss over epochs for each configuration -- deeper networks may show slower initial descent.
- Plot test accuracy -- very deep networks may overfit (training accuracy high, test accuracy plateaus or drops).

**Exercise:** Have students calculate the parameter count for a custom architecture of their choosing (e.g., [64, 64, 32]) and predict how it might perform relative to the configurations above.

Discussion points:
- Wider networks can represent more functions per layer but add parameters quadratically.
- Deeper networks can learn hierarchical features but are harder to train (vanishing gradients, optimization difficulty).
- For this small dataset, moderate architectures often win; very deep networks may overfit.
- In practice: depth tends to matter more for complex tasks (images, language); width matters for capacity.

**Q&A Checkpoint (2-3 min):** "If you had a fixed parameter budget, would you go wider or deeper? Why might the answer depend on the task?"

---

**[BREAK -- 5 min]**

> Students should review the architecture pseudocode and parameter counts. Check out `lecture/stage-2-mlp` if behind.

---

# STAGE 3 -- Optimizers and Learning Rate Schedules (45 min)

> **Goal:** Compare SGD and Adam, understand momentum and adaptive learning rates, explore learning rate schedulers, and reason about their effects.

## STEP 3.1 -- Gradient Descent Variants (12 min)

**[PACING: Conceptual + math notation. Build intuition before formulas.]**

### Vanilla SGD

Update rule:

```
w ← w − lr · ∂L/∂w
```

Problems: (1) same learning rate for all parameters, (2) oscillation in ravines, (3) sensitive to learning rate choice.

### SGD with Momentum

Maintains a velocity term that accumulates past gradients:

```
v ← momentum · v + ∂L/∂w
w ← w − lr · v
```

- Dampens oscillations, accelerates in consistent gradient directions.
- Think of a ball rolling downhill: momentum carries it through small bumps.
- Typical momentum value: 0.9.

### Adam (Adaptive Moment Estimation)

Tracks both first moment (mean of gradients) and second moment (mean of squared gradients):

```
m ← β₁ · m + (1 − β₁) · ∂L/∂w          # first moment estimate (direction)
v ← β₂ · v + (1 − β₂) · (∂L/∂w)²       # second moment estimate (magnitude)
m̂ = m / (1 − β₁ᵗ)                        # bias correction
v̂ = v / (1 − β₂ᵗ)                        # bias correction
w ← w − lr · m̂ / (√v̂ + ε)               # adaptive update
```

- Effectively gives each parameter its own adaptive learning rate.
- Usually converges faster with less tuning.
- Default go-to optimizer for most deep learning tasks.
- Typical defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

### Optimizer Comparison (Conceptual)

Describe the expected behavior when training the same MLP (e.g., [128, 128]) with different optimizers:

| Optimizer | Expected Behavior |
|-----------|-------------------|
| SGD (lr=0.01) | Slowest convergence; loss decreases steadily but gradually |
| SGD + Momentum (lr=0.01, momentum=0.9) | Noticeably faster than vanilla SGD; smoother loss curve |
| Adam (lr=0.001) | Fast early convergence; reaches good accuracy quickly |
| Adam (lr=0.01) | May overshoot or oscillate if learning rate is too high |

Narrate expected observations:
- Vanilla SGD is slowest to converge.
- Momentum significantly speeds up SGD.
- Adam converges fastest in early epochs.
- Adam with too-high a learning rate may overshoot or oscillate.

---

## STEP 3.2 -- SGD vs. Adam: When to Use Which (8 min)

**[PACING: Discussion-heavy. No code.]**

| Criterion | SGD (+ Momentum) | Adam |
|-----------|-------------------|------|
| Convergence speed | Slower | Faster |
| Hyperparameter sensitivity | High (lr critical) | More forgiving |
| Generalization | Often slightly better final performance | Can overfit or find sharper minima |
| Memory | Lower (no moment buffers) | 2x parameter memory for moments |
| Common use | Large-scale vision (ResNets), when tuning budget available | NLP, GANs, prototyping, default choice |

Rule of thumb: start with Adam for rapid prototyping; switch to SGD+momentum for final performance tuning if generalization matters.

**Discussion Prompt:** "You are training a model for production where every 0.1% accuracy matters and you have time to tune. Which optimizer do you start with? Which do you finish with?"

---

## STEP 3.3 -- Learning Rate Schedulers (15 min)

**[PACING: Conceptual + visualization. New concept for most students.]**

### Why Schedule the Learning Rate?

- A high learning rate helps escape shallow local minima early in training.
- A low learning rate enables fine-grained convergence as we approach a good solution.
- Schedulers automate this transition: start high, end low.

### Common Scheduler Types

**Step Decay:**
```
Every N epochs: lr ← lr × gamma

Example: lr=0.01, step_size=30, gamma=0.1
  Epochs  1-30:  lr = 0.01
  Epochs 31-60:  lr = 0.001
  Epochs 61-90:  lr = 0.0001
```
Simple and predictable. Requires choosing step_size and gamma.

**Cosine Annealing:**
```
lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t / T_max))
```
Smoothly reduces LR following a cosine curve from lr_max down to lr_min. Popular in modern training schedules. No sharp drops.

**Reduce on Plateau:**
```
IF monitored_metric has not improved for `patience` epochs:
    lr ← lr × factor

Example: patience=10, factor=0.5
  Adapts to the training dynamics automatically.
```
Monitors a metric (e.g., validation loss) and reduces LR when progress stalls. Adaptive and requires less manual tuning.

### Visualizing Scheduler Curves

```python
# STEP 3.3 -- Visualize learning rate schedules
epochs = np.arange(100)

# Step decay
lr_step = np.where(epochs < 30, 0.01, np.where(epochs < 60, 0.001, 0.0001))

# Cosine annealing
lr_cosine = 0.0001 + 0.5 * (0.01 - 0.0001) * (1 + np.cos(np.pi * epochs / 100))

# Exponential decay
lr_exp = 0.01 * (0.95 ** epochs)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(epochs, lr_step, linewidth=2)
axes[0].set_title("Step Decay (step=30, gamma=0.1)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Learning Rate")

axes[1].plot(epochs, lr_cosine, linewidth=2, color="orange")
axes[1].set_title("Cosine Annealing (T_max=100)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Learning Rate")

axes[2].plot(epochs, lr_exp, linewidth=2, color="green")
axes[2].set_title("Exponential Decay (gamma=0.95)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")

plt.tight_layout()
plt.show()
```

Key takeaways:
- **Step Decay:** Drops LR by a factor at fixed intervals. Simple, predictable.
- **Cosine Annealing:** Smoothly reduces LR following a cosine curve. Popular in modern training.
- **Reduce on Plateau:** Monitors a metric and reduces LR when it stops improving. Adaptive.
- A well-chosen scheduler can improve final accuracy and training stability.

---

## STEP 3.4 -- Putting It All Together (10 min)

**[PACING: Summarize best practices with a final pseudocode architecture.]**

### Best-Practice Architecture (Pseudocode)

```
# Final model combining best practices

SET random seed for reproducibility

model = MLP(
    input_dim  = 64,
    hidden_dims = [256, 128, 64],   # "funnel" pattern: decreasing width
    output_dim  = 10,
    activation  = ReLU
)

optimizer = Adam(model.parameters, lr=0.001)
scheduler = CosineAnnealing(optimizer, T_max=120)

FOR epoch = 1 TO 120:
    train_losses, test_acc = train_one_epoch(model, optimizer, X_train, y_train, X_test, y_test)
    scheduler.step()

PRINT "Final Test Accuracy:", test_acc
PRINT "Best Test Accuracy:", max(all_test_accs), "at epoch", argmax(all_test_accs)
```

Summarize the design choices:
- **Architecture:** 3 hidden layers with decreasing width (256 → 128 → 64) -- a common "funnel" pattern that progressively compresses representations.
- **Activation:** ReLU for all hidden layers (default, avoids vanishing gradient).
- **Optimizer:** Adam for reliable convergence without extensive hyperparameter tuning.
- **Scheduler:** Cosine annealing to fine-tune in later epochs.
- **Reproducibility:** Fixed random seed.

### What to Expect

For this dataset (1797 samples, 64 features, 10 classes), a well-tuned MLP can reach approximately 97-98% test accuracy. The key factors for success are:
- Appropriate network size (not too large for a small dataset).
- Good optimizer choice and learning rate.
- Standardized input features.

---

**[Q&A / WRAP-UP -- 15 min buffer]**

## Session Recap

| Topic | Key Takeaway |
|-------|-------------|
| Perceptron | Inspired by biological neurons; single linear layer with activation |
| Activation Functions | ReLU is the modern default; sigmoid/tanh suffer from vanishing gradients |
| Backpropagation | Systematic application of chain rule from loss backward through layers |
| Gradient Descent | Iterative weight updates in the direction that reduces loss |
| Width vs. Depth | Width adds capacity per layer; depth enables hierarchical features |
| SGD vs. Adam | Adam is faster to converge; SGD can generalize better with tuning |
| LR Schedulers | Reducing learning rate over time improves convergence and final performance |

## Git Activity

Have students:
1. Check out `lecture/stage-3-optimizers` to see the complete reference code.
2. Create a personal branch `student/<name>/experiments` to try their own architecture/optimizer combinations.

## Additional Resources

- 3Blue1Brown neural network series (visual intuition for backpropagation)
- "Deep Learning" by Goodfellow, Bengio, Courville -- Chapters 6 and 8
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition (lecture notes freely available)
- "Neural Networks and Deep Learning" by Michael Nielsen (free online book)
