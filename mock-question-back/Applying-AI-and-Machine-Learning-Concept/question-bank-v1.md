# Interview Questions: Applying AI and Machine Learning Concepts

## Beginner (Foundational)

### Q1: What is the difference between supervised, unsupervised, and reinforcement learning?
**Keywords:** Labels, Clustering, Rewards, Agent, Environment
<details>
<summary>Click to Reveal Answer</summary>

| Paradigm | Labels | Goal | Examples |
|----------|--------|------|----------|
| **Supervised** | Yes (input-output pairs) | Learn mapping from features to targets | Classification, regression |
| **Unsupervised** | No | Find structure, density, or groupings | Clustering, dimensionality reduction |
| **Reinforcement** | Reward signals (not fixed labels) | Learn a policy to maximize cumulative reward | Games, robotics, recommendation tuning |

Choosing the paradigm depends on whether you have labeled data and whether decisions unfold over time with delayed feedback.
</details>

---

### Q2: What is overfitting and how do you detect it?
**Keywords:** Training Error, Validation Error, Generalization, Memorization
<details>
<summary>Click to Reveal Answer</summary>

**Overfitting** means the model fits noise and idiosyncrasies in the training set instead of the underlying pattern, so it performs well on training data but poorly on unseen data.

**Signals:**
- Large gap between low training error and high validation/test error
- Performance degrades as you add capacity without more data or regularization
- High variance across random train/validation splits

**Mitigations:** more data, simpler model, regularization (L1/L2, dropout), early stopping, cross-validation, feature selection.
</details>

---

### Q3: Explain precision, recall, and F1 score in classification.
**Keywords:** True Positives, False Positives, False Negatives, Harmonic Mean
<details>
<summary>Click to Reveal Answer</summary>

For a positive class:

- **Precision** = TP / (TP + FP) — Of predicted positives, how many were correct?
- **Recall** = TP / (TP + FN) — Of actual positives, how many did we find?

**F1** = 2 * (precision * recall) / (precision + recall) — Harmonic mean; penalizes imbalanced precision/recall.

Use precision when false positives are costly; recall when false negatives are costly (e.g., fraud, disease screening).
</details>

---

### Q4: What is a confusion matrix?
**Keywords:** TP, TN, FP, FN, Multiclass
<details>
<summary>Click to Reveal Answer</summary>

A **confusion matrix** tabulates predicted vs. actual class labels. For binary classification:

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

For multiclass, it is a square matrix with one row per true class and one column per predicted class. It supports deriving accuracy, per-class metrics, and error analysis (which classes are confused).
</details>

---

### Q5: What is gradient descent and what is the role of the learning rate?
**Keywords:** Loss, Weights, Update Step, Convergence
<details>
<summary>Click to Reveal Answer</summary>

**Gradient descent** iteratively adjusts model parameters by moving opposite to the gradient of the loss with respect to those parameters, to reduce training loss.

**Learning rate** scales each update. Too large: unstable training, divergence, oscillation. Too small: very slow convergence, risk of getting stuck in poor local minima (especially in non-convex settings). Common practice: schedule decay, warmup, or adaptive optimizers (Adam, etc.).
</details>

---

### Q6: What is the purpose of an activation function in a neural network?
**Keywords:** Nonlinearity, Universal Approximation, Saturation
<details>
<summary>Click to Reveal Answer</summary>

Stacked linear layers without activation are equivalent to a single linear layer. **Nonlinear activations** (ReLU, sigmoid, tanh, GELU) let the network represent complex, curved decision boundaries.

**ReLU:** fast, sparse activations; can suffer from dead neurons if inputs are always negative. **Sigmoid/tanh:** bounded, smooth; sigmoid is common for probabilities in binary outputs. Choice affects gradients and training stability.
</details>

---

### Q7: What is backpropagation in one sentence, conceptually?
**Keywords:** Chain Rule, Computational Graph, Gradients
<details>
<summary>Click to Reveal Answer</summary>

**Backpropagation** computes gradients of the loss with respect to each parameter by applying the chain rule through the network’s computational graph, efficiently from output layer back to input layer.

Modern frameworks implement this via automatic differentiation; you define the forward pass and the library computes gradients for training.
</details>

---

### Q8: What is a convolutional layer designed to exploit in images?
**Keywords:** Local Patterns, Translation, Parameter Sharing
<details>
<summary>Click to Reveal Answer</summary>

**Convolutional layers** exploit **local spatial structure** (edges, textures, parts) using small filters that slide over the input. **Parameter sharing** applies the same filter everywhere, reducing parameters and encoding **translation equivariance** (shifted patterns produce shifted feature responses).

Pooling and deeper stacks build hierarchical features from low-level edges to high-level objects.
</details>

---

### Q9: Why are LSTM or GRU units used in sequence modeling?
**Keywords:** Vanishing Gradient, Long Dependencies, Gates
<details>
<summary>Click to Reveal Answer</summary>

Plain RNNs struggle with long-range dependencies due to **vanishing/exploding gradients** across many time steps. **LSTM** and **GRU** use **gating** (forget, input, output / update, reset, etc.) to regulate what is kept or discarded in the hidden state, allowing more stable propagation of information over longer sequences.

They are widely used when order matters and context must be retained (text, time series, speech), though Transformers often replace them in NLP at scale.
</details>

---

### Q10: In a Transformer, what do self-attention and multi-head attention accomplish?
**Keywords:** Query, Key, Value, Dependencies, Subspaces
<details>
<summary>Click to Reveal Answer</summary>

**Self-attention** lets each position in a sequence attend to all other positions, producing weighted combinations of **values** based on similarity of **queries** and **keys** — modeling long-range dependencies without recurrence.

**Multi-head attention** runs several attention mechanisms in parallel (different learned projections), so the model can capture different relationship types (syntax, coreference, etc.) in separate subspaces. Outputs are concatenated and projected.
</details>

---

## Intermediate (Application)

### Q11: You have imbalanced classes (95% negative, 5% positive). What techniques can you use besides changing the algorithm?
**Hint:** Think data, metrics, and training objective.
<details>
<summary>Click to Reveal Answer</summary>

- **Resampling:** oversample minority, undersample majority, or SMOTE-style synthetic minority examples (with care to avoid leakage).
- **Class weights:** higher loss weight on minority class during training.
- **Threshold tuning:** optimize decision threshold on validation PR/ROC curves rather than default 0.5.
- **Metrics:** use precision-recall, F1, ROC-AUC or PR-AUC instead of accuracy alone.
- **Stratified splits** for train/val/test to preserve class ratios.

Always validate on a holdout that reflects the real deployment distribution when possible.
</details>

---

### Q12: What is k-fold cross-validation and when is it preferred over a single train/validation split?
**Keywords:** Folds, Variance, Small Data
<details>
<summary>Click to Reveal Answer</summary>

**K-fold cross-validation** divides data into k partitions; each fold serves once as validation while the other k-1 folds train the model. Performance is averaged across folds.

**Preferred when:** labeled data is limited (reduces variance of a single split), you need a more stable estimate of generalization, or you are comparing many hyperparameter settings. **Cost:** k times more training.

For time series, use **time-series CV** (no random shuffle; train on past, validate on future folds) to avoid leakage.
</details>

---

### Q13: Explain the bias-variance tradeoff in the context of model complexity.
**Keywords:** Underfitting, Overfitting, Irreducible Error
<details>
<summary>Click to Reveal Answer</summary>

**Bias** is error from overly simplistic assumptions (underfitting). **Variance** is error from sensitivity to small fluctuations in the training set (overfitting). Total error decomposes conceptually into bias, variance, and irreducible noise.

- **High bias, low variance:** simple model; misses patterns.
- **Low bias, high variance:** complex model; fits training noise.

Goal: balance complexity so validation error is minimized. Regularization and ensemble methods often reduce variance; more features or deeper models can reduce bias at the risk of variance.
</details>

---

### Q14: How does dropout regularize neural networks?
**Keywords:** Random Units, Ensemble, Inference Scaling
<details>
<summary>Click to Reveal Answer</summary>

During training, **dropout** randomly sets a fraction of activations to zero each forward pass, forcing the network not to rely on specific neurons (co-adaptation). This acts like training an implicit **ensemble** of thinned subnetworks.

At inference, either **scale** activations by keep probability or use **inverted dropout** (scale during training) so inference is a standard forward pass without randomness.
</details>

---

### Q15: What is the difference between encoder-decoder architectures and plain sequence-to-sequence models for tasks like translation?
**Keywords:** Context Vector, Attention, Bottleneck
<details>
<summary>Click to Reveal Answer</summary>

Classic seq2seq uses an **encoder** RNN to compress the source into a **fixed-size context vector** and a **decoder** RNN to generate the target. That bottleneck limits long sentences.

**Encoder-decoder with attention** lets the decoder attend to all encoder hidden states at each step, focusing on relevant source tokens — addressing the bottleneck. **Transformers** replace RNNs with self-attention in encoder and decoder stacks for parallelization and long-range modeling.
</details>

---

### Q16: What is transfer learning in deep learning, and what is fine-tuning?
**Keywords:** Pretrained Weights, Frozen Layers, Domain Shift
<details>
<summary>Click to Reveal Answer</summary>

**Transfer learning** reuses a model trained on a large source task (e.g., ImageNet, web text) as initialization for a target task with less data.

**Fine-tuning** updates some or all layers on the target data with a typically smaller learning rate. Common patterns: freeze early layers (generic features), train only the head; or unfreeze progressively / use discriminative learning rates across layers.

Effective when target data is limited but related; less effective under strong domain shift without adaptation.
</details>

---

### Q17: What is positional encoding in Transformers and why is it needed?
**Keywords:** Permutation Invariant, Order, Sinusoidal, Learned
<details>
<summary>Click to Reveal Answer</summary>

Self-attention alone is **permutation invariant** over positions; it does not know token order. **Positional encodings** are added to input embeddings so the model can distinguish “cat sat mat” from “mat sat cat.”

Original Transformer used fixed **sinusoidal** encodings; many models use **learned** positional embeddings or **relative** position schemes (e.g., RoPE in some LLMs). Without position information, sequence order is lost.
</details>

---

## Advanced (Deep Dive)

### Q18: Compare batch norm, layer norm, and where they are typically applied in CNNs vs. Transformers.
<details>
<summary>Click to Reveal Answer</summary>

- **Batch normalization:** normalizes activations across the **batch** dimension (and spatial dims for conv). Strong dependence on batch size; common in CNNs; can behave poorly with very small batches.

- **Layer normalization:** normalizes across **features** for each sample, independent of batch size. Stable for variable batch sizes and standard in **Transformer** blocks (after attention and FFN sublayers).

Both stabilize training and can allow higher learning rates; normalization placement (pre vs. post) varies by architecture (Pre-LN vs. Post-LN in Transformers).
</details>

---

### Q19: Explain the attention mechanism’s scaled dot-product formula and why scaling is used.
**Keywords:** Softmax, Dot Product, Dimension
<details>
<summary>Click to Reveal Answer</summary>

Attention weights are often computed as scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

**Scaling by sqrt(d_k)** (dimension of keys/queries) prevents dot products from growing large in magnitude as d_k increases, which would push softmax into very small gradients (saturated regions). Scaling keeps variance of the dot products more stable.

**Multi-head** attention splits d_model into multiple heads with separate learned projections for Q, K, and V per head.
</details>

---

### Q20: What are residual connections and why do they help very deep networks?
**Keywords:** Skip Connections, Identity Mapping, Vanishing Gradient
<details>
<summary>Click to Reveal Answer</summary>

A **residual block** adds the input of a layer (or stack) to its output: \(y = F(x) + x\). The network learns a **residual** mapping \(F\) relative to identity.

Benefits: eases optimization by providing **gradient highways** (identity path), allows training much deeper networks (ResNet, Transformer sublayers). If optimal mapping is close to identity, learning small perturbations is easier than learning full transformations from scratch.
</details>

---

### Q21: How would you approach debugging poor test performance when training loss decreases but validation loss diverges?
<details>
<summary>Click to Reveal Answer</summary>

Systematically check:

1. **Overfitting:** add regularization, dropout, data augmentation, simpler model, early stopping.
2. **Data issues:** leakage between train/val, duplicate near-duplicates across splits, incorrect stratification.
3. **Distribution shift:** train/val/test from different sources or time periods.
4. **Preprocessing:** fit scalers only on training data; same pipeline at inference.
5. **Optimization:** learning rate too high (unstable val), batch norm stats, small validation set variance.

Use learning curves, error analysis on validation set, and holdout that mirrors production.
</details>

---
