# AIML-LEC-AppliedML-DeepLearning

**Activity Type:** Lecture / Conceptual Workshop  
**Duration:** 180 minutes (3 hours)  
**Module:** 004 -- Applied ML: Deep Learning  
**Prerequisites:** Students have completed pre-class readings and videos on CNNs, RNNs, LSTMs, GRUs, and encoder-decoder architectures. Students have built MLPs in Module 003.

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Design a CNN architecture for image classification using convolutional, pooling, and fully connected layers.
2. Architect an RNN pipeline for sequential data processing.
3. Describe how LSTMs solve the vanishing gradient / memory constraint problem of vanilla RNNs.
4. Differentiate LSTM and GRU architectures and articulate trade-offs.
5. Architect a seq2seq encoder-decoder model for sequence translation.
6. Apply data augmentation techniques to improve CNN generalization.
7. Architect residual (skip) connections and explain why they help deep networks (stretch).

---

## Tech Stack

| Tool | Version / Notes |
|------|-----------------|
| Python | 3.10+ |
| NumPy | latest stable |
| matplotlib | latest stable |
| Dataset (Stage 1) | CIFAR-10 or equivalent (3×32×32 color images, 10 classes) |
| Dataset (Stage 2) | Synthetic sentiment sequences generated inline |
| Dataset (Stage 3) | Synthetic digit-string reversal sequences generated inline |

> **Note:** All model architectures in this lecture are expressed as **framework-agnostic pseudocode**. Students may implement them in any deep learning framework (PyTorch, TensorFlow/Keras, JAX, etc.) during assignments.

---

## Preparation Checklist (Instructor)

- [ ] Verify Python 3.10+, numpy, and matplotlib installed in demo environment.
- [ ] Have backup diagrams ready for: CNN feature maps, RNN unrolled computation, LSTM cell gates, encoder-decoder information flow.
- [ ] Test projector / screen-share for live demonstrations.
- [ ] Font size increased for projector readability (16pt+ in editor).

---

## Scenario

Students will work through a **multi-modal deep learning project** across three stages:

- **Stage 1 -- CNN Image Classifier:** Design a convolutional neural network that classifies images (airplanes, cars, birds, cats, etc.), understand data augmentation, and optionally add residual connections.
- **Stage 2 -- RNN/LSTM Text Sentiment:** Switch modalities to sequential text data. Design a vanilla RNN, observe its limitations on longer sequences, then upgrade to LSTM and GRU architectures.
- **Stage 3 -- Encoder-Decoder Sequence Translation:** Combine encoder and decoder RNNs into a seq2seq model that learns to reverse digit strings, demonstrating the encoder-decoder pattern used in machine translation.

Each stage builds on concepts from the previous one. The progression from spatial (images) to temporal (sequences) to structured (seq2seq) mirrors how deep learning architectures evolved.

---

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | CNNs for Image Data: Convolutions, Pooling, Data Augmentation, Residual Connections | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | RNNs for Sequence Data: Vanilla RNN, LSTM, GRU | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Encoder-Decoder Architectures: Seq2Seq for Sequence Translation | 45 |
| Buffer | Open Q&A, Wrap-Up | 15 |

---

# STAGE 1 -- CNNs for Image Data (55 min)

> **Goal:** Design a convolutional neural network for image classification, understand data augmentation, and grasp residual connections.

## STEP 1.1 -- Why CNNs? From MLPs to Convolutions (8 min)

**[PACING: Conceptual; no code yet.]**

- Recall from Module 003: MLPs flatten images into 1D vectors, losing spatial structure.
- A 32×32×3 image flattened is 3,072 inputs. An MLP with one 512-unit hidden layer needs 3,072 × 512 = 1.57M parameters in the first layer alone.
- CNNs exploit two key ideas:
  - **Local connectivity:** each neuron connects to a small spatial region (receptive field), not the entire input.
  - **Weight sharing:** the same filter (kernel) slides across the entire image, so learned features are translation-invariant.
- Three building blocks: **convolutional layers** (learn filters), **pooling layers** (downsample spatial dimensions), **fully connected layers** (final classification).

Draw or display a diagram showing a 3×3 kernel sliding across a 2D feature map, producing an output feature map. Annotate stride and padding.

**The Convolution Operation (math):**

For a single-channel input \(X\) and kernel \(K\) of size \(k \times k\), the output feature map at position \((i, j)\) is:

\[
Y(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X(i+m,\; j+n) \cdot K(m, n) + b
\]

With multiple input channels \(C_{in}\) and multiple output filters \(C_{out}\), each filter produces one output channel. The total learnable parameters for one convolutional layer: \(C_{out} \times (C_{in} \times k \times k + 1)\).

**Output size formula:** For input size \(W\), kernel size \(k\), padding \(p\), and stride \(s\):

\[
W_{out} = \frac{W - k + 2p}{s} + 1
\]

**Discussion Prompt:** "If a 3×3 kernel slides across a 32×32 image with stride 1 and no padding, what is the output size?" (30×30. With padding=1, it stays 32×32.)

---

## STEP 1.2 -- Exploring Image Data (8 min)

**[PACING: Live code. Students follow along.]**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# In practice, load a dataset like CIFAR-10 from your framework of choice.
# Images are 32x32 pixels with 3 color channels (RGB).
# Standard normalization uses per-channel means and standard deviations:
#   means = (0.4914, 0.4822, 0.4465)
#   stds  = (0.2470, 0.2435, 0.2616)
#
# For this conceptual walkthrough, we simulate an image batch:
SUBSET_SIZE = 5000
TEST_SUBSET = 1000

# Simulated image batch for shape demonstration
sample_images = np.random.rand(10, 32, 32, 3)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i])
    ax.set_title(CLASSES[i % len(CLASSES)])
    ax.axis("off")
plt.suptitle("Sample Images (32x32x3)")
plt.tight_layout()
plt.show()

print(f"Image shape:        (3, 32, 32)  [channels, height, width]")
print(f"Training samples:   {SUBSET_SIZE}")
print(f"Test samples:       {TEST_SUBSET}")
print(f"Number of classes:  {len(CLASSES)}")
```

Key points to narrate:
- A standard image classification dataset like CIFAR-10 has 60,000 32×32 color images in 10 classes. We conceptually use a 5,000-sample subset for lecture speed.
- Normalization values are precomputed per-channel means and standard deviations.
- Images are represented as 3D tensors: (channels, height, width) or (height, width, channels) depending on the framework.

---

## STEP 1.3 -- Design a Basic CNN (15 min)

**[PACING: Walk through the pseudocode architecture. Explain each layer carefully.]**

```
PSEUDOCODE: BasicCNN Architecture
==================================

MODEL BasicCNN(num_classes=10):

  FEATURE EXTRACTOR:
    # Conv block 1: 3 input channels → 32 filters
    Convolution2D(in=3, out=32, kernel=3×3, padding=1)   → output: 32 × 32 × 32
    ReLU activation
    MaxPool(2×2)                                          → output: 32 × 16 × 16

    # Conv block 2: 32 → 64 filters
    Convolution2D(in=32, out=64, kernel=3×3, padding=1)  → output: 64 × 16 × 16
    ReLU activation
    MaxPool(2×2)                                          → output: 64 × 8 × 8

    # Conv block 3: 64 → 128 filters
    Convolution2D(in=64, out=128, kernel=3×3, padding=1) → output: 128 × 8 × 8
    ReLU activation
    MaxPool(2×2)                                          → output: 128 × 4 × 4

  CLASSIFIER:
    Flatten                                               → output: 2048  (128 × 4 × 4)
    FullyConnected(in=2048, out=256)
    ReLU activation
    Dropout(rate=0.3)
    FullyConnected(in=256, out=num_classes)

  FORWARD PASS:
    x ← FeatureExtractor(input)
    output ← Classifier(x)
    RETURN output
```

Walk through each component:
- **Convolution2D(3, 32, 3×3, padding=1):** 32 learnable 3×3 filters applied to 3-channel input. Padding preserves spatial size.
- **ReLU:** element-wise non-linearity \(f(x) = \max(0, x)\) after each convolution.
- **MaxPool(2×2):** halves spatial dimensions by taking the maximum in each 2×2 window.
- **Flatten + FullyConnected:** after convolutions reduce spatial size to 4×4, we flatten to a 2,048-element vector and classify with fully connected layers.
- **Dropout(0.3):** randomly zeros 30% of activations during training to reduce overfitting.

**Parameter count analysis:**

```
Layer                          Parameters
─────────────────────────────────────────
Conv2D(3→32, 3×3)             32 × (3×3×3 + 1)     =     896
Conv2D(32→64, 3×3)            64 × (32×3×3 + 1)    =  18,496
Conv2D(64→128, 3×3)           128 × (64×3×3 + 1)   =  73,856
FC(2048→256)                  2048 × 256 + 256      = 524,544
FC(256→10)                    256 × 10 + 10         =   2,570
─────────────────────────────────────────
Total                                               ≈ 620,362
```

Compare to an equivalent MLP: a single hidden layer of 512 units on the flattened input would need 3,072 × 512 = 1.57M parameters in that layer alone. Weight sharing makes CNNs dramatically more parameter-efficient.

---

## STEP 1.4 -- CNN Training Concepts (10 min)

**[PACING: Conceptual walkthrough of the training loop. Reinforce the pattern from Module 003.]**

```
PSEUDOCODE: CNN Training Loop
==============================

FUNCTION train_cnn(model, train_data, test_data, epochs=15, lr=0.001):

    loss_function ← CrossEntropyLoss()
    optimizer     ← Adam(model.parameters, learning_rate=lr)

    FOR epoch = 1 TO epochs:
        model.set_training_mode()

        FOR EACH (images, labels) IN train_data:
            optimizer.zero_gradients()
            logits      ← model.forward(images)       # shape: (batch, num_classes)
            loss        ← loss_function(logits, labels)
            gradients   ← backpropagate(loss)
            optimizer.update_parameters(gradients)

        model.set_evaluation_mode()
        predictions ← model.forward(test_images)
        accuracy    ← count_correct(predictions, test_labels) / num_test_samples

        PRINT epoch, loss, accuracy

    RETURN training_losses, test_accuracies
```

Narrate:
- The training loop is identical to Module 003 except inputs are 4D image tensors (batch, channels, height, width) instead of 2D tabular data.
- **CrossEntropyLoss** combines softmax and negative log-likelihood: \(\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)\)
- With only 5,000 training samples, overfitting is a real risk. Watch for a widening gap between training loss and test accuracy.

**Discussion Prompt:** "How can you tell from the loss and accuracy curves whether overfitting is occurring?" (Training loss keeps decreasing but test accuracy plateaus or drops.)

---

## STEP 1.5 -- Data Augmentation (8 min)

**[PACING: Conceptual with visual examples. Show the impact on generalization.]**

Data augmentation applies random, label-preserving transformations to training images to artificially expand the dataset.

**Common augmentation techniques:**

| Technique | What It Does | Why It Helps |
|-----------|-------------|--------------|
| Random Horizontal Flip | Mirrors the image left-right | A flipped car is still a car; teaches lateral invariance |
| Random Crop (with padding) | Pads image by a few pixels, then crops back to original size | Small spatial translations teach position invariance |
| Color Jitter | Randomly adjusts brightness, contrast, saturation | Teaches lighting/color invariance |
| Random Rotation | Rotates image by a small angle | Teaches rotation invariance |
| Random Erasing | Masks out a random patch of the image | Forces model to use multiple regions for classification |

```
PSEUDOCODE: Augmented vs. Basic Data Pipeline
===============================================

Basic pipeline:
    image → Normalize(mean, std) → model

Augmented pipeline (training only):
    image → RandomHorizontalFlip(p=0.5)
          → RandomCrop(32, padding=4)
          → ColorJitter(brightness=0.2, contrast=0.2)
          → Normalize(mean, std)
          → model

Test pipeline (never augmented):
    image → Normalize(mean, std) → model
```

**Expected effect on learning curves:**

```
Without augmentation:          With augmentation:
  Loss ↓ rapidly                 Loss ↓ more slowly (harder inputs)
  Test acc plateaus early        Test acc keeps improving (better generalization)
  Gap = overfitting              Smaller gap = better generalization
```

Narrate:
- Augmentation is applied **only** to training data, never to test data.
- The augmented model may have higher training loss (harder inputs) but better test accuracy (better generalization).

**Q&A Checkpoint (2 min):** "When would horizontal flip be a bad augmentation choice?" (Digit recognition -- a flipped 6 becomes a 9.)

---

## STEP 1.6 -- Residual Connections (Stretch) (6 min)

**[PACING: Conceptual with pseudocode. Show and discuss if time allows.]**

**The degradation problem:** Deeper networks should be at least as good as shallower ones (the extra layers could learn the identity function). In practice, very deep networks trained with standard methods perform *worse* than shallower ones because gradients vanish or explode through many layers.

**Solution: Skip (residual) connections**

```
PSEUDOCODE: Residual Block
===========================

FUNCTION ResidualBlock(x, channels):
    residual ← x                                    # save input

    out ← Convolution2D(channels, channels, 3×3, padding=1)(x)
    out ← BatchNorm(out)
    out ← ReLU(out)

    out ← Convolution2D(channels, channels, 3×3, padding=1)(out)
    out ← BatchNorm(out)

    out ← out + residual                            # skip connection (element-wise add)
    out ← ReLU(out)

    RETURN out
```

**Math behind the skip connection:**

Without skip connection, the layer learns: \(H(x) = F(x)\) where \(F\) is the stacked conv layers.

With skip connection, the layer learns: \(H(x) = F(x) + x\)

This means \(F(x) = H(x) - x\), the **residual**. Learning to output zeros (the residual is nothing) is much easier than learning an identity mapping from scratch.

**During backpropagation:**

\[
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)
\]

The "+1" ensures gradients always have a direct path through the identity connection, preventing vanishing gradients even in very deep networks.

```
PSEUDOCODE: Mini ResNet Architecture
======================================

MODEL ResNetMini(num_classes=10):

    Prep layer:
        Convolution2D(in=3, out=64, kernel=3×3, padding=1)
        BatchNorm(64)
        ReLU

    ResidualBlock_1(channels=64)
    MaxPool(2×2)
    ResidualBlock_2(channels=64)

    Classifier:
        AdaptiveAveragePool → 1×1 spatial
        Flatten
        FullyConnected(in=64, out=num_classes)
```

Key points:
- **BatchNorm** normalizes activations per mini-batch, stabilizing training.
- **AdaptiveAveragePool(1×1)** collapses spatial dimensions to 1×1 regardless of input size.
- ResNets won ImageNet 2015 by training networks 150+ layers deep -- impossible without skip connections.

---

**[BREAK -- 5 min]**

> Students should review the CNN architecture pseudocode and be ready to discuss design choices.

---

# STAGE 2 -- RNNs for Sequence Data (55 min)

> **Goal:** Understand RNNs for sequential data, the vanishing gradient problem in sequences, and upgrades via LSTM and GRU architectures.

## STEP 2.1 -- From Images to Sequences (5 min)

**[PACING: Conceptual bridge. No code.]**

- CNNs process spatial data where nearby pixels are related.
- Many real-world problems involve **sequential** data: text, time series, audio, gene sequences.
- In sequences, the order matters and context from earlier steps affects the interpretation of later steps.
- **Recurrent neural networks** process sequences one step at a time, maintaining a hidden state that carries information forward.

Draw or display the RNN unrolled diagram:

```
x_1 --> [RNN Cell] --> h_1 --> [RNN Cell] --> h_2 --> ... --> h_T --> output
           |                      |
         h_0 (init)             h_1
```

Key idea: the same weights are shared across all time steps (weight sharing in time, just as CNN filters share weights in space).

**The vanilla RNN equation:**

\[
h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)
\]

Where:
- \(x_t\) is the input at time step \(t\)
- \(h_{t-1}\) is the previous hidden state
- \(W_{xh}\) and \(W_{hh}\) are weight matrices (shared across all time steps)
- \(\tanh\) squashes the output to the range \([-1, 1]\)

---

## STEP 2.2 -- Generate Synthetic Sentiment Data (8 min)

**[PACING: Live code. Explain the synthetic data rationale.]**

```python
# STEP 2.2 -- Synthetic sentiment dataset
# Positive reviews: sequences with higher average token values
# Negative reviews: sequences with lower average token values

import numpy as np

VOCAB_SIZE = 50
SEQ_LEN = 20
NUM_TRAIN = 2000
NUM_TEST = 400

def generate_sentiment_data(n_samples, seq_len, vocab_size, seed=42):
    rng = np.random.RandomState(seed)
    sequences = []
    labels = []
    for _ in range(n_samples):
        label = rng.randint(0, 2)
        if label == 1:  # positive
            seq = rng.randint(vocab_size // 2, vocab_size, size=seq_len)
        else:           # negative
            seq = rng.randint(0, vocab_size // 2, size=seq_len)
        noise_idx = rng.choice(seq_len, size=seq_len // 4, replace=False)
        seq[noise_idx] = rng.randint(0, vocab_size, size=len(noise_idx))
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = generate_sentiment_data(NUM_TRAIN, SEQ_LEN, VOCAB_SIZE, seed=42)
X_test_seq, y_test_seq = generate_sentiment_data(NUM_TEST, SEQ_LEN, VOCAB_SIZE, seed=99)

print(f"Training sequences shape: {X_train_seq.shape}")   # (2000, 20)
print(f"Test sequences shape:     {X_test_seq.shape}")     # (400, 20)
print(f"Vocabulary size:          {VOCAB_SIZE}")
print(f"Sequence length:          {SEQ_LEN}")
print(f"Class balance:            {y_train_seq.mean():.2f} positive")
print(f"\nSample sequence:          {X_train_seq[0]}")
print(f"Sample label:             {y_train_seq[0]}")
```

> "We generate synthetic sentiment data so every student gets identical results and we control the difficulty. Positive sequences use tokens from the upper half of the vocabulary; negative sequences use the lower half, with 25% noise injected. A real NLP task would use word embeddings on actual text -- the architecture is the same."

---

## STEP 2.3 -- Vanilla RNN for Sentiment (12 min)

**[PACING: Pseudocode architecture walkthrough. This is the first RNN students design.]**

```
PSEUDOCODE: Vanilla RNN Classifier
====================================

MODEL VanillaRNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes):

  LAYERS:
    Embedding(vocab_size, embed_dim)
        Converts integer token IDs → dense vectors
        Input:  (batch, seq_len)  of integers
        Output: (batch, seq_len, embed_dim)

    RNN(input_size=embed_dim, hidden_size=hidden_dim)
        At each time step t:
            h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)
        Input:  (batch, seq_len, embed_dim)
        Output: all_hidden_states (batch, seq_len, hidden_dim)
                final_hidden h_T  (batch, hidden_dim)

    FullyConnected(hidden_dim, num_classes)
        Maps the final hidden state to class logits

  FORWARD PASS:
    embedded     ← Embedding(input_tokens)     # (batch, seq_len, embed_dim)
    all_h, h_T   ← RNN(embedded)               # h_T: (batch, hidden_dim)
    logits       ← FC(h_T)                     # (batch, num_classes)
    RETURN logits
```

Explain each component:
- **Embedding:** converts integer token IDs into dense vectors. Vocabulary of 50 tokens, each mapped to a 16-dimensional vector. This is a lookup table that is learned during training.
- **RNN:** processes the embedded sequence step by step. At each step: \(h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b)\).
- **h_T (final hidden state):** a summary of the whole input sequence after processing all time steps.
- **FullyConnected:** maps the summary vector to class logits.

**Parameter count for RNN(embed=16, hidden=32):**

```
Embedding(50, 16):     50 × 16            =    800
W_xh (input→hidden):  16 × 32            =    512
W_hh (hidden→hidden): 32 × 32            =  1,024
b_h:                   32                 =     32
FC(32→2):             32 × 2 + 2         =     66
──────────────────────────────────────────────────
Total                                     ≈  2,434
```

```
PSEUDOCODE: Training a Sequence Classifier
============================================

FUNCTION train_seq_model(model, train_data, X_test, y_test, epochs=20, lr=0.001):

    loss_function ← CrossEntropyLoss()
    optimizer     ← Adam(model.parameters, lr)

    FOR epoch = 1 TO epochs:
        FOR EACH (X_batch, y_batch) IN batches(train_data, batch_size=64):
            optimizer.zero_gradients()
            logits    ← model.forward(X_batch)
            loss      ← loss_function(logits, y_batch)
            gradients ← backpropagate(loss)
            optimizer.update_parameters(gradients)

        predictions ← model.forward(X_test).argmax(axis=-1)
        accuracy    ← mean(predictions == y_test)
        PRINT epoch, loss, accuracy

    RETURN losses, accuracies
```

---

## STEP 2.4 -- The Vanishing Gradient Problem in RNNs (8 min)

**[PACING: Conceptual + visual demonstration. Critical topic.]**

> "Vanilla RNNs work on short sequences but struggle with long-range dependencies. Here is why."

- During **backpropagation through time (BPTT)**, gradients are multiplied by the recurrent weight matrix \(W_{hh}\) at every time step.
- The gradient of the loss with respect to hidden state at step \(k\) involves:

\[
\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=k+1}^{T} W_{hh}^{\top} \cdot \text{diag}(\tanh'(z_t))
\]

- If the largest singular value of \(W_{hh}\) is < 1, this product shrinks exponentially → **vanishing gradients**.
- If the largest singular value of \(W_{hh}\) is > 1, this product grows exponentially → **exploding gradients**.
- Practical effect: vanilla RNNs cannot learn that a word at position 1 should influence the prediction 50 steps later.

**Visualizing gradient flow:**

```python
# Demonstrate how gradient magnitudes decay across time steps
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
seq_length = 50
W_singular_value = 0.9

gradient_norms = []
current_magnitude = 1.0
for t in range(seq_length):
    gradient_norms.append(current_magnitude)
    current_magnitude *= W_singular_value

gradient_norms = gradient_norms[::-1]

plt.figure(figsize=(10, 4))
plt.bar(range(seq_length), gradient_norms)
plt.xlabel("Time Step (earlier → later)")
plt.ylabel("Relative Gradient Magnitude")
plt.title("Gradient Flow in Vanilla RNN (50 steps, σ_max = 0.9)")
plt.tight_layout()
plt.show()

print(f"Gradient at step 0 (earliest):  {gradient_norms[0]:.6f}")
print(f"Gradient at step 49 (latest):   {gradient_norms[49]:.6f}")
print(f"Ratio (earliest/latest):        {gradient_norms[0] / gradient_norms[49]:.6f}")
```

> "Notice how the gradient magnitude decays exponentially as we move earlier in the sequence. The network receives almost no learning signal from the earliest time steps. This is the vanishing gradient problem for sequences."

---

## STEP 2.5 -- LSTM: Gated Memory (12 min)

**[PACING: Conceptual first, then pseudocode. Walk through gates carefully.]**

### How LSTMs Solve the Problem

Draw or display the LSTM cell diagram with four components:

1. **Forget gate** \(f_t\): decides what to discard from the cell state.

\[f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)\]

2. **Input gate** \(i_t\): decides which new information to store.

\[i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)\]

3. **Candidate values** \(\tilde{c}_t\): proposed new cell content.

\[\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)\]

4. **Cell state update**: the cell state is a **highway** where gradients can flow without repeated multiplication by \(W_{hh}\).

\[c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\]

5. **Output gate** \(o_t\): decides what to expose as the hidden state.

\[o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)\]
\[h_t = o_t \odot \tanh(c_t)\]

The critical insight: the cell state update is an **additive** operation (\(f_t \odot c_{t-1} + \ldots\)), not a repeated matrix multiplication. This is why gradients can flow across many time steps without vanishing.

```
PSEUDOCODE: LSTM Classifier
=============================

MODEL LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes):

  LAYERS:
    Embedding(vocab_size, embed_dim)
    LSTM(input_size=embed_dim, hidden_size=hidden_dim)
        Internally computes all four gates at each time step.
        Returns: all_outputs, (final_hidden h_T, final_cell c_T)
    FullyConnected(hidden_dim, num_classes)

  FORWARD PASS:
    embedded             ← Embedding(input_tokens)
    all_outputs, (h_T, c_T) ← LSTM(embedded)
    logits               ← FC(h_T)
    RETURN logits

  PARAMETER COUNT:
    The LSTM has ≈4× more parameters than a vanilla RNN of the same
    hidden size because it has four gate weight matrices instead of one.
    For embed=16, hidden=32:
        Embedding:  800
        LSTM:       4 × (16×32 + 32×32 + 32) = 4 × 1,568 = 6,272
        FC:         66
        Total:      ≈ 7,138
```

Narrate:
- The architectural change from RNN to LSTM is conceptually minimal: replace the single recurrence equation with the four-gate system.
- The LSTM has approximately 4× more parameters than a vanilla RNN of the same hidden size (four gate matrices instead of one).
- Despite the internal complexity, most frameworks wrap the LSTM in a single layer call.

---

## STEP 2.6 -- GRU: Simplified Gating (5 min)

**[PACING: Conceptual + comparison. Compare to LSTM.]**

The GRU simplifies the LSTM by merging the forget and input gates into a single **update gate**, and eliminating the separate cell state:

1. **Update gate** \(z_t\): controls how much of the previous hidden state to keep.

\[z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)\]

2. **Reset gate** \(r_t\): controls how much of the previous hidden state to use when computing the candidate.

\[r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)\]

3. **Candidate hidden state:**

\[\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)\]

4. **Final hidden state** (interpolation between old and new):

\[h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t\]

```
PSEUDOCODE: GRU Classifier
============================

MODEL GRUClassifier(vocab_size, embed_dim, hidden_dim, num_classes):

  LAYERS:
    Embedding(vocab_size, embed_dim)
    GRU(input_size=embed_dim, hidden_size=hidden_dim)
        Two gates: update (z) and reset (r)
        No separate cell state; uses h_t only
        Returns: all_outputs, final_hidden h_T
    FullyConnected(hidden_dim, num_classes)

  FORWARD PASS:
    embedded       ← Embedding(input_tokens)
    all_outputs, h_T ← GRU(embedded)
    logits         ← FC(h_T)
    RETURN logits

  PARAMETER COUNT (embed=16, hidden=32):
    Embedding:  800
    GRU:        3 × (16×32 + 32×32 + 32) = 3 × 1,568 = 4,704
    FC:         66
    Total:      ≈ 5,570   (vs. LSTM ≈ 7,138)
```

### LSTM vs GRU comparison:

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Separate c_t and h_t | No separate cell state; uses h_t only |
| Parameters | ~4× hidden_dim² | ~3× hidden_dim² |
| Performance | Slightly better on complex long-range tasks | Comparable; faster to train |
| When to use | Default for most tasks; well-studied | When training speed matters or data is limited |

---

## STEP 2.7 -- Compare All Three Architectures (5 min)

**[PACING: Conceptual discussion + expected behavior.]**

In practice, training all three models (Vanilla RNN, LSTM, GRU) on the same data and plotting their learning curves reveals characteristic patterns:

```
Expected Training Loss:              Expected Test Accuracy:
  Vanilla RNN: slow convergence        Vanilla RNN: lower ceiling
  LSTM:        faster convergence       LSTM:        highest on long sequences
  GRU:         fastest convergence      GRU:         close to LSTM, faster training
```

On short sequences (like our 20-token task), all three may perform similarly. The differences become stark on longer sequences where long-range dependencies matter.

```python
# Visualization of expected comparison (conceptual)
import matplotlib.pyplot as plt

epochs = list(range(1, 21))
# Simulated representative curves
losses_rnn  = [0.7 - 0.015*e + 0.002*e**0.5 for e in epochs]
losses_lstm = [0.7 - 0.025*e + 0.003*e**0.5 for e in epochs]
losses_gru  = [0.7 - 0.027*e + 0.003*e**0.5 for e in epochs]

accs_rnn  = [0.5 + 0.02*e - 0.0005*e**2 for e in epochs]
accs_lstm = [0.5 + 0.025*e - 0.0005*e**2 for e in epochs]
accs_gru  = [0.5 + 0.024*e - 0.0005*e**2 for e in epochs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, losses_rnn, label="Vanilla RNN")
ax1.plot(epochs, losses_lstm, label="LSTM")
ax1.plot(epochs, losses_gru, label="GRU")
ax1.set_title("Training Loss (Representative)"); ax1.set_xlabel("Epoch"); ax1.legend()

ax2.plot(epochs, accs_rnn, label="Vanilla RNN")
ax2.plot(epochs, accs_lstm, label="LSTM")
ax2.plot(epochs, accs_gru, label="GRU")
ax2.set_title("Test Accuracy (Representative)"); ax2.set_xlabel("Epoch"); ax2.legend()

plt.tight_layout(); plt.show()
```

**Q&A Checkpoint (3 min):** "On this short-sequence task, all three may perform similarly. When would you expect LSTM/GRU to pull ahead of vanilla RNN?" (Longer sequences, tasks requiring long-range memory, sentiment that depends on early context.)

---

**[BREAK -- 5 min]**

> Students should review the RNN/LSTM/GRU architectures and gate equations.

---

# STAGE 3 -- Encoder-Decoder Architectures (45 min)

> **Goal:** Understand the seq2seq encoder-decoder pattern and how it enables sequence-to-sequence tasks like translation.

## STEP 3.1 -- The Encoder-Decoder Idea (8 min)

**[PACING: Conceptual. Draw the architecture before pseudocode.]**

- So far, our RNNs processed a sequence and produced a single output (classification).
- Many tasks require **sequence-to-sequence** output: machine translation (English to French), summarization (long text to short text), chatbots (question to answer).
- The **encoder-decoder** architecture solves this:
  1. **Encoder:** reads the entire input sequence, compresses it into a fixed-size context vector (the final hidden state).
  2. **Decoder:** takes the context vector and generates the output sequence one token at a time.

Draw the architecture:

```
Encoder:  x_1, x_2, ..., x_n --> [LSTM] --> context vector (h_n, c_n)
                                                  |
Decoder:  <SOS> --> [LSTM] --> y_1 --> [LSTM] --> y_2 --> ... --> <EOS>
```

- The context vector is a **bottleneck**: it must capture everything the decoder needs to know about the input. This limitation motivated attention mechanisms (covered in Module 005).
- For today, we will use a simple task: **reversing digit strings** (e.g., "1 2 3 4" → "4 3 2 1"). This isolates the architecture from language complexity.

---

## STEP 3.2 -- Generate Sequence Reversal Data (5 min)

**[PACING: Live code.]**

```python
# STEP 3.2 -- Generate digit-string reversal dataset
import numpy as np

PAD_TOKEN = 0
SOS_TOKEN = 11
EOS_TOKEN = 12
NUM_TOKENS = 13  # 0=PAD, 1-10=digits, 11=SOS, 12=EOS
MAX_LEN = 6

def generate_reversal_data(n_samples, max_len=MAX_LEN, seed=42):
    rng = np.random.RandomState(seed)
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []

    for _ in range(n_samples):
        length = rng.randint(3, max_len + 1)
        seq = rng.randint(1, 11, size=length).tolist()
        reversed_seq = seq[::-1]

        enc_in = seq + [PAD_TOKEN] * (max_len - length)
        dec_in = [SOS_TOKEN] + reversed_seq + [PAD_TOKEN] * (max_len - length)
        dec_tgt = reversed_seq + [EOS_TOKEN] + [PAD_TOKEN] * (max_len - length)

        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(dec_tgt)

    return (np.array(encoder_inputs),
            np.array(decoder_inputs),
            np.array(decoder_targets))

enc_train, dec_in_train, dec_tgt_train = generate_reversal_data(3000, seed=42)
enc_test, dec_in_test, dec_tgt_test = generate_reversal_data(500, seed=99)

print(f"Encoder input shape:  {enc_train.shape}")
print(f"Decoder input shape:  {dec_in_train.shape}")
print(f"Decoder target shape: {dec_tgt_train.shape}")
print(f"\nExample:")
print(f"  Encoder input:  {enc_train[0].tolist()}")
print(f"  Decoder input:  {dec_in_train[0].tolist()}")
print(f"  Decoder target: {dec_tgt_train[0].tolist()}")
```

Explain the three sequences:
- **Encoder input:** the original digit string, padded.
- **Decoder input:** SOS token + reversed digits, padded. This is what the decoder sees during training (teacher forcing).
- **Decoder target:** reversed digits + EOS token, padded. This is what the decoder should predict.

---

## STEP 3.3 -- Design the Encoder (7 min)

**[PACING: Pseudocode walkthrough.]**

```
PSEUDOCODE: Encoder
=====================

MODEL Encoder(vocab_size, embed_dim, hidden_dim):

  LAYERS:
    Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        Zero embedding for padding tokens so they don't
        contribute to the learned representation.
    LSTM(input_size=embed_dim, hidden_size=hidden_dim)

  FORWARD PASS:
    embedded              ← Embedding(input_sequence)    # (batch, seq_len, embed_dim)
    all_outputs, (h_n, c_n) ← LSTM(embedded)

    RETURN (h_n, c_n)     # context vector only -- discard per-step outputs
```

- The encoder reads the input and produces only the final hidden state and cell state.
- `padding_idx=PAD_TOKEN` ensures padding tokens get zero embeddings.
- The `(h_n, c_n)` tuple is the context vector passed to the decoder.

**Conceptual diagram:**

```
Input:    [3, 7, 2, 5, 0, 0]   (padded to MAX_LEN)
           ↓  ↓  ↓  ↓  ↓  ↓
Embed:   [e3, e7, e2, e5, 0, 0]
           ↓  ↓  ↓  ↓  ↓  ↓
LSTM:    h0→h1→h2→h3→h4→h5→h6
                                 ↓
Context vector:               (h6, c6)  ← "summary" of input
```

---

## STEP 3.4 -- Design the Decoder (7 min)

**[PACING: Pseudocode walkthrough.]**

```
PSEUDOCODE: Decoder
=====================

MODEL Decoder(vocab_size, embed_dim, hidden_dim):

  LAYERS:
    Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
    LSTM(input_size=embed_dim, hidden_size=hidden_dim)
    FullyConnected(hidden_dim, vocab_size)
        Maps each hidden state to a distribution over the vocabulary

  FORWARD PASS:
    embedded                    ← Embedding(decoder_input)      # (batch, seq_len, embed_dim)
    output, (h_n, c_n)         ← LSTM(embedded, initial_hidden) # initial_hidden = context vector
    logits                     ← FC(output)                     # (batch, seq_len, vocab_size)

    RETURN logits, (h_n, c_n)
```

- The decoder is structurally similar to the encoder but adds a linear output layer.
- It receives the context vector as its initial hidden state.
- During training, we use **teacher forcing:** feed the correct previous token (from `decoder_inputs`) at each step instead of the model's own prediction.

**Teacher forcing vs. autoregressive decoding:**

```
Teacher forcing (training):
    Input to decoder at step t: ground-truth token y_{t-1}
    ✓ Stable training, faster convergence
    ✗ Exposure bias: model never sees its own mistakes during training

Autoregressive decoding (inference):
    Input to decoder at step t: model's prediction ŷ_{t-1}
    Errors can compound (one wrong token leads to more wrong tokens)
```

---

## STEP 3.5 -- Assemble the Seq2Seq Model (5 min)

**[PACING: Pseudocode walkthrough.]**

```
PSEUDOCODE: Seq2Seq Model
===========================

MODEL Seq2Seq(encoder, decoder):

  FORWARD PASS:
    context_vector  ← encoder.forward(source_sequence)      # (h_n, c_n)
    logits, _       ← decoder.forward(target_input, context_vector)

    RETURN logits    # (batch, target_seq_len, vocab_size)


INSTANTIATION:
    embed_dim  = 16
    hidden_dim = 64

    encoder  ← Encoder(NUM_TOKENS, embed_dim, hidden_dim)
    decoder  ← Decoder(NUM_TOKENS, embed_dim, hidden_dim)
    seq2seq  ← Seq2Seq(encoder, decoder)

PARAMETER COUNT:
    Encoder embedding:  13 × 16                    =     208
    Encoder LSTM:       4 × (16×64 + 64×64 + 64)  =  20,736
    Decoder embedding:  13 × 16                    =     208
    Decoder LSTM:       4 × (16×64 + 64×64 + 64)  =  20,736
    Decoder FC:         64 × 13 + 13               =     845
    ────────────────────────────────────────────────────────
    Total                                          ≈  42,733
```

> "The Seq2Seq model is two modules composed together. The encoder compresses; the decoder generates. This pattern is the foundation of modern NLP -- GPT is a decoder, BERT is an encoder, and T5 is an encoder-decoder."

---

## STEP 3.6 -- Seq2Seq Training Concepts (8 min)

**[PACING: Pseudocode walkthrough of training.]**

```
PSEUDOCODE: Seq2Seq Training Loop
====================================

FUNCTION train_seq2seq(model, train_data, test_data, epochs=30, lr=0.001):

    loss_function ← CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer     ← Adam(model.parameters, lr)

    FOR epoch = 1 TO epochs:
        model.set_training_mode()

        FOR EACH (enc_input, dec_input, dec_target) IN batches(train_data, batch_size=64):
            optimizer.zero_gradients()

            logits ← model.forward(enc_input, dec_input)
            # logits shape: (batch, seq_len, vocab_size)
            # Reshape for loss: (batch × seq_len, vocab_size) vs (batch × seq_len)
            loss ← loss_function(
                reshape(logits, [-1, vocab_size]),
                reshape(dec_target, [-1])
            )

            gradients ← backpropagate(loss)
            optimizer.update_parameters(gradients)

        # Evaluate: token-level accuracy on non-padded positions
        model.set_evaluation_mode()
        test_logits   ← model.forward(enc_test, dec_in_test)
        test_preds    ← argmax(test_logits, axis=-1)
        mask          ← (dec_tgt_test ≠ PAD_TOKEN)
        token_accuracy ← sum((test_preds == dec_tgt_test) AND mask) / sum(mask)

        PRINT epoch, loss, token_accuracy

    RETURN losses, accuracies
```

Narrate:
- `CrossEntropyLoss(ignore_index=PAD_TOKEN)` tells the loss function to skip padding positions.
- We compute **token-level accuracy** on non-padded positions.
- Teacher forcing during training means the decoder always sees the correct previous token. At inference time, we would feed the model's own predictions back in (autoregressive decoding).

---

## STEP 3.7 -- Qualitative Evaluation (5 min)

**[PACING: Discuss expected outputs.]**

After training, we evaluate the model by comparing its predictions to the expected reversed sequences:

```
PSEUDOCODE: Evaluation
========================

FUNCTION show_predictions(model, enc_input, dec_input, dec_target, n=5):
    model.set_evaluation_mode()

    logits ← model.forward(enc_input[:n], dec_input[:n])
    preds  ← argmax(logits, axis=-1)

    FOR i = 1 TO n:
        source    ← remove_padding(enc_input[i])
        expected  ← remove_padding_and_eos(dec_target[i])
        predicted ← preds[i][:length(expected)]
        match     ← "CORRECT" if predicted == expected else "WRONG"
        PRINT "Input:", source, "Expected:", expected, "Predicted:", predicted, match
```

**Example expected output after training:**

```
Input: [3, 7, 2, 5]  Expected: [5, 2, 7, 3]  Predicted: [5, 2, 7, 3]  [CORRECT]
Input: [8, 1, 4]      Expected: [4, 1, 8]      Predicted: [4, 1, 8]      [CORRECT]
Input: [6, 9, 2, 1, 5] Expected: [5, 1, 2, 9, 6] Predicted: [5, 1, 2, 9, 6] [CORRECT]
Input: [2, 10, 7]     Expected: [7, 10, 2]     Predicted: [7, 10, 2]     [CORRECT]
Input: [4, 3, 8, 6]   Expected: [6, 8, 3, 4]   Predicted: [6, 8, 3, 4]   [CORRECT]
```

> "This is a toy task, but the architecture is identical to what powers sequence-to-sequence models in production. Replace digit tokens with word tokens, scale up the hidden dimensions, and add attention -- and you have a machine translation system."

**Q&A Checkpoint (3 min):** "What is the main limitation of compressing the entire input into a single context vector?" (Information bottleneck. Long or complex inputs lose detail. This motivates attention mechanisms in Module 005.)

---

**[Q&A / WRAP-UP -- 15 min buffer]**

## Session Recap

| Topic | Key Takeaway |
|-------|-------------|
| CNNs | Exploit spatial structure with local filters and weight sharing; fewer parameters than MLPs on images |
| Data Augmentation | Artificially expands training set with label-preserving transforms; improves generalization |
| Residual Connections | Skip connections enable gradient flow in deep networks; additive, not multiplicative |
| Vanilla RNNs | Process sequences with shared weights across time steps; suffer from vanishing gradients |
| LSTMs | Gated architecture with cell state highway; solves vanishing gradient for sequences |
| GRUs | Simplified gating (2 vs 3 gates); comparable performance, fewer parameters |
| Encoder-Decoder | Encoder compresses input to context vector; decoder generates output sequence |
| Teacher Forcing | Feed ground-truth tokens to decoder during training for stable learning |

## Exit Criteria Checklist

| Exit Criterion | Where Demonstrated |
|---|---|
| Design a CNN for image classification | Steps 1.3, 1.4 (BasicCNN architecture and training concepts) |
| Architect an RNN pipeline | Steps 2.2, 2.3 (VanillaRNNClassifier design for sentiment) |
| Describe how LSTMs solve memory constraints | Step 2.4 (vanishing gradient visualization), Step 2.5 (LSTM gate walkthrough) |
| Architect seq2seq encoder-decoder | Steps 3.3-3.5 (Encoder, Decoder, Seq2Seq design) |
| Apply data augmentation | Step 1.5 (augmentation techniques, before/after analysis) |
| Differentiate LSTM vs GRU | Step 2.6 (GRU equations + comparison table) |
| Architect residual connections (stretch) | Step 1.6 (ResidualBlock, ResNetMini pseudocode) |

## Suggested Exercises

Have students:
1. Implement the BasicCNN pseudocode in a framework of their choice and train on CIFAR-10.
2. Implement the VanillaRNN, LSTM, and GRU classifiers and compare learning curves.
3. Implement the Seq2Seq model and experiment with:
   - Changing the number of CNN filters or adding a fourth conv block.
   - Increasing LSTM hidden size or adding a second LSTM layer.
   - Trying greedy autoregressive decoding instead of teacher forcing.

## Additional Resources

- Understanding LSTMs (Chris Olah): https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- "Deep Learning" by Goodfellow, Bengio, Courville -- Chapters 9 (CNNs) and 10 (RNNs)
- Stanford CS231n (CNNs for Visual Recognition): https://cs231n.stanford.edu/
- Stanford CS224n (NLP with Deep Learning): https://web.stanford.edu/class/cs224n/
