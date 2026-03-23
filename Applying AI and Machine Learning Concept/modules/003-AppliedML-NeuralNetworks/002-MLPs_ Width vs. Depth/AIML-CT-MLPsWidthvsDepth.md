# MLPs: Width vs. Depth

**Estimated Time:** 10 Minutes

---

## Introduction

Now that you understand how a single perceptron works, the natural next question is: what happens when you combine many of them? A **Multi-Layer Perceptron (MLP)** is exactly that -- layers of neurons stacked together, where each neuron in one layer connects to every neuron in the next. MLPs are the workhorse architecture behind a vast number of practical machine learning applications, from tabular data classification to serving as building blocks inside larger architectures like transformers.

But building an MLP immediately raises a design question that has real consequences: should you make your network **wider** (more neurons per layer) or **deeper** (more layers)? This reading will give you the vocabulary and intuition to reason about that tradeoff.

---

## Core Concepts

### Anatomy of an MLP

An MLP has three types of layers:

- **Input layer:** Receives the raw feature vector. Its size is determined by your data (e.g., 784 for a flattened 28x28 image).
- **Hidden layers:** One or more layers between input and output where the actual learning happens. Each hidden layer applies a linear transformation followed by a nonlinear activation function.
- **Output layer:** Produces the final prediction. Its size and activation depend on the task (1 neuron with sigmoid for binary classification, N neurons with softmax for N-class classification, 1 neuron with no activation for regression).

### Building an MLP (Conceptual Structure)

In pseudocode, an MLP with configurable width and depth looks like:

```
function MLP(input_size, hidden_size, num_hidden_layers, output_size):
    layers = []
    prev_size = input_size

    for i = 1 to num_hidden_layers:
        layers.add( LinearLayer(prev_size, hidden_size) )
        layers.add( ReLU() )
        prev_size = hidden_size

    layers.add( LinearLayer(prev_size, output_size) )
    return Sequential(layers)
```

Notice the pattern: each hidden layer is a **Linear + Activation** pair, and the output layer is just Linear (you typically apply the final activation, such as softmax, inside the loss function rather than in the model itself). This pattern is consistent across every major deep learning framework.

### What "Width" and "Depth" Mean

- **Width** refers to the number of neurons in each hidden layer. A wider layer can represent more features simultaneously within a single transformation step.
- **Depth** refers to the number of hidden layers. A deeper network can compose simpler features into increasingly abstract representations through successive transformations.

### The Universal Approximation Theorem

A classic result in neural network theory states that a single hidden layer with enough neurons can approximate any continuous function to arbitrary accuracy. So why bother with depth at all?

The key word is "enough." While a single wide layer *can* theoretically represent any function, it might need an impractically large number of neurons to do so. Deep networks can often represent the same function with **exponentially fewer total parameters** because each layer builds on the features extracted by the previous one.

Think of it this way: recognizing a face in an image could be done by memorizing every possible pixel pattern (very wide, single layer), or by learning edges first, then shapes, then facial features, then faces (deep, compositional). The second approach is far more parameter-efficient and generalizes better.

### The Width vs. Depth Tradeoff

| Consideration | Wider Networks | Deeper Networks |
|---|---|---|
| Expressiveness | Can approximate any function with one hidden layer (given enough width) | Can represent hierarchical, compositional features more efficiently |
| Parameter efficiency | Often needs more total parameters to match a deep network's capacity | Can achieve similar capacity with fewer parameters |
| Training difficulty | Generally easier to train; less prone to vanishing gradients | Harder to train; may require residual connections, careful initialization, or normalization |
| Overfitting risk | High width with limited data can memorize rather than generalize | Depth also adds capacity, but compositional structure can act as implicit regularization |
| Computational cost | Wide layers are easy to parallelize on GPUs | Deep networks require sequential computation through layers |

### Practical Guidelines

There is no universal formula, but these heuristics serve as good starting points:

- **Start simple.** Begin with 1-2 hidden layers and a moderate width (64-256 neurons). Only add complexity if the model underfits.
- **For tabular data,** 2-4 hidden layers with widths between 64 and 512 is usually sufficient. Going much deeper rarely helps and can hurt.
- **For problems with hierarchical structure** (images, sequences), depth tends to pay off more than raw width.
- **Monitor both training and validation loss.** If training loss is low but validation loss is high, your network has too much capacity (too wide, too deep, or both). If training loss stays high, you need more capacity.

### Counting Parameters

Understanding how width and depth affect parameter count helps you reason about model capacity. For a fully connected layer mapping \(m\) inputs to \(n\) outputs, the parameter count is \(m \times n + n\) (weights plus biases).

Consider two architectures for an input size of 784 and 10 output classes:

- **Narrow and deep** (6 hidden layers of 64 neurons): \(784 \times 64 + 5 \times 64 \times 64 + 64 \times 10 \approx 71{,}690\) parameters
- **Wide and shallow** (1 hidden layer of 512 neurons): \(784 \times 512 + 512 \times 10 \approx 406{,}538\) parameters

Two networks can have vastly different total parameter counts but achieve similar performance. What matters is how those parameters are organized -- depth enables compositional feature learning, while width enables broader single-step representation.

---

## Connecting to Practice

In the hands-on exercises that follow, you will experiment with building MLPs of varying widths and depths and observe how these architectural choices affect training speed, final accuracy, and overfitting behavior. The goal is not to find a single "best" architecture but to develop the intuition for adjusting width and depth based on what you observe during training.

When you encounter a new problem in practice, the mental framework should be: start with a simple architecture, check whether the model can fit the training data (if not, add capacity), then check whether it generalizes to validation data (if not, reduce capacity or add regularization). Width and depth are your two primary capacity knobs.

---

## Further Learning & Resources

### Documentation

1. [Deep Learning Book, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html) -- Goodfellow, Bengio, and Courville's treatment of MLP architecture, including the Universal Approximation Theorem.
2. [Wikipedia: Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) -- Formal statement and implications for neural network design.

### Interactive Resources

1. [TensorFlow Playground](https://playground.tensorflow.org/) -- Experiment with adding neurons (width) and layers (depth) and immediately see how decision boundaries change.
2. [Google Machine Learning Crash Course: Multi-Layer Perceptrons](https://developers.google.com/machine-learning/crash-course/multi-layer-perceptrons) -- Interactive exercises that walk through building and evaluating MLPs on real datasets.
3. [Distill.pub: Neural Network Architectures](https://distill.pub/) -- Beautifully visualized explorations of how neural network architecture choices affect learning, with interactive diagrams.
