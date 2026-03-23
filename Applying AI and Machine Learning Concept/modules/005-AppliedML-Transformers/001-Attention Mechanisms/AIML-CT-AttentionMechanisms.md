# Attention Mechanisms

**Estimated Time:** 10 Minutes

## Introduction

Imagine reading a long paragraph and trying to answer a question about it. You do not give every word equal weight -- you naturally focus on the parts that matter most. That is exactly what attention mechanisms do for neural networks. Before attention came along, sequence models like RNNs had to compress an entire input into a single fixed-size vector, which created a brutal information bottleneck for long sequences. Attention solved this by letting the model look back at all input positions and decide, dynamically, which ones are most relevant for producing each output.

This reading walks you through the core math and intuition behind scaled dot-product attention -- the building block that powers every modern Transformer.

## Core Concepts

### The Query-Key-Value Framework

Attention is built on three abstract roles: **Queries (Q)**, **Keys (K)**, and **Values (V)**. A useful analogy is a search engine. You type a query, the engine compares it against keys (page titles, metadata), and returns the values (page content) that best match.

In a neural network, Q, K, and V are all produced by applying learned linear projections to the input embeddings:

```
Q = X * W_q    (each token's "question")
K = X * W_k    (each token's "label for matching")
V = X * W_v    (each token's "content to retrieve")
```

where \(X\) is a matrix of token embeddings (shape: sequence length x embedding dimension) and \(W_q, W_k, W_v\) are learned weight matrices. In **self-attention**, Q, K, and V all come from the same input sequence. In **cross-attention** (used in encoder-decoder models), Q comes from one sequence and K/V come from another.

### Scaled Dot-Product Attention

The full formulation is:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

Breaking this down step by step:

1. **Dot products** (\(QK^\top\)): Compute a similarity score between every query and every key. The result is a matrix of shape (seq_len, seq_len).
2. **Scaling** (divide by \(\sqrt{d_k}\)): Without scaling, the dot products can grow large in magnitude as \(d_k\) increases, which pushes the softmax into regions with extremely small gradients. Dividing by the square root of the key dimension keeps the variance roughly at 1 and training stable.
3. **Softmax**: Converts raw scores into a probability distribution over keys for each query position.
4. **Weighted sum** (multiply by \(V\)): Each output position is a weighted combination of value vectors, where the weights come from the softmax.

In pseudocode:

```
function scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = dimension of Q's last axis
    scores = (Q * K_transposed) / sqrt(d_k)
    if mask is not None:
        scores[mask == 0] = -infinity
    weights = softmax(scores, axis=-1)
    return weights * V, weights
```

The optional `mask` parameter is how we prevent the model from attending to certain positions -- for example, future tokens in autoregressive generation or padding tokens in variable-length batches.

### Why Scaling Matters

This is a detail worth dwelling on because it has practical consequences. Consider \(d_k = 512\). Without scaling, the dot product of two random vectors with that dimensionality has a variance of 512. That means many values in \(QK^\top\) will be large, and softmax will output distributions that are nearly one-hot. When gradients are nearly zero for all but one position, the model cannot learn nuanced attention patterns. The \(1/\sqrt{d_k}\) factor is a simple fix that keeps learning healthy.

### Attention Weights as Interpretability Tools

One practical reason to care about the attention formula is that the weight matrix (the output of softmax) is interpretable. You can visualize which tokens a model is attending to at each layer and head. Tools like BertViz make this straightforward. While attention weights are not a perfect explanation of model behavior, they are a valuable debugging and exploration tool.

## Connecting to Practice

Understanding the attention formula helps in several concrete ways regardless of which framework you use:

- **Debugging unexpected outputs**: If a model is ignoring important context, inspecting attention weights can reveal whether the issue is in the attention pattern or elsewhere.
- **Custom architectures**: If you need to modify attention (add a custom mask, change the scoring function), knowing the formula lets you do it confidently.
- **Efficient inference**: Techniques like KV-caching for autoregressive generation make much more sense once you understand what Q, K, and V actually represent.

In the upcoming hands-on exercises, you will implement multi-head attention and see how these individual attention computations compose into richer representations.

## Further Learning & Resources

### Documentation and Articles

1. [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) -- A visual, step-by-step walkthrough of the Transformer and its attention mechanism. Excellent for building geometric intuition.
2. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) -- The original paper. Section 3.2 covers scaled dot-product attention directly.
3. [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) -- A line-by-line walkthrough of the original Transformer paper with detailed annotations.

### Interactive Resources

1. [BertViz: Attention Visualization Tool](https://github.com/jessevig/bertviz) -- Interactive tool for visualizing attention patterns in pre-trained Transformer models. Helps connect the math to actual model behavior.
2. [Transformer Explainer (Georgia Tech)](https://poloclub.github.io/transformer-explainer/) -- Interactive visualization that lets you step through each component of a Transformer layer and see how data flows through the architecture.
3. [3Blue1Brown: Attention in Transformers (interactive companion)](https://www.3blue1brown.com/lessons/attention) -- Companion page with interactive visuals for building intuition about attention geometry.
