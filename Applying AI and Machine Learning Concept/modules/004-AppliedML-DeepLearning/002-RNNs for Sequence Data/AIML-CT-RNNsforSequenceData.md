# RNNs for Sequence Data

**Estimated Time:** 10 Minutes

---

## Introduction

Not all data comes neatly arranged in a grid. Text, audio, stock prices, sensor readings, and DNA sequences all share a common trait: **order matters**. Shuffling the words in a sentence or rearranging the samples in a time series destroys the meaning. Convolutional Neural Networks, powerful as they are for images, do not naturally capture this sequential dependency. That is the gap **Recurrent Neural Networks (RNNs)** were designed to fill.

In this reading you will learn how RNNs process sequences one step at a time, how they maintain a "memory" of past inputs through a hidden state, and why their practical limitations motivated the more advanced architectures covered in the next reading.

---

## Core Concepts

### The Core Idea: Memory Through Recurrence

A feedforward network processes each input independently -- it has no concept of "what came before." An RNN adds a feedback loop. At each time step \(t\), the network receives two things: the current input \(\mathbf{x}_t\) and the hidden state \(\mathbf{h}_{t-1}\) from the previous step. It combines them to produce a new hidden state \(\mathbf{h}_t\), which then serves as memory for the next step.

Mathematically:

\[
\mathbf{h}_t = f(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})
\]

where \(\mathbf{W}_{hh}\) and \(\mathbf{W}_{xh}\) are weight matrices, \(\mathbf{b}\) is a bias, and \(f\) is typically tanh. The hidden state \(\mathbf{h}_t\) is a compressed summary of everything the network has seen up to time step \(t\).

### Key Properties of the RNN

- **Weight sharing across time:** The same weight matrices \(\mathbf{W}_{hh}\) and \(\mathbf{W}_{xh}\) are reused at every time step. This means the number of parameters does not grow with sequence length.
- **Variable-length inputs:** An RNN can process sequences of any length without architectural changes.
- **Two useful outputs:** The hidden state at *every* time step (useful for per-step predictions like tagging each word) and the *final* hidden state (useful for whole-sequence tasks like classifying a review as positive or negative).

### Architecting an RNN Sequence Pipeline

A complete RNN pipeline for sequence classification typically has three components:

```
1. Embedding layer: converts discrete tokens (words, characters) into dense vectors
2. RNN layer(s): processes the sequence and produces hidden states
3. Output head: takes the final hidden state and maps it to class logits
```

In pseudocode:

```
function SequenceClassifier(input_tokens):
    embedded = Embedding(input_tokens)          # (batch, seq_len, embed_dim)
    all_hidden, final_hidden = RNN(embedded)    # final_hidden: (batch, hidden_dim)
    logits = Linear(final_hidden)               # (batch, num_classes)
    return logits
```

This pattern -- **embed, recur, decode** -- is the backbone of most RNN-based sequence models. You will see it repeated with LSTMs and GRUs in the next reading.

### Sequence Padding and Batching

Real-world sequences rarely have the same length. In a batch of sentences, some may be 5 words long and others 50. To form a rectangular tensor for efficient computation, shorter sequences are **padded** with zeros (or a special padding token) to match the longest sequence in the batch.

However, padding introduces a problem: the RNN processes padding tokens as if they were real input, which can pollute the hidden state. The solution is **packing** -- telling the RNN which positions are real tokens and which are padding, so it can skip the padding during computation. Every major deep learning framework provides utilities for this.

Getting padding and packing right is one of the most common stumbling blocks when working with RNNs. Take the time to understand it now, and you will save yourself considerable debugging later.

### The Vanishing Gradient Problem (Again)

You encountered the vanishing gradient problem with deep feedforward networks and sigmoid activations. RNNs suffer from the same issue, but it manifests across **time steps** rather than across layers. During **backpropagation through time (BPTT)**, gradients must flow backward through every step in the sequence. For long sequences (hundreds or thousands of steps), the gradient signal shrinks exponentially, and the network effectively "forgets" the early parts of the sequence.

This is not a minor inconvenience -- it means a vanilla RNN struggles to learn dependencies that span more than about 10-20 time steps. If the word that determines the sentiment of a review appears at the very beginning and the review is 200 words long, a plain RNN is unlikely to capture that connection.

This fundamental limitation is exactly what LSTMs and GRUs were invented to address, which you will explore in the next reading.

---

## Connecting to Practice

In the upcoming exercise, you will build an RNN-based pipeline from scratch: tokenizing text, embedding it, running it through recurrent layers, and producing predictions. The pipeline pattern you learn here -- embed, recur, decode -- is not specific to vanilla RNNs. When you swap in an LSTM or GRU later, the surrounding architecture stays nearly identical. Mastering this structure now means you can upgrade components without redesigning the whole system.

---

## Further Learning & Resources

### Documentation

1. [Understanding BPTT (Backpropagation Through Time)](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) -- The Dive into Deep Learning textbook's clear walkthrough of how gradients flow through recurrent networks.
2. [Deep Learning Book, Chapter 10: Sequence Modeling](https://www.deeplearningbook.org/contents/rnn.html) -- Goodfellow, Bengio, and Courville's comprehensive treatment of recurrent architectures.
3. [Wikipedia: Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) -- Historical context, mathematical formulations, and architecture variants.

### Interactive Resources

1. [Dive into Deep Learning: Recurrent Neural Networks (Interactive)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) -- Interactive Jupyter-based chapter with runnable code that builds RNNs step by step.
2. [Distill.pub: Memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/) -- Beautifully illustrated interactive article exploring what RNNs memorize and how hidden states evolve.
3. [Google Machine Learning Crash Course: Sequence Models](https://developers.google.com/machine-learning/crash-course/sequences) -- Guided module with embedded exercises on sequence modeling fundamentals.
