# LSTMs & GRUs

**Estimated Time:** 10 Minutes

---

## Introduction

In the previous reading you saw that vanilla RNNs have a serious shortcoming: they struggle to learn long-range dependencies because gradients vanish as they travel backward through many time steps. This is not a theoretical curiosity -- it means a standard RNN can forget the beginning of a paragraph by the time it reaches the end.

**Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRUs)** were specifically engineered to solve this problem. Both architectures introduce gating mechanisms that give the network fine-grained control over what to remember, what to forget, and what to output at each time step. Understanding how these gates work -- and when to choose one architecture over the other -- is essential for building effective sequence models.

---

## Core Concepts

### Why Gates?

Think of a vanilla RNN's hidden state as a single notebook that gets completely rewritten at every time step. Important information from earlier steps can easily be overwritten by newer, less important information. Gates are like selective erasers and pens: they let the network decide, at each step, which parts of the notebook to keep, which to erase, and which new information to write in.

### The LSTM Architecture

An LSTM replaces the simple hidden-state update of a vanilla RNN with a more elaborate structure built around a **cell state** -- a separate memory channel that runs alongside the hidden state. The cell state is the key innovation: information can flow along it largely unchanged, protected from the vanishing gradient problem by additive (rather than multiplicative) updates.

Three gates control the cell state:

1. **Forget gate** -- looks at the previous hidden state and the current input, then outputs a value between 0 and 1 for each element of the cell state. A value near 0 means "erase this," and a value near 1 means "keep this."

2. **Input gate** -- decides which new information to write into the cell state. It has two parts: a sigmoid that determines *which* cells to update, and a tanh that proposes *candidate values* to write.

3. **Output gate** -- determines which parts of the (now updated) cell state to expose as the hidden state for this time step.

The update equations:

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}
\]
\[
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}
\]
\[
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(candidate)}
\]
\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell state update)}
\]
\[
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}
\]
\[
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}
\]

The critical insight is in the cell state update: \(\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t\). This is an **additive** update. Gradients can flow backward through the addition operation without shrinking, which is how LSTMs maintain long-range memory.

### The GRU Architecture

The GRU, introduced by Cho et al. in 2014, simplifies the LSTM by merging the cell state and hidden state into a single state vector and reducing three gates to two:

1. **Reset gate** -- controls how much of the previous hidden state to "forget" when computing the candidate hidden state.
2. **Update gate** -- controls the balance between keeping the old hidden state and adopting the new candidate. It plays the combined role of the LSTM's forget and input gates.

\[
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) \quad \text{(reset gate)}
\]
\[
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) \quad \text{(update gate)}
\]
\[
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}) \quad \text{(candidate)}
\]
\[
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(state update)}
\]

### LSTM vs. GRU: When to Use Which

| Consideration | LSTM | GRU |
|---|---|---|
| Parameters | More (3 gates + cell state) | Fewer (2 gates, no cell state) |
| Training speed | Slower per step | Faster per step |
| Long sequences (500+ steps) | Often performs better | May struggle on very long dependencies |
| Small datasets | Higher risk of overfitting | More parameter-efficient |
| Default recommendation | When in doubt, start here | When speed matters or data is limited |

In practice, the performance difference between LSTMs and GRUs is often small. Many practitioners start with a GRU for faster iteration and switch to an LSTM only if they observe that the model is failing to capture long-range dependencies.

### Bidirectional Variants

Both LSTMs and GRUs can be made **bidirectional**, meaning a second recurrent layer processes the sequence in reverse. The forward and backward hidden states are concatenated at each time step, giving the network context from both the past and the future. This doubles the output dimension (e.g., a hidden size of 64 produces a 128-dimensional output at each step).

Bidirectional processing is especially useful for tasks where the full sequence is available at inference time, such as named entity recognition or machine translation encoding. It should *not* be used for autoregressive tasks (like language generation) where future tokens are unknown.

---

## Connecting to Practice

In the upcoming exercise, you will take the RNN pipeline you built previously and upgrade it by swapping in LSTM and GRU layers. Because the overall pipeline architecture stays the same (embed, recur, decode), the change is minimal in structure but often dramatic in results -- especially on longer sequences. You will also experiment with bidirectional variants and observe how they affect both accuracy and training time.

---

## Further Learning & Resources

### Documentation

1. [Deep Learning Book, Chapter 10.10: Long Short-Term Memory](https://www.deeplearningbook.org/contents/rnn.html) -- Mathematical treatment of LSTM architecture and its gradient-flow properties.
2. [Wikipedia: Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory) -- Historical context, architecture variants, and applications.
3. [Wikipedia: Gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) -- Overview of GRU architecture, comparison with LSTM, and original references.

### Interactive Resources

1. [Colah's Blog: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) -- The classic illustrated walkthrough of LSTM internals, with clear diagrams of each gate's role.
2. [Dive into Deep Learning: LSTMs (Interactive)](https://d2l.ai/chapter_recurrent-modern/lstm.html) -- Interactive Jupyter-based chapter with runnable code that builds LSTMs from scratch and compares them to vanilla RNNs.
3. [Dive into Deep Learning: GRUs (Interactive)](https://d2l.ai/chapter_recurrent-modern/gru.html) -- Companion interactive chapter that implements GRUs and contrasts their behavior with LSTMs on the same tasks.
