# Transformer Architecture Details

**Estimated Time:** 10 Minutes

## Introduction

Now that you understand the attention mechanism itself, it is time to zoom out and see how the full Transformer architecture fits together. The original Transformer, introduced in 2017, replaced recurrence entirely with attention and a handful of other well-chosen components. The result was a model that trained faster, parallelized better, and scaled to lengths that RNNs struggled with.

This reading focuses on two things: positional encoding (how the model knows word order without recurrence) and multi-head self-attention (how the model attends to different relationship types simultaneously). These are the architectural details that separate the Transformer from a bag-of-words model with attention bolted on.

## Core Concepts

### Why Position Information Is Not Free

Self-attention, by itself, is permutation-invariant. If you shuffle the tokens in a sequence, the attention scores change, but only because the queries and keys moved -- the operation treats every position identically. That means a vanilla attention layer has no concept of "first word" or "last word." Without injecting position information, the sentence "the cat sat on the mat" and "mat the on sat cat the" would be indistinguishable at the attention level.

### Positional Encoding

The original Transformer uses fixed sinusoidal positional encodings. For each position \(\text{pos}\) and each dimension \(i\) of the embedding:

\[
PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
\]

These encodings are **added** to the token embeddings before the first Transformer layer. The result is that each token's representation now carries both semantic information (from the embedding) and positional information (from the encoding).

Why sinusoids? Two key properties: (1) the model can generalize to sequence lengths not seen during training because the functions are defined for any position, and (2) relative positions can be expressed as linear transformations of the encodings, giving the model a way to learn relative distance.

Modern models often use **learned positional embeddings** instead (BERT, GPT-2), or **rotary positional embeddings** (RoPE, used in LLaMA). The core idea is always the same -- inject order information so the model knows where tokens sit in the sequence.

### Multi-Head Self-Attention

A single attention head learns one type of relationship (maybe syntactic adjacency, maybe coreference). Multi-head attention runs several attention heads in parallel, each with its own Q, K, V projections, and concatenates their outputs:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \cdot W^O
\]

where each head is:

\[
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
\]

Concretely, if the model dimension is 768 and there are 12 heads, each head operates on a 64-dimensional subspace (768 / 12 = 64). The heads run independently and their outputs are concatenated back to 768 dimensions, then projected through \(W^O\).

This is computationally equivalent to running \(h\) separate attention operations, but much more efficient because it uses a single large matrix multiplication with reshaping rather than \(h\) separate ones.

### The Rest of the Block

Each Transformer layer also includes:

- **Layer normalization**: Stabilizes training by normalizing activations. Applied either before attention (pre-norm, used by GPT-2 and most modern models) or after (post-norm, used in the original paper).
- **Feed-forward network (FFN)**: A two-layer MLP applied independently to each position. Typically expands the dimension by 4x and then projects back down. This is where much of the model's capacity lives.
- **Residual connections**: The input to each sub-layer is added back to the output. This keeps gradients flowing through deep stacks and lets earlier layers pass information forward unchanged.

The combination -- **attention, add & norm, FFN, add & norm** -- repeats N times to form the full encoder or decoder stack.

### Putting the Numbers Together

When you look at a Transformer model's configuration, the numbers map directly to these components:

| Config Parameter | Meaning | BERT-base Example |
|---|---|---|
| hidden_size | Model dimension (\(d_{\text{model}}\)) | 768 |
| num_attention_heads | Number of heads (\(h\)) | 12 |
| head dimension | \(d_{\text{model}} / h\) | 64 |
| intermediate_size | FFN expansion dimension | 3072 (= 768 x 4) |
| num_hidden_layers | Number of repeated blocks | 12 |
| max_position_embeddings | Maximum sequence length | 512 |

Understanding these mappings demystifies model architectures. They are not magic numbers -- they are direct consequences of the design choices covered above.

## Connecting to Practice

Understanding positional encoding becomes immediately practical when you hit sequence length limits. BERT's maximum is 512 tokens because it uses learned positional embeddings of that size. Models with rotary embeddings can often extrapolate further. Knowing which encoding your model uses tells you what to expect when your inputs approach or exceed the limit.

The multi-head structure explains why attention visualizations show different patterns across heads -- each head is free to specialize in a different type of linguistic relationship.

## Further Learning & Resources

### Documentation and Articles

1. [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) -- A line-by-line walkthrough of the original Transformer paper with detailed annotations connecting paper math to implementation.
2. [Positional Encoding explained by Amirhossein Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) -- Deep dive into why sinusoidal encodings work, with visualizations of the frequency patterns.
3. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) -- The original paper. Sections 3.1-3.3 cover multi-head attention and positional encoding directly.

### Interactive Resources

1. [Transformer Explainer (Georgia Tech)](https://poloclub.github.io/transformer-explainer/) -- Interactive visualization that lets you step through each component of a Transformer layer and see how data flows through the architecture.
2. [3Blue1Brown: Attention in Transformers (interactive companion)](https://www.3blue1brown.com/lessons/attention) -- Companion page with interactive visuals for building intuition about attention geometry.
3. [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- Visual walkthrough with diagrams of positional encoding, multi-head attention, and the full Transformer block.
