# AIML-LEC-AppliedML-Transformers

**Activity Type:** Lecture / Conceptual Live Demo  
**Duration:** 180 minutes (3 hours)  
**Module:** 005 -- Applied ML: Transformers  
**Prerequisites:** Students have completed pre-class readings and videos on attention mechanisms, transformer architecture, BERT/GPT/T5 use cases, and transfer learning.

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Explain scaled dot-product attention and why scaling is necessary.
2. Explain positional encoding and why transformers need it.
3. Describe the components of a multi-head self-attention module using pseudocode and linear algebra.
4. Differentiate BERT, GPT, and T5 by architecture, pre-training objective, and use case.
5. Outline a transfer learning pipeline: pre-trained model selection, tokenization, task head, fine-tuning.
6. Explain the fine-tuning process for sentiment classification at a conceptual level.
7. (Stretch) Explain parameter-efficient fine-tuning (LoRA) and why it reduces cost.
8. (Stretch) Describe GPT-style text generation sampling strategies and their trade-offs.

---

## Tech Stack

| Tool | Version / Notes |
|------|-----------------|
| Python | 3.10+ |
| NumPy | latest stable |
| matplotlib | latest stable |
| Whiteboard / Slides | For diagrams and pseudocode walkthroughs |

> **Note:** This lecture is framework-agnostic. All model-building and fine-tuning concepts are presented as pseudocode and math notation so students learn the underlying principles independent of any specific library.

---

## Preparation Checklist (Instructor)

- [ ] Verify Python 3.10+, numpy, and matplotlib installed in demo environment.
- [ ] Prepare whiteboard or slide diagrams for: attention matrix visualization, transformer block diagram, encoder-vs-decoder comparison.
- [ ] Have the sinusoidal positional encoding NumPy visualization script ready to run.
- [ ] Have the scaling-effect NumPy demo script ready to run.
- [ ] Clone the lecture repo; confirm the starter branch `lecture/stage-0-starter` exists.
- [ ] Test projector / screen-share for live demos.

---

## Scenario

Students will study an **NLP pipeline** across three conceptual stages. In Stage 1 they work through attention and transformer components from first principles using math and pseudocode. In Stage 2 they explore pre-trained model architectures (BERT, GPT, T5), comparing their design choices, pre-training objectives, and strengths. In Stage 3 they walk through a full transfer learning pipeline for binary sentiment classification, understanding each step conceptually before ever touching a specific framework.

---

## Git Branch Strategy

| Branch | Purpose |
|--------|---------|
| `lecture/stage-0-starter` | Empty scaffold with notes only |
| `lecture/stage-1-attention` | Completed Stage 1 pseudocode and visualizations |
| `lecture/stage-2-pretrained` | Completed Stage 2 architecture comparison notes |
| `lecture/stage-3-finetune` | Completed Stage 3 pipeline walkthrough (final) |

Students should check out `lecture/stage-0-starter` at the start. At the end of each stage the instructor can show the corresponding branch as a reference checkpoint.

---

# STAGE 1 -- Attention and Transformer Components from Scratch (55 min)

> **Goal:** Understand why attention was invented, derive scaled dot-product attention and multi-head self-attention from math and pseudocode, and build positional encoding.

## STEP 1.1 -- Why Attention? Motivation from Sequence Models (10 min)

**[PACING: Conceptual; no code yet. Build the "why" before the "how".]**

- Recap the bottleneck in encoder-decoder RNNs: the entire input sequence is compressed into a single fixed-length context vector. Long sequences lose information.
- Attention was introduced (Bahdanau et al., 2014) so the decoder can look back at every encoder hidden state, not just the last one.
- The transformer (Vaswani et al., 2017) removed recurrence entirely and replaced it with **self-attention**: every position attends to every other position in parallel.
- Key advantages: parallelization (no sequential dependence), direct long-range connections, and the ability to learn which parts of the input matter for each output position.

Draw or display this progression on the board:

```
RNN Encoder-Decoder  -->  RNN + Attention  -->  Transformer (Self-Attention Only)
(fixed context vector)    (attend to all       (no recurrence; all attention)
                           encoder states)
```

**Discussion Prompt:** "If an RNN processes tokens left to right, what happens to the gradient signal from the first token by the time you reach token 500? How does attention sidestep this?"

---

## STEP 1.2 -- Scaled Dot-Product Attention (18 min)

**[PACING: Derive the math on the board, then demonstrate with NumPy.]**

### The Formula (5 min)

Write on board:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Walk through each component:
- **Q** (queries): "what am I looking for?" -- shape (seq_len, d_k)
- **K** (keys): "what do I contain?" -- shape (seq_len, d_k)
- **V** (values): "what information do I provide?" -- shape (seq_len, d_v)
- **Q @ K^T**: dot-product similarity between every query and every key -- shape (seq_len, seq_len)
- **/ sqrt(d_k)**: scaling factor. Without it, large d_k pushes dot products into regions where softmax saturates and gradients vanish.
- **softmax**: converts raw scores to a probability distribution over positions.
- **@ V**: weighted sum of values according to attention weights.

### Pseudocode

```
FUNCTION scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = number_of_columns(Q)
    scores = matrix_multiply(Q, transpose(K)) / sqrt(d_k)

    IF mask IS NOT None:
        SET scores WHERE mask IS True TO -infinity

    attention_weights = softmax(scores, axis=-1)    // each row sums to 1
    output = matrix_multiply(attention_weights, V)
    RETURN output, attention_weights
```

### NumPy Demonstration (13 min)

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention using NumPy.

    Args:
        Q: Queries, shape (..., seq_len_q, d_k)
        K: Keys, shape (..., seq_len_k, d_k)
        V: Values, shape (..., seq_len_k, d_v)
        mask: Optional boolean mask, shape broadcastable to (..., seq_len_q, seq_len_k).
              Positions with True are masked (set to -inf before softmax).

    Returns:
        output: Weighted values, shape (..., seq_len_q, d_v)
        attention_weights: Softmax weights, shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V
    return output, attention_weights

# Demo: 4-token sequence, d_k = 8
np.random.seed(42)
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")                    # (4, 8)
print(f"Attention weights shape: {weights.shape}")         # (4, 4)
print(f"Weights sum per query (should be ~1.0): {weights.sum(axis=-1)}")

# Visualize the attention matrix
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(weights, cmap="Blues")
ax.set_xlabel("Key position")
ax.set_ylabel("Query position")
ax.set_title("Attention Weights")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
```

Key points to narrate:
- Each row of the attention matrix sums to 1 (probability distribution over keys).
- The output for position i is a weighted combination of all value vectors, where the weights are determined by how much query i "matches" each key.
- The mask argument is essential for causal (autoregressive) models like GPT, where position i must not attend to positions j > i.

### Demonstrate the scaling effect

```python
# Why scaling matters
d_k_small, d_k_large = 8, 512
np.random.seed(42)
Q_s = np.random.randn(4, d_k_small)
K_s = np.random.randn(4, d_k_small)
Q_l = np.random.randn(4, d_k_large)
K_l = np.random.randn(4, d_k_large)

scores_small = Q_s @ K_s.T
scores_large = Q_l @ K_l.T

print(f"Unscaled scores (d_k=8), std:   {scores_small.std():.2f}")
print(f"Unscaled scores (d_k=512), std: {scores_large.std():.2f}")
print(f"Scaled scores (d_k=512), std:   {(scores_large / np.sqrt(512)).std():.2f}")
```

Explain: when d_k is large, unscaled dot products have high variance, pushing softmax into saturated regions where most weight goes to one position and gradients are near zero.

---

## STEP 1.3 -- Multi-Head Self-Attention (15 min)

**[PACING: Pseudocode on board. Build on the previous function.]**

### Math Notation

For \(h\) attention heads with model dimension \(d_{model}\) and head dimension \(d_k = d_{model} / h\):

```
For each head i:
    Q_i = X @ W_q_i        // project input to queries for head i
    K_i = X @ W_k_i        // project input to keys for head i
    V_i = X @ W_v_i        // project input to values for head i
    head_i = Attention(Q_i, K_i, V_i)

MultiHead(X) = Concatenate(head_1, ..., head_h) @ W_o
```

### Pseudocode

```
CLASS MultiHeadSelfAttention:
    PARAMETERS:
        W_q: linear projection  (d_model -> d_model)
        W_k: linear projection  (d_model -> d_model)
        W_v: linear projection  (d_model -> d_model)
        W_o: linear projection  (d_model -> d_model)

    INIT(d_model, num_heads):
        ASSERT d_model IS divisible BY num_heads
        d_k = d_model / num_heads
        INITIALIZE W_q, W_k, W_v, W_o as learnable weight matrices

    FORWARD(X, mask=None):
        // X has shape (batch_size, seq_len, d_model)
        batch_size, seq_len = shape(X)[0], shape(X)[1]

        // Step 1: Project to Q, K, V
        Q = X @ W_q      // (batch, seq_len, d_model)
        K = X @ W_k
        V = X @ W_v

        // Step 2: Reshape to separate heads
        Q = reshape(Q, [batch_size, seq_len, num_heads, d_k])
        Q = transpose(Q, axes=[0, 2, 1, 3])    // (batch, num_heads, seq_len, d_k)
        // same for K, V

        // Step 3: Apply scaled dot-product attention per head
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        // Step 4: Concatenate heads
        output = transpose(output, axes=[0, 2, 1, 3])
        output = reshape(output, [batch_size, seq_len, d_model])

        // Step 5: Final linear projection
        output = output @ W_o
        RETURN output, attention_weights
```

### Worked Example (on board)

```
Given: d_model = 64, num_heads = 8, batch_size = 2, seq_len = 10

d_k = 64 / 8 = 8  (each head works in 8 dimensions)

Input X:          (2, 10, 64)
After Q = X @ W_q: (2, 10, 64)
After reshape:      (2, 8, 10, 8)   -- 8 heads, each with d_k=8
Attention output:   (2, 8, 10, 8)
After concatenate:  (2, 10, 64)     -- heads reassembled
After W_o:          (2, 10, 64)     -- same shape as input

Learnable parameters: 4 × (64 × 64) = 16,384
```

Explain the design:
- Each head operates on a d_k-dimensional slice of the representation, learning different attention patterns.
- The four linear projections (W_q, W_k, W_v, W_o) are the learnable parameters.
- The reshape-transpose-reshape pattern is the standard way to split and reassemble heads efficiently without explicit loops.
- W_o recombines the concatenated head outputs into the original d_model dimension.

**Discussion Prompt:** "Why might multiple heads be better than one head with the same total dimensionality?" (Different heads can attend to different types of relationships: syntactic, semantic, positional.)

---

## STEP 1.4 -- Positional Encoding (10 min)

**[PACING: Conceptual motivation, then NumPy visualization.]**

- Self-attention is **permutation invariant**: shuffling the input tokens produces shuffled outputs with the same attention weights. The model has no inherent notion of order.
- Positional encoding injects sequence-order information by adding a position-dependent signal to each token embedding.
- The original transformer uses sinusoidal positional encoding with different frequencies for each dimension.

### Math Notation

For position \(pos\) and dimension \(i\):

```
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

The positional encoding is **added** to the token embedding:

```
input_to_transformer = token_embedding(x) + PE
```

### NumPy Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encoding matrix.

    Returns:
        pe: shape (max_len, d_model)
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

encoding = sinusoidal_positional_encoding(max_len=50, d_model=64)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(encoding.T, cmap="RdBu", aspect="auto")
ax.set_xlabel("Position")
ax.set_ylabel("Dimension")
ax.set_title("Sinusoidal Positional Encoding (d_model=64)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
```

Key points:
- Low-frequency sinusoids (high dimensions) capture long-range position differences; high-frequency sinusoids (low dimensions) capture fine-grained position.
- The encoding is deterministic (no learned parameters), which means it generalizes to sequence lengths not seen during training.
- Modern models (BERT, GPT-2) often use **learned** positional embeddings instead, but the sinusoidal version remains a strong baseline and is easier to reason about.

**Q&A Checkpoint (2-3 min):** "If we removed positional encoding entirely, could a transformer still distinguish 'dog bites man' from 'man bites dog'?"

---

**[BREAK -- 5 min]**

> Students should review all Stage 1 material. Check out `lecture/stage-1-attention` if behind.

---

# STAGE 2 -- Exploring Pre-trained Models: BERT, GPT, T5 (55 min)

> **Goal:** Understand how BERT, GPT, and T5 differ in architecture and pre-training objective. Explore each model's design, capabilities, and best-fit use cases.

## STEP 2.1 -- Transformer Encoder vs. Decoder Recap (8 min)

**[PACING: Conceptual. Draw the diagrams side by side.]**

| Property | Encoder (BERT-style) | Decoder (GPT-style) | Encoder-Decoder (T5-style) |
|----------|---------------------|---------------------|---------------------------|
| Attention | Bidirectional (sees full sequence) | Causal (sees only past tokens) | Encoder is bidirectional; decoder is causal with cross-attention to encoder |
| Pre-training | Masked Language Modeling (fill in blanks) | Autoregressive LM (predict next token) | Span corruption (predict masked spans) |
| Best for | Classification, NER, sentence similarity | Text generation, dialogue, code | Translation, summarization, question answering |

Key architectural difference:
- **Encoder-only:** No causal mask. Every token can attend to every other token. Good for understanding.
- **Decoder-only:** Causal mask prevents attending to future tokens. Good for generation.
- **Encoder-decoder:** Encoder processes input bidirectionally; decoder generates output autoregressively with cross-attention to encoder states.

---

## STEP 2.2 -- BERT: Bidirectional Understanding (15 min)

**[PACING: Architecture walkthrough with diagrams and pseudocode.]**

### Architecture

```
BERT Architecture:
    Input: [CLS] token_1 token_2 ... token_n [SEP]

    1. Token Embedding:     lookup each token → d_model-dim vector
    2. Position Embedding:  learned embedding for each position
    3. Segment Embedding:   distinguishes sentence A from sentence B

    Embedding = Token_Emb + Position_Emb + Segment_Emb

    4. Pass through N transformer ENCODER layers:
       FOR each layer:
           x = LayerNorm(x + MultiHeadSelfAttention(x))     // bidirectional
           x = LayerNorm(x + FeedForward(x))

    Output: one contextualized vector per token, shape (seq_len, d_model)
```

### Key Model Facts (bert-base-uncased)

```
Parameters:      ~110 million
Hidden size:     768
Layers:          12 transformer encoder blocks
Attention heads: 12
Vocabulary:      ~30,000 (WordPiece tokenizer)
```

### Pre-training Objective: Masked Language Modeling (MLM)

```
Training example:
    Input:  "The [MASK] sat on the [MASK]."
    Target: "The  cat  sat on the  mat ."

    - Randomly mask 15% of tokens
    - Model must predict the original token at each masked position
    - Because attention is bidirectional, the model uses BOTH left and right context
```

### Conceptual Exercise: What BERT Produces

```
Input sentence: "The transformer architecture revolutionized NLP."

After tokenization (WordPiece):
    [CLS]  the  transform  ##er  architecture  revolution  ##ized  nl  ##p  .  [SEP]

After BERT forward pass:
    Output shape: (1, 11, 768)   -- one 768-dim vector per token

    The [CLS] vector is commonly used as a "sentence representation"
    for downstream classification tasks.
```

### Masked LM Conceptual Demo

```
Input:    "The [MASK] sat on the mat."
Top predictions (with confidence):
    cat        (score: 0.45)
    dog        (score: 0.12)
    boy        (score: 0.08)

BERT predicts by looking at BOTH sides of the mask -- "The __ sat on the mat"
gives strong contextual signal from both directions.
```

Narrate:
- BERT sees the **entire** sentence in both directions (bidirectional attention). This is why it excels at tasks where understanding context on both sides matters.
- The `[MASK]` token is central to pre-training: BERT learns by predicting randomly masked tokens.
- The last hidden state gives one contextualized embedding per token. For classification, we typically use the `[CLS]` token embedding or pool across all tokens.
- BERT cannot generate text because it was not trained autoregressively.

---

## STEP 2.3 -- GPT: Autoregressive Generation (15 min)

**[PACING: Architecture walkthrough. Contrast with BERT.]**

### Architecture

```
GPT Architecture:
    Input: token_1 token_2 ... token_n

    1. Token Embedding:     lookup each token → d_model-dim vector
    2. Position Embedding:  learned embedding for each position

    Embedding = Token_Emb + Position_Emb

    3. Pass through N transformer DECODER layers:
       FOR each layer:
           x = LayerNorm(x + CAUSAL_MultiHeadSelfAttention(x))  // masked: can only see past
           x = LayerNorm(x + FeedForward(x))

    4. Language Model Head:
       logits = x @ Embedding_Matrix^T       // project back to vocabulary
       next_token_probs = softmax(logits)     // probability over vocab
```

### Key Model Facts (GPT-2)

```
Parameters:      ~124 million (small), up to 1.5B (XL)
Hidden size:     768
Layers:          12 transformer decoder blocks
Attention heads: 12
Vocabulary:      ~50,257 (Byte-Pair Encoding tokenizer)
```

### Causal Masking

```
Attention mask for a 5-token sequence:

Token:    t1  t2  t3  t4  t5
t1 sees:  [✓] [✗] [✗] [✗] [✗]     ← only itself
t2 sees:  [✓] [✓] [✗] [✗] [✗]     ← t1, t2
t3 sees:  [✓] [✓] [✓] [✗] [✗]     ← t1, t2, t3
t4 sees:  [✓] [✓] [✓] [✓] [✗]     ← t1 through t4
t5 sees:  [✓] [✓] [✓] [✓] [✓]     ← all tokens

This lower-triangular mask ensures token i can only attend to tokens 0..i,
which is what makes left-to-right generation possible.
```

### Autoregressive Generation (Conceptual)

```
FUNCTION generate(prompt, max_new_tokens):
    tokens = tokenize(prompt)
    FOR step IN 1..max_new_tokens:
        logits = model_forward(tokens)          // shape: (seq_len, vocab_size)
        next_token_logits = logits[-1]          // only the last position matters
        next_token = sample(next_token_logits)  // sampling strategy applied here
        tokens = append(tokens, next_token)
        IF next_token == END_OF_SEQUENCE:
            BREAK
    RETURN detokenize(tokens)
```

Narrate:
- GPT uses a **causal mask**: token i can only attend to tokens 0..i. This is what makes left-to-right generation possible.
- Pre-training objective: predict the next token given all previous tokens.
- GPT is not well-suited for classification without fine-tuning because it was trained to generate, not to understand bidirectionally.

**Discussion Prompt:** "Why can't we just use GPT for everything, including classification? What does BERT's bidirectional attention give us that GPT's causal attention does not?"

---

## STEP 2.4 -- T5: Text-to-Text Framework (12 min)

**[PACING: Architecture walkthrough. Show the unifying framework.]**

### Architecture

```
T5 Architecture (Encoder-Decoder):

    ENCODER (bidirectional):
        Input: "translate English to French: The house is wonderful."

        FOR each encoder layer:
            x = LayerNorm(x + MultiHeadSelfAttention(x))   // bidirectional
            x = LayerNorm(x + FeedForward(x))

        encoder_output = x    // rich bidirectional representation of input

    DECODER (causal + cross-attention):
        Input: <start> La maison est ...

        FOR each decoder layer:
            x = LayerNorm(x + CAUSAL_MultiHeadSelfAttention(x))
            x = LayerNorm(x + CrossAttention(x, encoder_output))  // attend to encoder
            x = LayerNorm(x + FeedForward(x))

        logits = x @ Embedding_Matrix^T
        next_token_probs = softmax(logits)
```

### Key Model Facts (T5-small)

```
Parameters:      ~60 million
Encoder layers:  6
Decoder layers:  6
Hidden size:     512
Attention heads: 8
Vocabulary:      ~32,000 (SentencePiece tokenizer)
```

### The Text-to-Text Paradigm

T5 frames **every** NLP task as "text in, text out":

```
[Translation]
    Input:  "translate English to French: The house is wonderful."
    Output: "La maison est merveilleuse."

[Summarization]
    Input:  "summarize: Transformers are a class of deep learning models..."
    Output: "Transformers use self-attention for parallel sequence processing."

[Sentiment Classification]
    Input:  "sst2 sentence: This movie was absolutely fantastic!"
    Output: "positive"

[Question Answering]
    Input:  "question: What is the capital of France? context: France is a country in Europe. Its capital is Paris."
    Output: "Paris"
```

Narrate:
- T5 frames **every** NLP task as "text in, text out." Classification becomes "is this positive or negative?" with the answer as generated text.
- The prefix (e.g., "translate English to French:") tells T5 which task to perform.
- Architecture: full encoder-decoder. The encoder reads the input bidirectionally; the decoder generates the output autoregressively.
- T5 was pre-trained with **span corruption**: random spans of text are replaced with sentinel tokens, and the model learns to reconstruct them.

### Summary Comparison

| | BERT | GPT-2 | T5-small |
|---|---|---|---|
| **Architecture** | Encoder-only | Decoder-only | Encoder-Decoder |
| **Attention** | Bidirectional | Causal (left-to-right) | Encoder: bidir, Decoder: causal |
| **Pre-training** | Masked LM | Next-token prediction | Span corruption |
| **Strengths** | Classification, NER | Text generation | Multi-task: translate, summarize, classify |

**Q&A Checkpoint (5 min):** "Given a task -- say, building a chatbot that answers questions about a product manual -- which architecture would you start with and why?"

---

**[BREAK -- 5 min]**

> Students should review Stage 2 architecture comparisons. Check out `lecture/stage-2-pretrained` if behind.

---

# STAGE 3 -- Transfer Learning and Fine-tuning (55 min)

> **Goal:** Walk through a complete transfer learning pipeline for binary sentiment classification, understanding each step conceptually. Stretch goals cover parameter-efficient fine-tuning (LoRA) and generation strategies.

## STEP 3.1 -- Transfer Learning Pipeline (12 min)

**[PACING: Conceptual overview of the full pipeline.]**

### Why Transfer Learning?

- Pre-trained models have already learned rich language representations from massive corpora.
- Fine-tuning adapts these representations to a specific task with relatively little labeled data and compute.
- The pipeline: (1) choose a pre-trained model, (2) load and tokenize your task-specific data, (3) add a task head (e.g., classification layer), (4) fine-tune on your data.

### The Full Pipeline (Pseudocode)

```
TRANSFER LEARNING PIPELINE:

// ─── STEP 1: Choose a pre-trained model ───
model_name = "distilbert-base-uncased"     // smaller, faster variant of BERT
// Why DistilBERT? 66M params vs 110M for BERT; similar accuracy, 60% faster

// ─── STEP 2: Load and tokenize task-specific data ───
dataset = load_dataset("imdb")
// IMDB: 25,000 train reviews, 25,000 test reviews, binary labels (pos/neg)

sample = dataset["train"][0]
// sample["text"]  = "This movie was absolutely wonderful..."
// sample["label"] = 1  (positive)

// ─── STEP 3: Tokenization ───
FUNCTION tokenize(text):
    tokens = tokenizer.tokenize(text)
    // "This movie was great" → ["this", "movie", "was", "great"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    // ["this", "movie", "was", "great"] → [2023, 3185, 2001, 2307]
    APPLY truncation to max_length (e.g., 256)
    APPLY padding to max_length
    RETURN input_ids, attention_mask

tokenized_dataset = dataset.map(tokenize, batched=True)

// ─── STEP 4: Subset for training speed ───
train_data = tokenized_dataset["train"].shuffle().select(2000 samples)
test_data  = tokenized_dataset["test"].shuffle().select(500 samples)
```

Narrate:
- Truncation to max_length=256 keeps sequences manageable. Real IMDB reviews can be thousands of tokens.
- We subset the data for lecture speed; in production you would use the full dataset.
- The tokenizer is tied to the pre-trained model; always use the matching tokenizer.

---

## STEP 3.2 -- Fine-tuning for Sentiment Classification (22 min)

**[PACING: Walk through pseudocode step by step. This is the core deliverable of Stage 3.]**

### Model Architecture for Classification

```
CLASSIFICATION MODEL = Pre-trained Transformer + Task Head

┌──────────────────────────────────┐
│     Classification Head          │
│   (Linear: d_model → num_labels) │    ← NEW: randomly initialized
├──────────────────────────────────┤
│                                  │
│   Pre-trained Transformer        │
│   (e.g., DistilBERT: 6 layers,  │    ← LOADED: from pre-trained weights
│    768 hidden, 12 heads)         │
│                                  │
├──────────────────────────────────┤
│   Token + Position Embeddings    │    ← LOADED: from pre-trained weights
└──────────────────────────────────┘

Input: token IDs → Embeddings → Transformer → [CLS] vector → Linear → logits
```

### Fine-tuning Pseudocode

```
// ─── Model Setup ───
model = load_pretrained_classification_model(model_name, num_labels=2)
// This loads pre-trained weights AND adds a randomly initialized classification head

total_parameters = count_parameters(model)     // ~67 million
print("Total parameters:", total_parameters)

// ─── Training Configuration ───
optimizer = AdamW(model.parameters, learning_rate=2e-5, weight_decay=0.01)
num_epochs = 3
batch_size = 16

// ─── Training Loop ───
FOR epoch IN 1..num_epochs:
    model.set_mode(TRAINING)
    total_loss = 0
    correct = 0
    total = 0

    FOR batch IN create_batches(train_data, batch_size, shuffle=True):
        // batch contains: input_ids, attention_mask, labels

        // Forward pass
        outputs = model.forward(batch.input_ids, batch.attention_mask)
        logits = outputs.logits              // shape: (batch_size, 2)
        loss = cross_entropy(logits, batch.labels)

        // Backward pass
        optimizer.zero_gradients()
        loss.backward()                      // compute gradients
        optimizer.step()                     // update weights

        // Track metrics
        total_loss += loss.value * batch.size
        predictions = argmax(logits, axis=-1)
        correct += count(predictions == batch.labels)
        total += batch.size

    train_accuracy = correct / total
    avg_loss = total_loss / total

    // ─── Evaluation ───
    model.set_mode(EVALUATION)
    eval_correct = 0
    eval_total = 0

    FOR batch IN create_batches(test_data, batch_size=32, shuffle=False):
        WITH no_gradient_tracking:
            outputs = model.forward(batch.input_ids, batch.attention_mask)
            predictions = argmax(outputs.logits, axis=-1)
            eval_correct += count(predictions == batch.labels)
            eval_total += batch.size

    test_accuracy = eval_correct / eval_total
    PRINT "Epoch {epoch} | Loss: {avg_loss} | Train Acc: {train_accuracy} | Test Acc: {test_accuracy}"

// Expected output (approximate):
// Epoch 1 | Loss: 0.42 | Train Acc: 0.81 | Test Acc: 0.86
// Epoch 2 | Loss: 0.25 | Train Acc: 0.91 | Test Acc: 0.88
// Epoch 3 | Loss: 0.15 | Train Acc: 0.95 | Test Acc: 0.89
```

Walk through the key design decisions:
1. **Classification head** adds a linear layer on top of the pre-trained transformer. The rest of the model is initialized from pre-trained weights.
2. **Learning rate 2e-5** is much lower than training from scratch. Pre-trained weights are already good; large updates would destroy them.
3. **AdamW with weight decay** is the standard optimizer for transformer fine-tuning (decoupled weight decay regularization).
4. **Cross-entropy loss** is computed from the logits and integer class labels.
5. Only 3 epochs because the pre-trained model already understands language; fine-tuning just teaches it the task-specific mapping.

### Inference (Conceptual)

```
FUNCTION predict_sentiment(model, tokenizer, text):
    input_ids = tokenizer.encode(text, max_length=256, truncation=True)
    model.set_mode(EVALUATION)
    WITH no_gradient_tracking:
        logits = model.forward(input_ids).logits    // shape: (1, 2)
    probabilities = softmax(logits, axis=-1)
    prediction = argmax(logits)
    label = "POSITIVE" IF prediction == 1 ELSE "NEGATIVE"
    confidence = probabilities[prediction]
    RETURN label, confidence

// Example predictions from a fine-tuned model:
// [POSITIVE] (confidence: 97.3%) -- "This film was an absolute masterpiece."
// [NEGATIVE] (confidence: 95.1%) -- "Terrible movie. Waste of time."
// [NEGATIVE] (confidence: 62.4%) -- "It was okay. Some parts were good, others dragged."
```

---

## STEP 3.3 -- (Stretch) Parameter-Efficient Fine-tuning: LoRA (10 min)

**[PACING: Conceptual. Only if time permits.]**

- Full fine-tuning updates all model parameters. For large models this is expensive in memory and storage.
- **LoRA (Low-Rank Adaptation)** freezes the pre-trained weights and injects small trainable rank-decomposition matrices into each attention layer.
- Instead of updating W directly (d × d), LoRA learns A (d × r) and B (r × d) where r << d, so the update is W + AB.
- Reduces trainable parameters by 10-100x while maintaining accuracy.

### How LoRA Works

```
STANDARD FINE-TUNING:
    W_new = W_pretrained + ΔW          // ΔW is (d × d) = d² parameters
    For d=768: 768 × 768 = 589,824 trainable parameters per weight matrix

LoRA FINE-TUNING:
    W_new = W_pretrained + A @ B       // A is (d × r), B is (r × d)
    W_pretrained is FROZEN (no gradients)
    Only A and B are trained

    For d=768, r=8:
        A: 768 × 8  =  6,144 parameters
        B: 8 × 768  =  6,144 parameters
        Total:         12,288 parameters  (2.1% of full update)
```

### LoRA Configuration (Conceptual)

```
LORA CONFIGURATION:
    task_type    = SEQUENCE_CLASSIFICATION
    rank (r)     = 8               // rank of decomposition
    alpha        = 32              // scaling factor (applied as alpha/r)
    dropout      = 0.1
    target       = [W_q, W_v]     // which weight matrices to apply LoRA to

RESULT:
    Trainable parameters:    ~300K
    Total parameters:        ~67M
    Percentage trainable:    0.45%
```

### Parameter Comparison

```python
# LoRA savings calculation (can run with plain Python)
d, r = 768, 8
full_params = d * d
lora_params = d * r + r * d
print(f"Full weight matrix: {full_params:,} parameters")
print(f"LoRA (rank {r}):    {lora_params:,} parameters ({lora_params/full_params:.1%} of full)")
```

Note: **QLoRA** takes this further by quantizing the frozen weights to 4-bit precision, reducing memory even more.

---

## STEP 3.4 -- (Stretch) GPT-Style Text Generation Strategies (10 min)

**[PACING: Walk through the concepts on the board. Show the trade-offs.]**

### Sampling Strategies

```
GENERATION PIPELINE:
    prompt = "The future of artificial intelligence is"
    tokens = tokenize(prompt)

    FOR each new token:
        logits = model.forward(tokens)[-1]       // logits for next position
        logits = apply_strategy(logits)           // modify logits based on strategy
        next_token = sample_from(softmax(logits))
        tokens = append(tokens, next_token)
```

### Strategy Comparison

| Strategy | How It Works | Effect |
|----------|-------------|--------|
| **Greedy** | Always pick argmax(logits) | Deterministic but repetitive |
| **Temperature (τ)** | Divide logits by τ before softmax | τ < 1: sharper (more confident); τ > 1: flatter (more random) |
| **Top-k** | Zero out all logits except the k highest | Limits sampling to k most probable tokens |
| **Top-p (nucleus)** | Keep smallest set of tokens whose cumulative probability ≥ p | Adapts candidate set size dynamically |

### Temperature Math

```
Standard softmax:           P(token_i) = exp(logit_i) / Σ exp(logit_j)

With temperature τ:         P(token_i) = exp(logit_i / τ) / Σ exp(logit_j / τ)

When τ → 0:  distribution collapses to argmax (greedy)
When τ = 1:  standard softmax (model's learned distribution)
When τ → ∞:  uniform distribution (fully random)
```

### Conceptual Examples

```
Prompt: "The future of artificial intelligence is"

[Greedy]
    "The future of artificial intelligence is going to be a very important
     part of the future of the world. The future of..."
    → Repetitive, gets stuck in loops

[Temperature=0.5]
    "The future of artificial intelligence is likely to transform healthcare,
     education, and transportation in the coming decades."
    → Coherent and focused

[Temperature=1.2]
    "The future of artificial intelligence is dancing with quantum ferrets
     across the crystalline landscapes of imagination."
    → Creative but may lose coherence

[Top-k=50]
    "The future of artificial intelligence is both exciting and uncertain,
     with researchers working to ensure safe development."
    → Balanced diversity

[Top-p=0.9 (nucleus)]
    "The future of artificial intelligence is intertwined with questions
     of ethics, governance, and human augmentation."
    → Dynamically adjusts diversity based on model confidence
```

Key concepts:
- **Greedy:** Always pick the highest-probability token. Deterministic but repetitive.
- **Temperature:** Scales logits before softmax. < 1 sharpens the distribution (more confident), > 1 flattens it (more random).
- **Top-k:** Only sample from the k most likely tokens.
- **Top-p (nucleus):** Sample from the smallest set of tokens whose cumulative probability exceeds p. Adapts the candidate set size dynamically.

---

**[Q&A / WRAP-UP -- 15 min buffer]**

## Session Recap

| Topic | Key Takeaway |
|-------|-------------|
| Scaled Dot-Product Attention | Q @ K^T / sqrt(d_k) computes similarity; softmax produces weights over values |
| Multi-Head Self-Attention | Multiple heads learn different attention patterns; concatenated and projected |
| Positional Encoding | Injects order information since self-attention is permutation invariant |
| BERT | Encoder-only, bidirectional, masked LM; best for classification and understanding |
| GPT | Decoder-only, causal, autoregressive; best for text generation |
| T5 | Encoder-decoder, text-to-text; flexible multi-task architecture |
| Transfer Learning | Pre-trained models + task-specific fine-tuning; small LR, few epochs |
| LoRA (Stretch) | Low-rank weight updates; 100x fewer trainable parameters |
| Generation Strategies (Stretch) | Temperature, top-k, top-p control the diversity-quality trade-off |

## Git Activity

Have students:
1. Check out `lecture/stage-3-finetune` to see the complete reference materials.
2. Create a personal branch `student/<name>/transformers` to try their own conceptual exercises and explorations.

## Additional Resources

- "Attention Is All You Need" (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019): https://arxiv.org/abs/1810.04805
- The Illustrated Transformer by Jay Alammar: https://jalammar.github.io/illustrated-transformer/
- LoRA paper: https://arxiv.org/abs/2106.09685
- Stanford CS224N -- Natural Language Processing with Deep Learning: https://web.stanford.edu/class/cs224n/
