# AIML-AM-AppliedML-Transformers

**Activity Type:** Assignment  
**Duration:** 120 minutes  
**Module:** 005 -- Applied ML: Transformers

---

## Overview

In this assignment you will demonstrate your understanding of transformer architecture concepts through mathematical computation, dimension analysis, and architectural reasoning. You will work through five milestones covering the core ideas that power modern transformers:

1. **Scaled dot-product attention** -- Compute attention scores, weights, and outputs by hand using NumPy.
2. **Multi-head attention analysis** -- Analyze dimension splitting, shape transformations, and parameter counts.
3. **Positional encoding** -- Compute sinusoidal positional encodings and verify their mathematical properties.
4. **Architecture selection** -- Choose the right transformer architecture (BERT / GPT / T5) for given NLP tasks.
5. **Fine-tuning pipeline design** -- Order pipeline steps, recommend strategies, and analyze computational complexity.

---

## Learning Objectives

- Compute scaled dot-product attention step by step using only NumPy.
- Analyze multi-head attention dimension splitting, shape transformations, and parameter counts.
- Construct sinusoidal positional encodings and verify their mathematical properties.
- Select appropriate transformer architectures for different NLP tasks with justified reasoning.
- Design fine-tuning pipelines, recommend training strategies, and analyze attention complexity.

---

## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| NumPy | latest stable |
| pytest | latest stable |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
assignment/
  README.md              # This file
  requirements.txt       # Python dependencies
  .gitignore             # Git ignore rules
  src/
    __init__.py
    attention.py         # TODO: Implement attention computation with NumPy
    transformer.py       # TODO: Implement dimension analysis and positional encoding
    finetune.py          # TODO: Implement architecture selection and pipeline design
  tests/
    conftest.py          # Shared fixtures
    test_attention.py    # Tests for attention computation
    test_transformer.py  # Tests for transformer concepts
    test_finetune.py     # Tests for architecture and pipeline decisions
  solutions/
    __init__.py
    attention.py         # Reference solution
    transformer.py       # Reference solution
    finetune.py          # Reference solution
```

---

## Instructions

### Milestone 1: Scaled Dot-Product Attention (25 min)

Open `src/attention.py`. Implement three functions:

- **`softmax(x)`** -- Implement numerically stable softmax along the last axis using NumPy. Subtract the row-wise max before exponentiating to prevent overflow.
- **`scaled_dot_product_attention(Q, K, V, mask)`** -- Compute the full attention pipeline: raw scores (Q @ K^T), scaling by sqrt(d_k), optional masking, softmax, and weighted sum with V. Use only NumPy operations.
- **`compute_attention_example()`** -- Compute attention for a specific small example (Q = K = identity, V = [[1,2],[3,4]]) and return all intermediate values as a dictionary.

Run tests: `pytest tests/test_attention.py -v`

### Milestone 2: Multi-Head Attention Analysis (20 min)

Open `src/transformer.py`. Implement:

- **`compute_head_dimensions(d_model, num_heads)`** -- Given model dimension and number of heads, compute per-head dimensions and total parameter counts for the four projection matrices (W_q, W_k, W_v, W_o).
- **`compute_mha_shapes(batch_size, seq_len, d_model, num_heads)`** -- Trace the tensor shapes through every stage of multi-head attention: input, after projection, after head splitting, attention scores, per-head output, after concatenation, and final output.

Run tests: `pytest tests/test_transformer.py::TestMultiHeadDimensions -v`

### Milestone 3: Positional Encoding (25 min)

In `src/transformer.py`, implement:

- **`compute_positional_encoding(max_len, d_model)`** -- Generate the sinusoidal positional encoding matrix using the standard formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)).
- **`analyze_positional_encoding(pe)`** -- Given a PE matrix, verify its mathematical properties: values bounded in [-1, 1], unique row vectors, even columns use sin, odd columns use cos.
- **`compute_transformer_block_params(d_model, num_heads, d_ff)`** -- Count total parameters in a transformer encoder block (MHA + FFN + 2 LayerNorms).

Run tests: `pytest tests/test_transformer.py -v`

### Milestone 4: Architecture Selection (25 min)

Open `src/finetune.py`. Implement:

- **`select_architecture(task)`** -- Given an NLP task name, select the best-fit architecture (BERT, GPT, or T5) and explain why. BERT for classification/understanding tasks, GPT for generation tasks, T5 for sequence-to-sequence tasks.
- **`order_finetuning_steps(steps)`** -- Given a shuffled list of fine-tuning pipeline steps, return them in correct execution order.

Run tests: `pytest tests/test_finetune.py::TestSelectArchitecture -v` and `pytest tests/test_finetune.py::TestOrderFinetuningSteps -v`

### Milestone 5: Fine-tuning Strategy and Complexity (25 min)

In `src/finetune.py`, implement:

- **`recommend_finetuning_strategy(dataset_size, compute_budget)`** -- Recommend feature extraction, partial fine-tuning, or full fine-tuning based on dataset size and compute budget. Return strategy details including whether to freeze the base model, learning rate, and epoch count.
- **`analyze_attention_complexity(seq_length, d_model, num_heads)`** -- Compute the number of multiplications for each stage of self-attention (QKV projections, score computation, output computation, output projection) and total memory for attention score storage.

Run tests: `pytest tests/test_finetune.py -v`

---

## Running All Tests

```bash
# All tests (should FAIL on starter code, PASS on completed code)
pytest tests/ -v

# Individual test files
pytest tests/test_attention.py -v
pytest tests/test_transformer.py -v
pytest tests/test_finetune.py -v
```

---

## Submission Checklist

- [ ] All functions in `src/attention.py` implemented and passing tests.
- [ ] All functions in `src/transformer.py` implemented and passing tests.
- [ ] All functions in `src/finetune.py` implemented and passing tests.
- [ ] `pytest tests/ -v` shows all tests passing.
- [ ] Code is clean, follows PEP 8, and contains no unnecessary comments.
