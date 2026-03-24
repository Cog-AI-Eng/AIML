"""
Transformer architecture conceptual exercises.

Associates answer questions about multi-head attention dimensions,
positional encoding computation, and parameter counting using
NumPy and pure Python -- no deep learning frameworks required.
"""

import math

import numpy as np


def compute_head_dimensions(d_model: int, num_heads: int) -> dict:
    """Compute per-head dimensions and parameter counts for multi-head attention.

    Multi-head attention uses four projection matrices (W_q, W_k, W_v, W_o),
    each of shape (d_model, d_model), plus a bias vector of length d_model.

    Args:
        d_model: Total model dimension.
        num_heads: Number of attention heads.

    Returns:
        dict with:
            "d_k": per-head key/query dimension (d_model // num_heads)
            "d_v": per-head value dimension (same as d_k)
            "num_projection_matrices": number of projection matrices (always 4)
            "params_per_projection": parameters per projection (d_model * d_model + d_model)
            "total_parameters": total parameters across all 4 projections
            "valid": bool -- whether d_model is evenly divisible by num_heads
    """
    # TODO: Compute each value and return the dict.
    #   Remember: each linear projection has weight (d_model x d_model)
    #   plus a bias vector (d_model), so params_per_projection = d_model^2 + d_model.
    #   total_parameters = 4 * params_per_projection.
    #   valid = (d_model % num_heads == 0).
    raise NotImplementedError("Implement compute_head_dimensions")


def compute_mha_shapes(
    batch_size: int, seq_len: int, d_model: int, num_heads: int
) -> dict:
    """Trace tensor shapes through every stage of multi-head self-attention.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        d_model: Model dimension.
        num_heads: Number of attention heads.

    Returns:
        dict with shape tuples:
            "input": input tensor shape
            "after_projection": shape after Q/K/V linear projection
            "after_split": shape after reshaping to separate heads
            "attention_scores": shape of Q @ K^T score matrix (per batch)
            "attention_output_per_head": shape of weighted V output per head
            "after_concat": shape after concatenating all heads
            "final_output": shape after output projection W_o
    """
    # TODO: Compute each shape tuple. The key insight is that splitting
    #   heads reshapes (batch, seq, d_model) into (batch, num_heads, seq, d_k)
    #   where d_k = d_model // num_heads.
    raise NotImplementedError("Implement compute_mha_shapes")


def compute_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Compute the sinusoidal positional encoding matrix.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        max_len: Number of positions to encode.
        d_model: Encoding dimension (must be even).

    Returns:
        np.ndarray of shape (max_len, d_model) with positional encodings.
    """
    # TODO: Build the positional encoding matrix:
    #   1. Create pe = np.zeros((max_len, d_model)).
    #   2. Create position = np.arange(max_len).reshape(-1, 1) as float.
    #   3. Create div_term = np.exp(
    #          np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    #      ).
    #   4. Set pe[:, 0::2] = np.sin(position * div_term).
    #   5. Set pe[:, 1::2] = np.cos(position * div_term).
    #   6. Return pe.
    raise NotImplementedError("Implement compute_positional_encoding")


def analyze_positional_encoding(pe: np.ndarray) -> dict:
    """Analyze mathematical properties of a positional encoding matrix.

    Args:
        pe: Positional encoding matrix, shape (max_len, d_model).

    Returns:
        dict with:
            "values_bounded": bool -- all values are in [-1, 1]
            "rows_unique": bool -- no two position vectors are identical
            "deterministic": bool -- always True for sinusoidal encoding
            "even_cols_use_sin": bool -- even columns are in sin range pattern
            "odd_cols_use_cos": bool -- odd columns are in cos range pattern
    """
    # TODO: Check each property:
    #   - values_bounded: np.all(pe >= -1) and np.all(pe <= 1)
    #   - rows_unique: no two rows are exactly equal (compare all pairs,
    #     or check that the number of unique rows equals max_len)
    #   - deterministic: always True
    #   - even_cols_use_sin: pe[0, 0::2] are all 0.0 (sin(0) = 0)
    #   - odd_cols_use_cos: pe[0, 1::2] are all 1.0 (cos(0) = 1)
    raise NotImplementedError("Implement analyze_positional_encoding")


def compute_transformer_block_params(
    d_model: int, num_heads: int, d_ff: int, bias: bool = True
) -> dict:
    """Count parameters in a complete transformer encoder block.

    A transformer encoder block contains:
        - Multi-head self-attention: 4 projection matrices (W_q, W_k, W_v, W_o)
        - Feed-forward network: two linear layers (d_model -> d_ff, d_ff -> d_model)
        - Two layer norms: each has scale (d_model) and bias (d_model) vectors

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        bias: Whether linear layers include bias terms.

    Returns:
        dict with:
            "mha_params": parameters in multi-head attention
            "ffn_params": parameters in feed-forward network
            "layernorm_params": parameters in both layer norms combined
            "total_params": total parameter count
    """
    # TODO: Compute each component:
    #   MHA: 4 projections, each d_model*d_model (+ d_model if bias)
    #   FFN: layer 1 is d_model*d_ff (+ d_ff if bias),
    #        layer 2 is d_ff*d_model (+ d_model if bias)
    #   LayerNorm: 2 norms, each with d_model scale + d_model bias = 2*d_model
    #   total = mha + ffn + layernorm
    raise NotImplementedError("Implement compute_transformer_block_params")
