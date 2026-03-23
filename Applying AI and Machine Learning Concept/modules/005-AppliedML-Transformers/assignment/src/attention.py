"""
Scaled dot-product attention computation using NumPy.

Students must implement the softmax function and the full attention
pipeline using only NumPy -- no deep learning frameworks allowed.
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last axis of x.

    Must be numerically stable: subtract the row-wise maximum before
    exponentiating to prevent overflow.

    Args:
        x: Input array of any shape.

    Returns:
        Array of same shape with softmax applied along the last axis.
        Each row (last axis) sums to 1.0.
    """
    # TODO: Implement numerically stable softmax:
    #   1. Compute the max along the last axis (keepdims=True).
    #   2. Subtract the max from x for numerical stability.
    #   3. Compute exp of the shifted values.
    #   4. Divide by the sum along the last axis (keepdims=True).
    #   5. Return the result.
    raise NotImplementedError("Implement softmax")


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None,
) -> tuple:
    """Compute scaled dot-product attention using NumPy.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: Queries, shape (..., seq_len_q, d_k).
        K: Keys, shape (..., seq_len_k, d_k).
        V: Values, shape (..., seq_len_k, d_v).
        mask: Optional boolean mask, broadcastable to (..., seq_len_q, seq_len_k).
              Positions where mask is True are set to -1e9 before softmax.

    Returns:
        tuple of (output, attention_weights):
            output: shape (..., seq_len_q, d_v).
            attention_weights: shape (..., seq_len_q, seq_len_k).
    """
    # TODO: Implement scaled dot-product attention:
    #   1. Get d_k from the last dimension of Q.
    #   2. Compute raw scores: Q @ K^T. For multi-dim arrays, transpose
    #      the last two axes using np.swapaxes(K, -2, -1).
    #   3. Scale scores by dividing by sqrt(d_k).
    #   4. If mask is not None, set masked positions (where mask is True)
    #      to -1e9.
    #   5. Apply your softmax function to get attention_weights.
    #   6. Compute output = attention_weights @ V.
    #   7. Return (output, attention_weights).
    raise NotImplementedError("Implement scaled_dot_product_attention")


def compute_attention_example() -> dict:
    """Compute attention for a small hand-worked example.

    Use the following fixed inputs:
        Q = [[1, 0],
             [0, 1]]
        K = [[1, 0],
             [0, 1]]
        V = [[1, 2],
             [3, 4]]

    Compute every intermediate step of scaled dot-product attention.

    Returns:
        dict with keys:
            "Q": the query matrix (2x2 numpy array)
            "K": the key matrix (2x2 numpy array)
            "V": the value matrix (2x2 numpy array)
            "d_k": the key dimension (int)
            "raw_scores": Q @ K^T before scaling (2x2 numpy array)
            "scaled_scores": after dividing by sqrt(d_k) (2x2 numpy array)
            "attention_weights": after softmax (2x2 numpy array)
            "output": final attention output (2x2 numpy array)
    """
    # TODO: Create the Q, K, V matrices as numpy arrays, then compute
    #   each step manually. You may call your own softmax and
    #   scaled_dot_product_attention functions, or compute each
    #   intermediate value directly.
    raise NotImplementedError("Implement compute_attention_example")
