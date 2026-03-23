"""Tests for scaled dot-product attention (NumPy implementation)."""

import pytest
import numpy as np

from src.attention import softmax, scaled_dot_product_attention, compute_attention_example


class TestSoftmax:
    """Tests for the softmax function."""

    def test_output_sums_to_one_1d(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0, atol=1e-7)

    def test_output_sums_to_one_2d(self):
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax(x)
        row_sums = result.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-7)

    def test_output_non_negative(self):
        x = np.array([-5.0, 0.0, 5.0])
        result = softmax(x)
        assert np.all(result >= 0)

    def test_preserves_shape(self):
        x = np.random.randn(3, 4, 5)
        result = softmax(x)
        assert result.shape == x.shape

    def test_known_values(self):
        x = np.array([0.0, 0.0, 0.0])
        result = softmax(x)
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        assert np.allclose(result, expected, atol=1e-7)

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result)), "Softmax must handle large values without overflow"
        assert np.isclose(result.sum(), 1.0, atol=1e-7)

    def test_large_negative_values(self):
        x = np.array([-1000.0, -999.0, -998.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        assert np.isclose(result.sum(), 1.0, atol=1e-7)

    def test_single_element(self):
        x = np.array([42.0])
        result = softmax(x)
        assert np.isclose(result[0], 1.0)


class TestScaledDotProductAttention:
    """Tests for the scaled_dot_product_attention function."""

    def test_output_shapes_2d(self):
        seq_len, d_k = 4, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (seq_len, d_k)
        assert weights.shape == (seq_len, seq_len)

    def test_output_shapes_batched(self):
        batch, heads, seq_len, d_k = 2, 4, 6, 16
        Q = np.random.randn(batch, heads, seq_len, d_k)
        K = np.random.randn(batch, heads, seq_len, d_k)
        V = np.random.randn(batch, heads, seq_len, d_k)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (batch, heads, seq_len, d_k)
        assert weights.shape == (batch, heads, seq_len, seq_len)

    def test_weights_sum_to_one(self):
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        _, weights = scaled_dot_product_attention(Q, K, V)
        row_sums = weights.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            "Each row of attention weights must sum to 1.0"

    def test_weights_are_non_negative(self):
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 8)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert np.all(weights >= 0), "Attention weights must be non-negative"

    def test_mask_zeros_future_positions(self):
        seq_len, d_k = 4, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        upper = weights[np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)]
        assert np.allclose(upper, 0.0, atol=1e-6), \
            "Masked positions should have zero attention weight"

    def test_mask_preserves_sum_to_one(self):
        seq_len, d_k = 4, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        row_sums = weights.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            "Attention weights must still sum to 1.0 even with masking"

    def test_scaling_effect(self):
        np.random.seed(42)
        seq_len = 4
        d_k_large = 512
        Q = np.random.randn(seq_len, d_k_large)
        K = np.random.randn(seq_len, d_k_large)
        V = np.random.randn(seq_len, d_k_large)
        _, weights = scaled_dot_product_attention(Q, K, V)
        max_weight = weights.max()
        assert max_weight < 0.99, \
            "With proper scaling, attention should not collapse to near-one-hot"

    def test_different_qk_lengths(self):
        seq_q, seq_k, d_k = 3, 6, 8
        Q = np.random.randn(seq_q, d_k)
        K = np.random.randn(seq_k, d_k)
        V = np.random.randn(seq_k, d_k)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (seq_q, d_k)
        assert weights.shape == (seq_q, seq_k)

    def test_identity_qk_known_output(self):
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert weights.shape == (2, 2)
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
        assert weights[0, 0] > weights[0, 1], \
            "Q=[1,0] should attend more to K=[1,0] than K=[0,1]"
        assert weights[1, 1] > weights[1, 0], \
            "Q=[0,1] should attend more to K=[0,1] than K=[1,0]"


class TestComputeAttentionExample:
    """Tests for the hand-computed attention example."""

    def test_returns_dict(self):
        result = compute_attention_example()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = compute_attention_example()
        required = {"Q", "K", "V", "d_k", "raw_scores", "scaled_scores",
                     "attention_weights", "output"}
        assert required.issubset(result.keys())

    def test_correct_qkv(self):
        result = compute_attention_example()
        assert np.allclose(result["Q"], [[1, 0], [0, 1]])
        assert np.allclose(result["K"], [[1, 0], [0, 1]])
        assert np.allclose(result["V"], [[1, 2], [3, 4]])

    def test_correct_dk(self):
        result = compute_attention_example()
        assert result["d_k"] == 2

    def test_correct_raw_scores(self):
        result = compute_attention_example()
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert np.allclose(result["raw_scores"], expected, atol=1e-7)

    def test_correct_scaled_scores(self):
        result = compute_attention_example()
        expected = np.array([[1.0 / np.sqrt(2), 0.0],
                             [0.0, 1.0 / np.sqrt(2)]])
        assert np.allclose(result["scaled_scores"], expected, atol=1e-6)

    def test_correct_attention_weights(self):
        result = compute_attention_example()
        s = 1.0 / np.sqrt(2)
        w00 = np.exp(s) / (np.exp(s) + np.exp(0))
        w01 = np.exp(0) / (np.exp(s) + np.exp(0))
        expected = np.array([[w00, w01], [w01, w00]])
        assert np.allclose(result["attention_weights"], expected, atol=1e-5)

    def test_weights_sum_to_one(self):
        result = compute_attention_example()
        row_sums = result["attention_weights"].sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_correct_output(self):
        result = compute_attention_example()
        expected_weights = result["attention_weights"]
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_output = expected_weights @ V
        assert np.allclose(result["output"], expected_output, atol=1e-5)
