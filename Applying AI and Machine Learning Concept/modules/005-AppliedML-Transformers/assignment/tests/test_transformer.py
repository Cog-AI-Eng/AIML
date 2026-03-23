"""Tests for transformer conceptual exercises: multi-head dimensions,
positional encoding, and parameter counting."""

import pytest
import numpy as np

from src.transformer import (
    compute_head_dimensions,
    compute_mha_shapes,
    compute_positional_encoding,
    analyze_positional_encoding,
    compute_transformer_block_params,
)


class TestMultiHeadDimensions:
    """Tests for compute_head_dimensions."""

    def test_basic_dimensions(self):
        result = compute_head_dimensions(512, 8)
        assert result["d_k"] == 64
        assert result["d_v"] == 64

    def test_num_projections(self):
        result = compute_head_dimensions(256, 4)
        assert result["num_projection_matrices"] == 4

    def test_params_per_projection(self):
        result = compute_head_dimensions(512, 8)
        assert result["params_per_projection"] == 512 * 512 + 512

    def test_total_parameters(self):
        result = compute_head_dimensions(512, 8)
        expected = 4 * (512 * 512 + 512)
        assert result["total_parameters"] == expected

    def test_valid_divisibility(self):
        assert compute_head_dimensions(512, 8)["valid"] is True
        assert compute_head_dimensions(64, 8)["valid"] is True

    def test_invalid_divisibility(self):
        assert compute_head_dimensions(33, 4)["valid"] is False
        assert compute_head_dimensions(100, 3)["valid"] is False

    def test_single_head(self):
        result = compute_head_dimensions(128, 1)
        assert result["d_k"] == 128
        assert result["valid"] is True

    def test_max_heads(self):
        result = compute_head_dimensions(64, 64)
        assert result["d_k"] == 1
        assert result["valid"] is True

    def test_small_model(self):
        result = compute_head_dimensions(16, 4)
        assert result["d_k"] == 4
        assert result["total_parameters"] == 4 * (16 * 16 + 16)


class TestMHAShapes:
    """Tests for compute_mha_shapes."""

    def test_input_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["input"] == (2, 10, 512)

    def test_after_projection_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["after_projection"] == (2, 10, 512)

    def test_after_split_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["after_split"] == (2, 8, 10, 64)

    def test_attention_scores_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["attention_scores"] == (2, 8, 10, 10)

    def test_attention_output_per_head_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["attention_output_per_head"] == (2, 8, 10, 64)

    def test_after_concat_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["after_concat"] == (2, 10, 512)

    def test_final_output_shape(self):
        result = compute_mha_shapes(2, 10, 512, 8)
        assert result["final_output"] == (2, 10, 512)

    def test_single_head_shapes(self):
        result = compute_mha_shapes(1, 5, 64, 1)
        assert result["after_split"] == (1, 1, 5, 64)
        assert result["attention_scores"] == (1, 1, 5, 5)

    def test_different_config(self):
        result = compute_mha_shapes(4, 20, 256, 4)
        assert result["after_split"] == (4, 4, 20, 64)
        assert result["attention_scores"] == (4, 4, 20, 20)


class TestPositionalEncoding:
    """Tests for compute_positional_encoding."""

    def test_output_shape(self):
        pe = compute_positional_encoding(100, 32)
        assert pe.shape == (100, 32)

    def test_values_bounded(self):
        pe = compute_positional_encoding(500, 64)
        assert np.all(pe >= -1.0) and np.all(pe <= 1.0), \
            "Sinusoidal PE values must be in [-1, 1]"

    def test_position_zero_even_cols(self):
        pe = compute_positional_encoding(10, 16)
        assert np.allclose(pe[0, 0::2], 0.0, atol=1e-7), \
            "sin(0) = 0 for all even columns at position 0"

    def test_position_zero_odd_cols(self):
        pe = compute_positional_encoding(10, 16)
        assert np.allclose(pe[0, 1::2], 1.0, atol=1e-7), \
            "cos(0) = 1 for all odd columns at position 0"

    def test_different_positions_differ(self):
        pe = compute_positional_encoding(100, 32)
        assert not np.allclose(pe[0], pe[1]), \
            "Different positions must have different encodings"
        assert not np.allclose(pe[5], pe[50])

    def test_rows_unique(self):
        pe = compute_positional_encoding(50, 16)
        unique_rows = np.unique(pe, axis=0)
        assert unique_rows.shape[0] == 50, "All position encodings should be unique"

    def test_deterministic(self):
        pe1 = compute_positional_encoding(10, 8)
        pe2 = compute_positional_encoding(10, 8)
        assert np.allclose(pe1, pe2), "Positional encoding must be deterministic"

    def test_first_column_is_sin(self):
        pe = compute_positional_encoding(100, 8)
        positions = np.arange(100).astype(float)
        expected_col0 = np.sin(positions)
        assert np.allclose(pe[:, 0], expected_col0, atol=1e-6), \
            "Column 0 should be sin(pos) with the lowest frequency"


class TestAnalyzePositionalEncoding:
    """Tests for analyze_positional_encoding."""

    def test_all_properties_true_for_valid_pe(self):
        pe = compute_positional_encoding(100, 32)
        props = analyze_positional_encoding(pe)
        assert props["values_bounded"] is True
        assert props["rows_unique"] is True
        assert props["deterministic"] is True
        assert props["even_cols_use_sin"] is True
        assert props["odd_cols_use_cos"] is True

    def test_values_bounded_detects_out_of_range(self):
        bad_pe = np.ones((10, 4)) * 2.0
        props = analyze_positional_encoding(bad_pe)
        assert props["values_bounded"] is False

    def test_rows_unique_detects_duplicates(self):
        bad_pe = np.zeros((10, 4))
        props = analyze_positional_encoding(bad_pe)
        assert props["rows_unique"] is False

    def test_returns_all_keys(self):
        pe = compute_positional_encoding(10, 8)
        props = analyze_positional_encoding(pe)
        required = {"values_bounded", "rows_unique", "deterministic",
                     "even_cols_use_sin", "odd_cols_use_cos"}
        assert required.issubset(props.keys())


class TestTransformerBlockParams:
    """Tests for compute_transformer_block_params."""

    def test_mha_params_with_bias(self):
        result = compute_transformer_block_params(512, 8, 2048, bias=True)
        expected_mha = 4 * (512 * 512 + 512)
        assert result["mha_params"] == expected_mha

    def test_ffn_params_with_bias(self):
        result = compute_transformer_block_params(512, 8, 2048, bias=True)
        expected_ffn = (512 * 2048 + 2048) + (2048 * 512 + 512)
        assert result["ffn_params"] == expected_ffn

    def test_layernorm_params(self):
        result = compute_transformer_block_params(512, 8, 2048)
        expected_ln = 2 * (512 + 512)
        assert result["layernorm_params"] == expected_ln

    def test_total_params(self):
        result = compute_transformer_block_params(512, 8, 2048)
        expected = result["mha_params"] + result["ffn_params"] + result["layernorm_params"]
        assert result["total_params"] == expected

    def test_no_bias(self):
        result = compute_transformer_block_params(256, 4, 1024, bias=False)
        expected_mha = 4 * (256 * 256)
        expected_ffn = 256 * 1024 + 1024 * 256
        assert result["mha_params"] == expected_mha
        assert result["ffn_params"] == expected_ffn

    def test_small_model(self):
        result = compute_transformer_block_params(32, 4, 128, bias=True)
        mha = 4 * (32 * 32 + 32)
        ffn = (32 * 128 + 128) + (128 * 32 + 32)
        ln = 2 * (32 + 32)
        assert result["total_params"] == mha + ffn + ln

    def test_returns_all_keys(self):
        result = compute_transformer_block_params(64, 4, 256)
        required = {"mha_params", "ffn_params", "layernorm_params", "total_params"}
        assert required.issubset(result.keys())
