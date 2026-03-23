"""Tests for RNN conceptual exercises."""

import pytest
import numpy as np

from src.rnn_concepts import (
    sigmoid,
    vanilla_rnn_step,
    lstm_parameter_count,
    gru_parameter_count,
    lstm_cell_forward,
    gru_vs_lstm_tradeoffs,
    encoder_decoder_bottleneck,
)


class TestVanillaRnnStep:
    """Tests for vanilla_rnn_step."""

    def test_output_shape(self):
        W_xh = np.random.randn(4, 3)
        W_hh = np.random.randn(4, 4)
        b_h = np.zeros(4)
        x_t = np.random.randn(3)
        h_prev = np.zeros(4)
        h_t = vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h_prev)
        assert h_t.shape == (4,)

    def test_zero_hidden_state(self):
        W_xh = np.array([[0.5, 0.3], [-0.2, 0.4]])
        W_hh = np.array([[0.1, -0.1], [0.2, 0.3]])
        b_h = np.zeros(2)
        x_t = np.array([1.0, 0.5])
        h_prev = np.zeros(2)
        h_t = vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h_prev)
        expected = np.tanh(W_xh @ x_t)
        np.testing.assert_allclose(h_t, expected, atol=1e-7)

    def test_nonzero_hidden_state(self):
        W_xh = np.array([[0.5, 0.3], [-0.2, 0.4]])
        W_hh = np.array([[0.1, -0.1], [0.2, 0.3]])
        b_h = np.array([0.1, -0.1])
        x_t = np.array([1.0, 0.5])
        h_prev = np.array([0.5, -0.3])
        h_t = vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h_prev)
        expected = np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
        np.testing.assert_allclose(h_t, expected, atol=1e-7)

    def test_output_bounded(self):
        np.random.seed(42)
        W_xh = np.random.randn(5, 3)
        W_hh = np.random.randn(5, 5)
        b_h = np.random.randn(5)
        x_t = np.random.randn(3) * 10
        h_prev = np.random.randn(5) * 10
        h_t = vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h_prev)
        assert np.all(np.abs(h_t) <= 1.0)

    def test_zero_input_zero_bias(self):
        W_xh = np.ones((2, 3))
        W_hh = np.eye(2) * 0.5
        b_h = np.zeros(2)
        x_t = np.zeros(3)
        h_prev = np.array([0.8, -0.4])
        h_t = vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h_prev)
        expected = np.tanh(W_hh @ h_prev)
        np.testing.assert_allclose(h_t, expected, atol=1e-7)

    def test_multiple_steps_evolve(self):
        np.random.seed(0)
        W_xh = np.random.randn(3, 2) * 0.5
        W_hh = np.random.randn(3, 3) * 0.5
        b_h = np.zeros(3)
        h = np.zeros(3)
        inputs = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                  np.array([1.0, 1.0])]
        states = []
        for x_t in inputs:
            h = vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h)
            states.append(h.copy())
        assert not np.allclose(states[0], states[1])
        assert not np.allclose(states[1], states[2])


class TestLstmParameterCount:
    """Tests for lstm_parameter_count."""

    def test_small_lstm(self):
        assert lstm_parameter_count(10, 20, bias=True) == 4 * (10 * 20 + 20 * 20 + 20)

    def test_small_lstm_no_bias(self):
        assert lstm_parameter_count(10, 20, bias=False) == 4 * (10 * 20 + 20 * 20)

    def test_embedding_lstm(self):
        assert lstm_parameter_count(16, 32, bias=True) == 4 * (16 * 32 + 32 * 32 + 32)

    def test_large_lstm(self):
        expected = 4 * (512 * 256 + 256 * 256 + 256)
        assert lstm_parameter_count(512, 256, bias=True) == expected

    def test_bias_adds_4h_params(self):
        with_bias = lstm_parameter_count(10, 20, bias=True)
        without_bias = lstm_parameter_count(10, 20, bias=False)
        assert with_bias - without_bias == 4 * 20

    def test_known_value(self):
        assert lstm_parameter_count(16, 32, bias=True) == 6272


class TestGruParameterCount:
    """Tests for gru_parameter_count."""

    def test_small_gru(self):
        assert gru_parameter_count(10, 20, bias=True) == 3 * (10 * 20 + 20 * 20 + 20)

    def test_small_gru_no_bias(self):
        assert gru_parameter_count(10, 20, bias=False) == 3 * (10 * 20 + 20 * 20)

    def test_embedding_gru(self):
        assert gru_parameter_count(16, 32, bias=True) == 3 * (16 * 32 + 32 * 32 + 32)

    def test_bias_adds_3h_params(self):
        with_bias = gru_parameter_count(10, 20, bias=True)
        without_bias = gru_parameter_count(10, 20, bias=False)
        assert with_bias - without_bias == 3 * 20

    def test_gru_fewer_than_lstm(self):
        lstm = lstm_parameter_count(32, 64, bias=True)
        gru = gru_parameter_count(32, 64, bias=True)
        assert gru < lstm

    def test_gru_to_lstm_ratio(self):
        lstm = lstm_parameter_count(32, 64, bias=True)
        gru = gru_parameter_count(32, 64, bias=True)
        assert gru / lstm == pytest.approx(0.75, abs=1e-6)


class TestLstmCellForward:
    """Tests for lstm_cell_forward."""

    def _make_params(self, input_size, hidden_size, seed=42):
        np.random.seed(seed)
        d = hidden_size + input_size
        W_f = np.random.randn(hidden_size, d) * 0.1
        W_i = np.random.randn(hidden_size, d) * 0.1
        W_g = np.random.randn(hidden_size, d) * 0.1
        W_o = np.random.randn(hidden_size, d) * 0.1
        b_f = np.zeros(hidden_size)
        b_i = np.zeros(hidden_size)
        b_g = np.zeros(hidden_size)
        b_o = np.zeros(hidden_size)
        return W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o

    def test_output_shapes(self):
        input_size, hidden_size = 3, 2
        x_t = np.random.randn(input_size)
        h_prev = np.zeros(hidden_size)
        c_prev = np.zeros(hidden_size)
        params = self._make_params(input_size, hidden_size)
        h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev, *params)
        assert h_t.shape == (hidden_size,)
        assert c_t.shape == (hidden_size,)

    def test_hidden_state_bounded(self):
        input_size, hidden_size = 5, 4
        np.random.seed(0)
        x_t = np.random.randn(input_size) * 5
        h_prev = np.random.randn(hidden_size)
        c_prev = np.random.randn(hidden_size)
        params = self._make_params(input_size, hidden_size, seed=1)
        h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev, *params)
        assert np.all(np.abs(h_t) <= 1.0)

    def test_zero_input_zero_state(self):
        input_size, hidden_size = 3, 2
        x_t = np.zeros(input_size)
        h_prev = np.zeros(hidden_size)
        c_prev = np.zeros(hidden_size)
        d = hidden_size + input_size
        W = np.ones((hidden_size, d)) * 0.1
        b = np.zeros(hidden_size)
        h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev,
                                     W, W, W, W, b, b, b, b)
        concat = np.zeros(d)
        f = sigmoid(W @ concat)
        i = sigmoid(W @ concat)
        g = np.tanh(W @ concat)
        o = sigmoid(W @ concat)
        expected_c = f * c_prev + i * g
        expected_h = o * np.tanh(expected_c)
        np.testing.assert_allclose(c_t, expected_c, atol=1e-7)
        np.testing.assert_allclose(h_t, expected_h, atol=1e-7)

    def test_forget_gate_erases_cell(self):
        """When forget gate is ~0 and input gate is ~0, cell state should vanish."""
        input_size, hidden_size = 2, 2
        d = hidden_size + input_size
        W_f = np.ones((hidden_size, d)) * (-10.0)
        W_i = np.ones((hidden_size, d)) * (-10.0)
        W_g = np.zeros((hidden_size, d))
        W_o = np.ones((hidden_size, d)) * 10.0
        b = np.zeros(hidden_size)
        x_t = np.ones(input_size)
        h_prev = np.ones(hidden_size)
        c_prev = np.array([1.0, -1.0])
        h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev,
                                     W_f, W_i, W_g, W_o, b, b, b, b)
        np.testing.assert_allclose(c_t, np.zeros(hidden_size), atol=1e-4)

    def test_numerical_correctness(self):
        input_size, hidden_size = 3, 2
        x_t = np.array([1.0, 0.5, -0.5])
        h_prev = np.array([0.1, -0.2])
        c_prev = np.array([0.3, 0.1])
        params = self._make_params(input_size, hidden_size, seed=99)
        W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o = params
        h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev, *params)
        concat = np.concatenate([h_prev, x_t])
        f = sigmoid(W_f @ concat + b_f)
        i = sigmoid(W_i @ concat + b_i)
        g = np.tanh(W_g @ concat + b_g)
        o = sigmoid(W_o @ concat + b_o)
        expected_c = f * c_prev + i * g
        expected_h = o * np.tanh(expected_c)
        np.testing.assert_allclose(c_t, expected_c, atol=1e-7)
        np.testing.assert_allclose(h_t, expected_h, atol=1e-7)

    def test_multiple_steps(self):
        input_size, hidden_size = 3, 2
        params = self._make_params(input_size, hidden_size)
        h = np.zeros(hidden_size)
        c = np.zeros(hidden_size)
        np.random.seed(7)
        for _ in range(5):
            x = np.random.randn(input_size)
            h, c = lstm_cell_forward(x, h, c, *params)
        assert h.shape == (hidden_size,)
        assert c.shape == (hidden_size,)
        assert not np.allclose(h, 0.0)


class TestGruVsLstmTradeoffs:
    """Tests for gru_vs_lstm_tradeoffs."""

    def test_returns_dict(self):
        result = gru_vs_lstm_tradeoffs()
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        result = gru_vs_lstm_tradeoffs()
        required = {"lstm_gate_count", "gru_gate_count",
                     "lstm_has_cell_state", "gru_has_cell_state",
                     "fewer_parameters", "gru_param_ratio"}
        assert required.issubset(result.keys())

    def test_lstm_gate_count(self):
        result = gru_vs_lstm_tradeoffs()
        assert result["lstm_gate_count"] == 4

    def test_gru_gate_count(self):
        result = gru_vs_lstm_tradeoffs()
        assert result["gru_gate_count"] == 3

    def test_lstm_has_cell_state(self):
        result = gru_vs_lstm_tradeoffs()
        assert result["lstm_has_cell_state"] is True

    def test_gru_has_no_cell_state(self):
        result = gru_vs_lstm_tradeoffs()
        assert result["gru_has_cell_state"] is False

    def test_fewer_parameters(self):
        result = gru_vs_lstm_tradeoffs()
        assert result["fewer_parameters"] == "GRU"

    def test_param_ratio(self):
        result = gru_vs_lstm_tradeoffs()
        assert result["gru_param_ratio"] == pytest.approx(0.75, abs=1e-6)


class TestEncoderDecoderBottleneck:
    """Tests for encoder_decoder_bottleneck."""

    def test_returns_dict(self):
        result = encoder_decoder_bottleneck(50, 512, 256)
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        result = encoder_decoder_bottleneck(50, 512, 256)
        required = {"total_input_features", "bottleneck_size",
                     "compression_ratio"}
        assert required.issubset(result.keys())

    def test_total_input_features(self):
        result = encoder_decoder_bottleneck(50, 512, 256)
        assert result["total_input_features"] == 25600

    def test_bottleneck_size(self):
        result = encoder_decoder_bottleneck(50, 512, 256)
        assert result["bottleneck_size"] == 256

    def test_compression_ratio(self):
        result = encoder_decoder_bottleneck(50, 512, 256)
        assert result["compression_ratio"] == pytest.approx(0.01, abs=1e-6)

    def test_short_sequence(self):
        result = encoder_decoder_bottleneck(10, 64, 128)
        assert result["total_input_features"] == 640
        assert result["bottleneck_size"] == 128
        assert result["compression_ratio"] == pytest.approx(0.2, abs=1e-6)

    def test_large_hidden_dim(self):
        result = encoder_decoder_bottleneck(5, 10, 100)
        assert result["total_input_features"] == 50
        assert result["bottleneck_size"] == 100
        assert result["compression_ratio"] == pytest.approx(2.0, abs=1e-6)

    def test_compression_ratio_is_float(self):
        result = encoder_decoder_bottleneck(10, 10, 10)
        assert isinstance(result["compression_ratio"], float)
