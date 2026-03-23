"""Tests for Milestone 3: MLP Architecture Design."""

import numpy as np
import pytest

from src.architecture import count_parameters, forward_pass, compare_architectures


class TestCountParameters:
    def test_single_layer(self):
        assert count_parameters([64, 10]) == 64 * 10 + 10

    def test_one_hidden_layer(self):
        expected = (64 * 128 + 128) + (128 * 10 + 10)
        assert count_parameters([64, 128, 10]) == expected

    def test_two_hidden_layers(self):
        expected = (64 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
        assert count_parameters([64, 128, 64, 10]) == expected

    def test_wide_shallow(self):
        expected = (64 * 512 + 512) + (512 * 10 + 10)
        assert count_parameters([64, 512, 10]) == expected

    def test_narrow_deep(self):
        expected = (
            (64 * 32 + 32)
            + (32 * 32 + 32)
            + (32 * 32 + 32)
            + (32 * 32 + 32)
            + (32 * 10 + 10)
        )
        assert count_parameters([64, 32, 32, 32, 32, 10]) == expected

    def test_perceptron_equivalent(self):
        assert count_parameters([784, 10]) == 784 * 10 + 10

    def test_returns_int(self):
        result = count_parameters([64, 128, 10])
        assert isinstance(result, int)


class TestForwardPass:
    def test_single_layer_no_activation(self):
        X = np.array([[1.0, 2.0]])
        weights = [np.array([[0.5], [0.3]])]
        biases = [np.array([0.1])]
        result = forward_pass(X, weights, biases, "relu")
        expected = X @ weights[0] + biases[0]
        np.testing.assert_allclose(result[-1], expected)

    def test_output_list_length(self):
        X = np.random.randn(4, 3)
        weights = [np.random.randn(3, 5), np.random.randn(5, 2)]
        biases = [np.random.randn(5), np.random.randn(2)]
        result = forward_pass(X, weights, biases, "relu")
        assert len(result) == 2

    def test_relu_applied_to_hidden(self):
        X = np.array([[1.0, -1.0]])
        w1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
        b1 = np.array([0.0, 0.0])
        w2 = np.array([[1.0], [1.0]])
        b2 = np.array([0.0])
        result = forward_pass(X, [w1, w2], [b1, b2], "relu")
        z1 = X @ w1 + b1
        a1 = np.maximum(0, z1)
        expected_output = a1 @ w2 + b2
        np.testing.assert_allclose(result[0], a1)
        np.testing.assert_allclose(result[1], expected_output)

    def test_sigmoid_activation(self):
        X = np.array([[0.5, -0.5]])
        w1 = np.array([[1.0], [1.0]])
        b1 = np.array([0.0])
        w2 = np.array([[2.0]])
        b2 = np.array([0.0])
        result = forward_pass(X, [w1, w2], [b1, b2], "sigmoid")
        z1 = X @ w1 + b1
        a1 = 1.0 / (1.0 + np.exp(-z1))
        expected_output = a1 @ w2 + b2
        np.testing.assert_allclose(result[0], a1, atol=1e-10)
        np.testing.assert_allclose(result[1], expected_output, atol=1e-10)

    def test_no_activation_on_output_layer(self):
        X = np.array([[1.0, 1.0]])
        w1 = np.array([[1.0], [1.0]])
        b1 = np.array([0.0])
        w2 = np.array([[-5.0]])
        b2 = np.array([0.0])
        result = forward_pass(X, [w1, w2], [b1, b2], "relu")
        assert result[-1][0, 0] < 0, (
            "Output layer should not have activation applied"
        )

    def test_output_shapes(self):
        batch_size = 8
        X = np.random.randn(batch_size, 4)
        w1 = np.random.randn(4, 6)
        b1 = np.random.randn(6)
        w2 = np.random.randn(6, 3)
        b2 = np.random.randn(3)
        result = forward_pass(X, [w1, w2], [b1, b2], "relu")
        assert result[0].shape == (batch_size, 6)
        assert result[1].shape == (batch_size, 3)


class TestCompareArchitectures:
    def test_sorted_by_param_count(self):
        configs = {
            "Large": [64, 512, 10],
            "Small": [64, 32, 10],
            "Medium": [64, 128, 10],
        }
        result = compare_architectures(configs)
        counts = [r[1] for r in result]
        assert counts == sorted(counts)

    def test_correct_names_returned(self):
        configs = {
            "A": [64, 128, 10],
            "B": [64, 32, 10],
        }
        result = compare_architectures(configs)
        names = {r[0] for r in result}
        assert names == {"A", "B"}

    def test_correct_param_counts(self):
        configs = {
            "Net1": [10, 5, 2],
            "Net2": [10, 20, 2],
        }
        result = compare_architectures(configs)
        result_dict = dict(result)
        assert result_dict["Net1"] == (10 * 5 + 5) + (5 * 2 + 2)
        assert result_dict["Net2"] == (10 * 20 + 20) + (20 * 2 + 2)

    def test_returns_list_of_tuples(self):
        configs = {"X": [5, 3, 1]}
        result = compare_architectures(configs)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_smallest_first(self):
        configs = {
            "Wide-Shallow": [64, 512, 10],
            "Narrow-Deep": [64, 32, 32, 32, 32, 10],
            "Balanced": [64, 128, 128, 10],
        }
        result = compare_architectures(configs)
        assert result[0][0] == "Narrow-Deep"
