"""Tests for Milestone 1: Perceptron Mathematics."""

import numpy as np
import pytest

from src.perceptron import (
    perceptron_output,
    perceptron_parameter_count,
    decision_boundary_2d,
)


class TestPerceptronOutput:
    def test_single_sample_single_output(self):
        inputs = np.array([[1.0, 2.0, 3.0]])
        weights = np.array([[0.1], [0.2], [0.3]])
        bias = np.array([0.5])
        result = perceptron_output(inputs, weights, bias)
        expected = np.array([[1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 0.5]])
        np.testing.assert_allclose(result, expected)

    def test_batch_output(self):
        inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = np.array([[0.5, -0.5], [0.3, 0.7]])
        bias = np.array([0.1, -0.1])
        result = perceptron_output(inputs, weights, bias)
        expected = inputs @ weights + bias
        np.testing.assert_allclose(result, expected)

    def test_output_shape(self):
        inputs = np.random.randn(16, 64)
        weights = np.random.randn(64, 10)
        bias = np.random.randn(10)
        result = perceptron_output(inputs, weights, bias)
        assert result.shape == (16, 10)

    def test_zero_weights_returns_bias(self):
        inputs = np.random.randn(5, 3)
        weights = np.zeros((3, 2))
        bias = np.array([1.0, -1.0])
        result = perceptron_output(inputs, weights, bias)
        for i in range(5):
            np.testing.assert_allclose(result[i], bias)

    def test_zero_bias(self):
        inputs = np.array([[2.0, 3.0]])
        weights = np.array([[1.0], [1.0]])
        bias = np.array([0.0])
        result = perceptron_output(inputs, weights, bias)
        np.testing.assert_allclose(result, np.array([[5.0]]))


class TestPerceptronParameterCount:
    def test_small_perceptron(self):
        assert perceptron_parameter_count(2, 1) == 3

    def test_digit_classifier(self):
        assert perceptron_parameter_count(64, 10) == 64 * 10 + 10

    def test_single_input_single_output(self):
        assert perceptron_parameter_count(1, 1) == 2

    def test_large_perceptron(self):
        assert perceptron_parameter_count(784, 10) == 784 * 10 + 10

    def test_multi_output(self):
        assert perceptron_parameter_count(5, 3) == 5 * 3 + 3


class TestDecisionBoundary2D:
    def test_simple_boundary(self):
        slope, intercept = decision_boundary_2d(1.0, 1.0, 0.0)
        assert pytest.approx(slope) == -1.0
        assert pytest.approx(intercept) == 0.0

    def test_with_bias(self):
        slope, intercept = decision_boundary_2d(2.0, 1.0, -4.0)
        assert pytest.approx(slope) == -2.0
        assert pytest.approx(intercept) == 4.0

    def test_negative_weights(self):
        slope, intercept = decision_boundary_2d(-3.0, 2.0, 1.0)
        assert pytest.approx(slope) == 3.0 / 2.0
        assert pytest.approx(intercept) == -1.0 / 2.0

    def test_boundary_passes_through_origin(self):
        slope, intercept = decision_boundary_2d(1.0, 2.0, 0.0)
        assert pytest.approx(intercept) == 0.0

    def test_point_on_boundary(self):
        w1, w2, bias = 2.0, -3.0, 6.0
        slope, intercept = decision_boundary_2d(w1, w2, bias)
        x1 = 0.0
        x2 = slope * x1 + intercept
        assert pytest.approx(w1 * x1 + w2 * x2 + bias) == 0.0
