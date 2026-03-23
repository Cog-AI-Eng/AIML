"""Tests for Milestone 2: Activation Function Properties."""

import numpy as np
import pytest

from src.activations import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh_activation,
    tanh_derivative,
    classify_activation,
)


class TestSigmoid:
    def test_zero(self):
        assert pytest.approx(sigmoid(np.array([0.0]))[0]) == 0.5

    def test_large_positive(self):
        result = sigmoid(np.array([100.0]))[0]
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_large_negative(self):
        result = sigmoid(np.array([-100.0]))[0]
        assert pytest.approx(result, abs=1e-6) == 0.0

    def test_output_range(self):
        x = np.linspace(-10, 10, 1000)
        result = sigmoid(x)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_monotonic(self):
        x = np.linspace(-5, 5, 100)
        result = sigmoid(x)
        assert np.all(np.diff(result) > 0)

    def test_symmetry(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            sigmoid(x) + sigmoid(-x), np.ones(3), atol=1e-10
        )


class TestSigmoidDerivative:
    def test_at_zero(self):
        result = sigmoid_derivative(np.array([0.0]))[0]
        assert pytest.approx(result) == 0.25

    def test_positive(self):
        x = np.array([1.0])
        s = sigmoid(x)[0]
        expected = s * (1 - s)
        assert pytest.approx(sigmoid_derivative(x)[0]) == expected

    def test_always_positive(self):
        x = np.linspace(-10, 10, 1000)
        result = sigmoid_derivative(x)
        assert np.all(result > 0)

    def test_max_at_zero(self):
        x = np.linspace(-5, 5, 1001)
        result = sigmoid_derivative(x)
        max_idx = np.argmax(result)
        assert abs(x[max_idx]) < 0.1


class TestRelu:
    def test_positive_values(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_values(self):
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu(x), np.zeros(3))

    def test_mixed(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_zero(self):
        assert relu(np.array([0.0]))[0] == 0.0


class TestReluDerivative:
    def test_positive(self):
        x = np.array([1.0, 5.0, 0.5])
        np.testing.assert_array_equal(relu_derivative(x), np.ones(3))

    def test_negative(self):
        x = np.array([-1.0, -5.0, -0.5])
        np.testing.assert_array_equal(relu_derivative(x), np.zeros(3))

    def test_at_zero(self):
        assert relu_derivative(np.array([0.0]))[0] == 0.0

    def test_mixed(self):
        x = np.array([-2.0, 3.0, 0.0, -1.0, 5.0])
        expected = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu_derivative(x), expected)


class TestTanhActivation:
    def test_zero(self):
        assert pytest.approx(tanh_activation(np.array([0.0]))[0]) == 0.0

    def test_output_range(self):
        x = np.linspace(-10, 10, 1000)
        result = tanh_activation(x)
        assert np.all(result > -1)
        assert np.all(result < 1)

    def test_antisymmetric(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            tanh_activation(x), -tanh_activation(-x), atol=1e-10
        )

    def test_known_value(self):
        result = tanh_activation(np.array([1.0]))[0]
        assert pytest.approx(result, abs=1e-6) == np.tanh(1.0)


class TestTanhDerivative:
    def test_at_zero(self):
        result = tanh_derivative(np.array([0.0]))[0]
        assert pytest.approx(result) == 1.0

    def test_always_positive(self):
        x = np.linspace(-5, 5, 1000)
        result = tanh_derivative(x)
        assert np.all(result > 0)

    def test_formula(self):
        x = np.array([0.5, 1.0, -1.0])
        expected = 1 - np.tanh(x) ** 2
        np.testing.assert_allclose(tanh_derivative(x), expected, atol=1e-10)


class TestClassifyActivation:
    def test_sigmoid_range(self):
        props = classify_activation("sigmoid")
        assert props["output_range"] == (0, 1)

    def test_sigmoid_not_zero_centered(self):
        props = classify_activation("sigmoid")
        assert props["zero_centered"] is False

    def test_sigmoid_saturating(self):
        props = classify_activation("sigmoid")
        assert props["saturating"] is True

    def test_sigmoid_no_dead_neuron(self):
        props = classify_activation("sigmoid")
        assert props["dead_neuron_risk"] is False

    def test_tanh_range(self):
        props = classify_activation("tanh")
        assert props["output_range"] == (-1, 1)

    def test_tanh_zero_centered(self):
        props = classify_activation("tanh")
        assert props["zero_centered"] is True

    def test_tanh_saturating(self):
        props = classify_activation("tanh")
        assert props["saturating"] is True

    def test_tanh_no_dead_neuron(self):
        props = classify_activation("tanh")
        assert props["dead_neuron_risk"] is False

    def test_relu_range(self):
        props = classify_activation("relu")
        assert props["output_range"] == (0, None)

    def test_relu_not_zero_centered(self):
        props = classify_activation("relu")
        assert props["zero_centered"] is False

    def test_relu_not_saturating(self):
        props = classify_activation("relu")
        assert props["saturating"] is False

    def test_relu_dead_neuron_risk(self):
        props = classify_activation("relu")
        assert props["dead_neuron_risk"] is True
