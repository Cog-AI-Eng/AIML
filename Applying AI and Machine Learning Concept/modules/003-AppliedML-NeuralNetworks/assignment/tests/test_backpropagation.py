"""Tests for Milestone 4: Backpropagation and the Chain Rule."""

import numpy as np
import pytest

from src.backpropagation import (
    mse_loss,
    mse_loss_gradient,
    linear_backward,
    two_layer_gradients,
)


class TestMSELoss:
    def test_perfect_prediction(self):
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(mse_loss(preds, targets)) == 0.0

    def test_known_value(self):
        preds = np.array([1.0, 2.0])
        targets = np.array([3.0, 4.0])
        expected = ((1 - 3) ** 2 + (2 - 4) ** 2) / 2
        assert pytest.approx(mse_loss(preds, targets)) == expected

    def test_always_non_negative(self):
        preds = np.random.randn(50)
        targets = np.random.randn(50)
        assert mse_loss(preds, targets) >= 0

    def test_2d_input(self):
        preds = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[1.0, 1.0], [1.0, 1.0]])
        expected = np.mean((preds - targets) ** 2)
        assert pytest.approx(mse_loss(preds, targets)) == expected

    def test_returns_float(self):
        result = mse_loss(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)


class TestMSELossGradient:
    def test_perfect_prediction(self):
        preds = np.array([[1.0, 2.0]])
        targets = np.array([[1.0, 2.0]])
        grad = mse_loss_gradient(preds, targets)
        np.testing.assert_allclose(grad, np.zeros_like(preds))

    def test_known_gradient(self):
        preds = np.array([[3.0]])
        targets = np.array([[1.0]])
        grad = mse_loss_gradient(preds, targets)
        expected = np.array([[2.0 / 1.0 * (3.0 - 1.0)]])
        np.testing.assert_allclose(grad, expected)

    def test_shape_preserved(self):
        preds = np.random.randn(5, 3)
        targets = np.random.randn(5, 3)
        grad = mse_loss_gradient(preds, targets)
        assert grad.shape == preds.shape

    def test_batch_gradient(self):
        preds = np.array([[1.0], [2.0], [3.0]])
        targets = np.array([[0.0], [0.0], [0.0]])
        grad = mse_loss_gradient(preds, targets)
        expected = (2.0 / 3.0) * (preds - targets)
        np.testing.assert_allclose(grad, expected)


class TestLinearBackward:
    def test_gradient_shapes(self):
        x = np.random.randn(4, 3)
        w = np.random.randn(3, 5)
        upstream = np.random.randn(4, 5)
        result = linear_backward(x, w, upstream)
        assert result["dw"].shape == (3, 5)
        assert result["db"].shape == (5,)
        assert result["dx"].shape == (4, 3)

    def test_known_gradients(self):
        x = np.array([[1.0, 2.0]])
        w = np.array([[0.5, 0.3], [0.7, 0.1]])
        upstream = np.array([[1.0, 1.0]])
        result = linear_backward(x, w, upstream)
        expected_dw = x.T @ upstream
        expected_db = np.sum(upstream, axis=0)
        expected_dx = upstream @ w.T
        np.testing.assert_allclose(result["dw"], expected_dw)
        np.testing.assert_allclose(result["db"], expected_db)
        np.testing.assert_allclose(result["dx"], expected_dx)

    def test_batch_aggregation(self):
        x = np.array([[1.0], [2.0], [3.0]])
        w = np.array([[0.5]])
        upstream = np.array([[1.0], [1.0], [1.0]])
        result = linear_backward(x, w, upstream)
        assert result["db"].shape == (1,)
        assert pytest.approx(result["db"][0]) == 3.0

    def test_has_required_keys(self):
        x = np.random.randn(2, 3)
        w = np.random.randn(3, 4)
        upstream = np.random.randn(2, 4)
        result = linear_backward(x, w, upstream)
        assert "dw" in result
        assert "db" in result
        assert "dx" in result


class TestTwoLayerGradients:
    def test_returns_required_keys(self):
        np.random.seed(42)
        x = np.random.randn(2, 3)
        w1 = np.random.randn(3, 4)
        b1 = np.random.randn(4)
        w2 = np.random.randn(4, 1)
        b2 = np.random.randn(1)
        targets = np.random.randn(2, 1)
        result = two_layer_gradients(x, w1, b1, w2, b2, targets)
        for key in ["loss", "dw1", "db1", "dw2", "db2", "z1", "a1", "z2"]:
            assert key in result, f"Missing key: {key}"

    def test_gradient_shapes(self):
        x = np.random.randn(4, 3)
        w1 = np.random.randn(3, 5)
        b1 = np.random.randn(5)
        w2 = np.random.randn(5, 2)
        b2 = np.random.randn(2)
        targets = np.random.randn(4, 2)
        result = two_layer_gradients(x, w1, b1, w2, b2, targets)
        assert result["dw1"].shape == w1.shape
        assert result["db1"].shape == b1.shape
        assert result["dw2"].shape == w2.shape
        assert result["db2"].shape == b2.shape

    def test_forward_pass_values(self):
        x = np.array([[1.0, 2.0]])
        w1 = np.array([[0.1, 0.3], [0.2, 0.4]])
        b1 = np.array([0.1, 0.1])
        w2 = np.array([[0.5], [0.6]])
        b2 = np.array([0.1])
        targets = np.array([[1.0]])
        result = two_layer_gradients(x, w1, b1, w2, b2, targets)
        np.testing.assert_allclose(result["z1"], np.array([[0.6, 1.2]]))
        np.testing.assert_allclose(result["a1"], np.array([[0.6, 1.2]]))
        np.testing.assert_allclose(result["z2"], np.array([[1.12]]),
                                   atol=1e-10)

    def test_loss_value(self):
        x = np.array([[1.0, 2.0]])
        w1 = np.array([[0.1, 0.3], [0.2, 0.4]])
        b1 = np.array([0.1, 0.1])
        w2 = np.array([[0.5], [0.6]])
        b2 = np.array([0.1])
        targets = np.array([[1.0]])
        result = two_layer_gradients(x, w1, b1, w2, b2, targets)
        expected_loss = (1.12 - 1.0) ** 2
        assert pytest.approx(result["loss"], abs=1e-6) == expected_loss

    def test_relu_zeroes_negative(self):
        x = np.array([[1.0]])
        w1 = np.array([[-1.0, 1.0]])
        b1 = np.array([0.0, 0.0])
        w2 = np.array([[1.0], [1.0]])
        b2 = np.array([0.0])
        targets = np.array([[0.5]])
        result = two_layer_gradients(x, w1, b1, w2, b2, targets)
        assert result["z1"][0, 0] < 0
        assert result["a1"][0, 0] == 0.0
        assert result["a1"][0, 1] > 0

    def test_gradient_descent_reduces_loss(self):
        np.random.seed(0)
        x = np.random.randn(10, 3)
        w1 = np.random.randn(3, 4)
        b1 = np.zeros(4)
        w2 = np.random.randn(4, 1)
        b2 = np.zeros(1)
        targets = np.random.randn(10, 1)
        lr = 0.01
        result_before = two_layer_gradients(x, w1, b1, w2, b2, targets)
        w1 = w1 - lr * result_before["dw1"]
        b1 = b1 - lr * result_before["db1"]
        w2 = w2 - lr * result_before["dw2"]
        b2 = b2 - lr * result_before["db2"]
        result_after = two_layer_gradients(x, w1, b1, w2, b2, targets)
        assert result_after["loss"] < result_before["loss"], (
            "One gradient step should reduce the loss"
        )
