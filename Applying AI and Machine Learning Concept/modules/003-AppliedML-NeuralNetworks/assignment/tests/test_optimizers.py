"""Tests for Milestone 5: Optimizer Comparison and Learning Rate Scheduling."""

import numpy as np
import pytest

from src.optimizers import (
    sgd_step,
    sgd_momentum_step,
    adam_step,
    step_decay_lr,
    exponential_decay_lr,
)


class TestSGDStep:
    def test_basic_update(self):
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        lr = 0.1
        result = sgd_step(param, grad, lr)
        expected = param - lr * grad
        np.testing.assert_allclose(result, expected)

    def test_zero_gradient(self):
        param = np.array([5.0, -3.0])
        grad = np.zeros(2)
        result = sgd_step(param, grad, 0.01)
        np.testing.assert_array_equal(result, param)

    def test_shape_preserved(self):
        param = np.random.randn(3, 4)
        grad = np.random.randn(3, 4)
        result = sgd_step(param, grad, 0.1)
        assert result.shape == param.shape

    def test_large_lr_moves_far(self):
        param = np.array([1.0])
        grad = np.array([1.0])
        result = sgd_step(param, grad, 10.0)
        assert result[0] == -9.0


class TestSGDMomentumStep:
    def test_first_step_from_zero_velocity(self):
        param = np.array([1.0])
        grad = np.array([0.5])
        velocity = np.zeros(1)
        lr = 0.1
        momentum = 0.9
        new_param, new_vel = sgd_momentum_step(
            param, grad, velocity, lr, momentum
        )
        expected_vel = -lr * grad
        expected_param = param + expected_vel
        np.testing.assert_allclose(new_vel, expected_vel)
        np.testing.assert_allclose(new_param, expected_param)

    def test_momentum_accumulates(self):
        param = np.array([0.0])
        grad = np.array([1.0])
        velocity = np.array([-0.1])
        lr = 0.01
        momentum = 0.9
        new_param, new_vel = sgd_momentum_step(
            param, grad, velocity, lr, momentum
        )
        expected_vel = momentum * velocity - lr * grad
        expected_param = param + expected_vel
        np.testing.assert_allclose(new_vel, expected_vel)
        np.testing.assert_allclose(new_param, expected_param)

    def test_returns_tuple(self):
        result = sgd_momentum_step(
            np.array([1.0]), np.array([0.1]), np.zeros(1), 0.01
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shapes_preserved(self):
        shape = (3, 4)
        param = np.random.randn(*shape)
        grad = np.random.randn(*shape)
        velocity = np.zeros(shape)
        new_param, new_vel = sgd_momentum_step(param, grad, velocity, 0.01)
        assert new_param.shape == shape
        assert new_vel.shape == shape


class TestAdamStep:
    def test_first_step(self):
        param = np.array([1.0])
        grad = np.array([0.5])
        m = np.zeros(1)
        v = np.zeros(1)
        t = 1
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        new_param, new_m, new_v = adam_step(
            param, grad, m, v, t, lr, beta1, beta2, epsilon
        )
        expected_m = (1 - beta1) * grad
        expected_v = (1 - beta2) * grad ** 2
        m_hat = expected_m / (1 - beta1 ** t)
        v_hat = expected_v / (1 - beta2 ** t)
        expected_param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        np.testing.assert_allclose(new_m, expected_m)
        np.testing.assert_allclose(new_v, expected_v)
        np.testing.assert_allclose(new_param, expected_param)

    def test_bias_correction(self):
        param = np.array([0.0])
        grad = np.array([1.0])
        m = np.zeros(1)
        v = np.zeros(1)
        _, m1, v1 = adam_step(param, grad, m, v, t=1)
        _, m2, v2 = adam_step(param, grad, m1, v1, t=2)
        assert m2[0] > m1[0]

    def test_returns_tuple_of_three(self):
        result = adam_step(
            np.array([1.0]), np.array([0.1]),
            np.zeros(1), np.zeros(1), t=1
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_param_moves_in_negative_gradient_direction(self):
        param = np.array([5.0])
        grad = np.array([2.0])
        new_param, _, _ = adam_step(
            param, grad, np.zeros(1), np.zeros(1), t=1
        )
        assert new_param[0] < param[0]

    def test_shapes_preserved(self):
        shape = (2, 3)
        param = np.random.randn(*shape)
        grad = np.random.randn(*shape)
        m = np.zeros(shape)
        v = np.zeros(shape)
        new_param, new_m, new_v = adam_step(param, grad, m, v, t=1)
        assert new_param.shape == shape
        assert new_m.shape == shape
        assert new_v.shape == shape


class TestStepDecayLR:
    def test_no_decay_initially(self):
        lr = step_decay_lr(0.1, epoch=0, drop_factor=0.5, drop_every=10)
        assert pytest.approx(lr) == 0.1

    def test_first_drop(self):
        lr = step_decay_lr(0.1, epoch=10, drop_factor=0.5, drop_every=10)
        assert pytest.approx(lr) == 0.05

    def test_second_drop(self):
        lr = step_decay_lr(0.1, epoch=20, drop_factor=0.5, drop_every=10)
        assert pytest.approx(lr) == 0.025

    def test_between_drops(self):
        lr5 = step_decay_lr(0.1, epoch=5, drop_factor=0.5, drop_every=10)
        lr9 = step_decay_lr(0.1, epoch=9, drop_factor=0.5, drop_every=10)
        assert pytest.approx(lr5) == 0.1
        assert pytest.approx(lr9) == 0.1

    def test_custom_drop_factor(self):
        lr = step_decay_lr(1.0, epoch=10, drop_factor=0.1, drop_every=10)
        assert pytest.approx(lr) == 0.1


class TestExponentialDecayLR:
    def test_no_decay_at_epoch_zero(self):
        lr = exponential_decay_lr(0.1, epoch=0, decay_rate=0.95)
        assert pytest.approx(lr) == 0.1

    def test_first_epoch(self):
        lr = exponential_decay_lr(0.1, epoch=1, decay_rate=0.95)
        assert pytest.approx(lr) == 0.1 * 0.95

    def test_tenth_epoch(self):
        lr = exponential_decay_lr(0.1, epoch=10, decay_rate=0.95)
        assert pytest.approx(lr) == 0.1 * (0.95 ** 10)

    def test_monotonically_decreasing(self):
        lrs = [exponential_decay_lr(0.1, e, 0.95) for e in range(50)]
        for i in range(1, len(lrs)):
            assert lrs[i] < lrs[i - 1]

    def test_never_negative(self):
        lrs = [exponential_decay_lr(0.1, e, 0.95) for e in range(200)]
        assert all(lr > 0 for lr in lrs)
