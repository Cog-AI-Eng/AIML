"""Milestone 5: Optimizer Comparison and Learning Rate Scheduling

Implement optimizer update rules and learning rate schedules from scratch
using only NumPy. Compare how different optimizers behave mathematically.
"""

import numpy as np


def sgd_step(param: np.ndarray, grad: np.ndarray,
             lr: float) -> np.ndarray:
    """Apply one step of vanilla Stochastic Gradient Descent.

    Update rule: param_new = param - lr * grad

    Args:
        param: Current parameter values, any shape.
        grad: Gradient of loss w.r.t. param, same shape.
        lr: Learning rate.

    Returns:
        Updated parameter array (same shape).
    """
    # TODO: Implement vanilla SGD update
    raise NotImplementedError("Implement sgd_step")


def sgd_momentum_step(param: np.ndarray, grad: np.ndarray,
                      velocity: np.ndarray, lr: float,
                      momentum: float = 0.9) -> tuple:
    """Apply one step of SGD with momentum.

    Update rules:
        velocity_new = momentum * velocity - lr * grad
        param_new = param + velocity_new

    Args:
        param: Current parameter values.
        grad: Gradient of loss w.r.t. param.
        velocity: Current velocity (same shape as param).
        lr: Learning rate.
        momentum: Momentum coefficient (default 0.9).

    Returns:
        Tuple of (updated_param, updated_velocity).
    """
    # TODO: Implement SGD with momentum
    raise NotImplementedError("Implement sgd_momentum_step")


def adam_step(param: np.ndarray, grad: np.ndarray,
             m: np.ndarray, v: np.ndarray, t: int,
             lr: float = 0.001, beta1: float = 0.9,
             beta2: float = 0.999,
             epsilon: float = 1e-8) -> tuple:
    """Apply one step of the Adam optimizer.

    Update rules:
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * grad^2
        m_hat = m_new / (1 - beta1^t)
        v_hat = v_new / (1 - beta2^t)
        param_new = param - lr * m_hat / (sqrt(v_hat) + epsilon)

    Args:
        param: Current parameter values.
        grad: Gradient of loss w.r.t. param.
        m: First moment estimate (same shape as param).
        v: Second moment estimate (same shape as param).
        t: Current timestep (1-indexed, for bias correction).
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        epsilon: Small constant for numerical stability.

    Returns:
        Tuple of (updated_param, updated_m, updated_v).
    """
    # TODO: Implement Adam optimizer step
    raise NotImplementedError("Implement adam_step")


def step_decay_lr(initial_lr: float, epoch: int,
                  drop_factor: float = 0.5,
                  drop_every: int = 10) -> float:
    """Compute learning rate using step decay schedule.

    lr = initial_lr * (drop_factor ^ floor(epoch / drop_every))

    Args:
        initial_lr: Starting learning rate.
        epoch: Current epoch (0-indexed).
        drop_factor: Factor to multiply LR by at each drop.
        drop_every: Drop the LR every this many epochs.

    Returns:
        Learning rate at the given epoch (float).
    """
    # TODO: Implement step decay schedule
    raise NotImplementedError("Implement step_decay_lr")


def exponential_decay_lr(initial_lr: float, epoch: int,
                         decay_rate: float = 0.95) -> float:
    """Compute learning rate using exponential decay schedule.

    lr = initial_lr * (decay_rate ^ epoch)

    Args:
        initial_lr: Starting learning rate.
        epoch: Current epoch (0-indexed).
        decay_rate: Decay rate per epoch.

    Returns:
        Learning rate at the given epoch (float).
    """
    # TODO: Implement exponential decay schedule
    raise NotImplementedError("Implement exponential_decay_lr")
