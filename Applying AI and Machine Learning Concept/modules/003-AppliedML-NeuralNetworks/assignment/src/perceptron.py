"""Milestone 1: Perceptron Mathematics

Implement functions that demonstrate understanding of how a perceptron
(single-layer linear model) computes its output, how to count its
parameters, and how to derive its decision boundary.
"""

import numpy as np


def perceptron_output(inputs: np.ndarray, weights: np.ndarray,
                      bias: np.ndarray) -> np.ndarray:
    """Compute the raw output (logits) of a perceptron.

    A perceptron computes: z = X @ W + b

    Args:
        inputs: Input array of shape (batch_size, input_dim).
        weights: Weight matrix of shape (input_dim, output_dim).
        bias: Bias vector of shape (output_dim,).

    Returns:
        Output array of shape (batch_size, output_dim).
    """
    # TODO: Compute and return inputs @ weights + bias
    raise NotImplementedError("Implement perceptron_output")


def perceptron_parameter_count(input_dim: int, output_dim: int) -> int:
    """Count the total number of trainable parameters in a perceptron.

    A perceptron has:
      - A weight matrix of shape (input_dim, output_dim)
      - A bias vector of shape (output_dim,)

    Args:
        input_dim: Number of input features.
        output_dim: Number of output units.

    Returns:
        Total number of trainable parameters (int).
    """
    # TODO: Return input_dim * output_dim + output_dim
    raise NotImplementedError("Implement perceptron_parameter_count")


def decision_boundary_2d(w1: float, w2: float, bias: float) -> tuple:
    """Compute the slope and y-intercept of a 2D perceptron's decision boundary.

    For a 2D perceptron with weights [w1, w2] and bias b, the decision
    boundary is the line where:
        w1*x1 + w2*x2 + b = 0

    Rearranging for x2:
        x2 = -(w1/w2)*x1 - (b/w2)

    So slope = -w1/w2 and intercept = -b/w2.

    Args:
        w1: Weight for the first input feature.
        w2: Weight for the second input feature (must be non-zero).
        bias: Bias term.

    Returns:
        Tuple of (slope, intercept) as floats.
    """
    # TODO: Compute and return (slope, intercept)
    raise NotImplementedError("Implement decision_boundary_2d")
