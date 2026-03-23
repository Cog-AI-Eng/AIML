"""Milestone 2: Activation Function Properties

Implement activation functions and their derivatives using only NumPy.
Also classify activation functions by their mathematical properties.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function element-wise.

    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input array of any shape.

    Returns:
        Array of same shape with sigmoid applied element-wise.
    """
    # TODO: Implement sigmoid
    raise NotImplementedError("Implement sigmoid")


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Compute the derivative of sigmoid with respect to its input.

    sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    Args:
        x: Input array of any shape (pre-activation values).

    Returns:
        Array of same shape with the derivative computed element-wise.
    """
    # TODO: Implement sigmoid derivative
    raise NotImplementedError("Implement sigmoid_derivative")


def relu(x: np.ndarray) -> np.ndarray:
    """Compute the ReLU activation function element-wise.

    relu(x) = max(0, x)

    Args:
        x: Input array of any shape.

    Returns:
        Array of same shape with ReLU applied element-wise.
    """
    # TODO: Implement ReLU
    raise NotImplementedError("Implement relu")


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Compute the derivative of ReLU with respect to its input.

    relu'(x) = 1 if x > 0, else 0   (use 0 at x=0 by convention)

    Args:
        x: Input array of any shape (pre-activation values).

    Returns:
        Array of same shape with the derivative computed element-wise.
    """
    # TODO: Implement ReLU derivative
    raise NotImplementedError("Implement relu_derivative")


def tanh_activation(x: np.ndarray) -> np.ndarray:
    """Compute the tanh activation function element-wise.

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x: Input array of any shape.

    Returns:
        Array of same shape with tanh applied element-wise.
    """
    # TODO: Implement tanh (you may use np.tanh)
    raise NotImplementedError("Implement tanh_activation")


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Compute the derivative of tanh with respect to its input.

    tanh'(x) = 1 - tanh(x)^2

    Args:
        x: Input array of any shape (pre-activation values).

    Returns:
        Array of same shape with the derivative computed element-wise.
    """
    # TODO: Implement tanh derivative
    raise NotImplementedError("Implement tanh_derivative")


def classify_activation(name: str) -> dict:
    """Return mathematical properties of the named activation function.

    For each of "sigmoid", "tanh", and "relu", return a dict with:

    - "output_range": tuple (min, max).
        Use None for unbounded. E.g. relu -> (0, None).
        sigmoid -> (0, 1), tanh -> (-1, 1).
    - "zero_centered": bool
        True if the output is centered around zero.
        sigmoid: False, tanh: True, relu: False.
    - "saturating": bool
        True if the function has regions where the gradient approaches 0.
        sigmoid: True, tanh: True, relu: False.
    - "dead_neuron_risk": bool
        True if neurons can permanently stop learning.
        sigmoid: False, tanh: False, relu: True.

    Args:
        name: One of "sigmoid", "tanh", "relu".

    Returns:
        Dict with keys "output_range", "zero_centered", "saturating",
        "dead_neuron_risk".
    """
    # TODO: Return the correct properties dict for the given activation
    raise NotImplementedError("Implement classify_activation")
