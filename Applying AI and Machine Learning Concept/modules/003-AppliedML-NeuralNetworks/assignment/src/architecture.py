"""Milestone 3: MLP Architecture Design

Implement functions to analyze MLP architectures: count parameters, trace
forward passes manually, and compare different designs.
"""

import numpy as np


def count_parameters(layer_dims: list) -> int:
    """Count the total trainable parameters in a fully-connected MLP.

    Each pair of consecutive dimensions (layer_dims[i], layer_dims[i+1])
    forms one linear layer with:
      - Weight matrix: layer_dims[i] * layer_dims[i+1] parameters
      - Bias vector:   layer_dims[i+1] parameters

    Args:
        layer_dims: List of layer sizes including input and output.
            Example: [64, 128, 64, 10] means input=64, two hidden
            layers (128, 64), output=10.

    Returns:
        Total number of trainable parameters (int).
    """
    # TODO: Sum (layer_dims[i] * layer_dims[i+1] + layer_dims[i+1])
    #       for all consecutive pairs
    raise NotImplementedError("Implement count_parameters")


def forward_pass(X: np.ndarray, weights: list, biases: list,
                 activation: str = "relu") -> list:
    """Manually compute a forward pass through an MLP using NumPy.

    For each layer i (except the last):
        z = previous_output @ weights[i] + biases[i]
        output = activation(z)
    For the final layer:
        output = previous_output @ weights[-1] + biases[-1]  (no activation)

    Supported activations: "relu", "sigmoid", "tanh".

    Args:
        X: Input array of shape (batch_size, input_dim).
        weights: List of weight matrices. weights[i] has shape
            (dim_i, dim_{i+1}).
        biases: List of bias vectors. biases[i] has shape (dim_{i+1},).
        activation: Name of activation function for hidden layers.

    Returns:
        List of all layer outputs (post-activation for hidden layers,
        raw for the output layer). Length equals len(weights).
    """
    # TODO: Implement the forward pass using only numpy operations.
    #       Apply the activation after each hidden layer.
    #       The final layer should NOT have an activation applied.
    raise NotImplementedError("Implement forward_pass")


def compare_architectures(configs: dict) -> list:
    """Compare MLP architectures by parameter count.

    Args:
        configs: Dict mapping architecture names (str) to layer_dims lists.
            Example: {
                "Wide-Shallow": [64, 512, 10],
                "Narrow-Deep": [64, 32, 32, 32, 32, 10],
                "Balanced": [64, 128, 128, 10],
            }

    Returns:
        List of tuples (name, param_count) sorted by param_count ascending.
    """
    # TODO: Compute parameter counts and return sorted list
    raise NotImplementedError("Implement compare_architectures")
