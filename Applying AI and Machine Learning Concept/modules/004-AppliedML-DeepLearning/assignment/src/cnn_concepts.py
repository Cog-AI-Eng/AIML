"""
Conceptual exercises for Convolutional Neural Networks.

Associates must implement functions that compute CNN output dimensions,
parameter counts, and trace information flow through network architectures.
"""

import numpy as np


def conv_output_size(input_size: int, kernel_size: int,
                     stride: int = 1, padding: int = 0) -> int:
    """Compute the spatial output size after a convolution operation.

    Formula:
        output = floor((input_size - kernel_size + 2 * padding) / stride) + 1

    Args:
        input_size: Height (or width) of the input feature map.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides.

    Returns:
        The spatial dimension (height or width) of the output feature map.
    """
    raise NotImplementedError("Implement conv_output_size")


def conv_layer_params(in_channels: int, out_channels: int,
                      kernel_size: int, bias: bool = True) -> int:
    """Compute the number of trainable parameters in a convolution layer.

    A conv layer has:
        - Weight parameters: in_channels * out_channels * kernel_size^2
        - Bias parameters (if bias=True): out_channels

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Spatial size of each filter (assumed square).
        bias: Whether the layer includes a bias term.

    Returns:
        Total number of trainable parameters.
    """
    raise NotImplementedError("Implement conv_layer_params")


def pooling_output_size(input_size: int, pool_size: int,
                        stride: int = None) -> int:
    """Compute the spatial output size after a pooling operation.

    Formula:
        output = floor((input_size - pool_size) / stride) + 1

    When stride is None, it defaults to pool_size (non-overlapping pooling).

    Args:
        input_size: Height (or width) of the input feature map.
        pool_size: Size of the pooling window.
        stride: Stride of the pooling operation. Defaults to pool_size.

    Returns:
        The spatial dimension of the output after pooling.
    """
    raise NotImplementedError("Implement pooling_output_size")


def trace_cnn_forward(input_shape: tuple, layers: list) -> list:
    """Trace the output shape after each layer in a CNN architecture.

    Given an input shape (C, H, W) and a list of layer configurations,
    compute the output shape after each layer.

    Supported layer types and their config keys:
        - "conv": {"type": "conv", "out_channels": int, "kernel_size": int,
                    "stride": int (default 1), "padding": int (default 0)}
        - "pool": {"type": "pool", "pool_size": int,
                    "stride": int (default pool_size)}
        - "flatten": {"type": "flatten"} -- converts (C, H, W) to (C*H*W,)
        - "linear": {"type": "linear", "out_features": int}

    Activations (ReLU), dropout, and batch normalization do not change
    shapes and are not included as layer types.

    Args:
        input_shape: Tuple of (channels, height, width) for the input.
        layers: List of layer configuration dicts.

    Returns:
        List of output shape tuples after each layer.
        Conv/pool layers produce (C, H, W) tuples.
        Flatten produces a 1-tuple (N,).
        Linear produces a 1-tuple (out_features,).
    """
    raise NotImplementedError("Implement trace_cnn_forward")


def compute_receptive_field(layers: list) -> int:
    """Compute the receptive field size of a stack of conv/pool layers.

    The receptive field is the region of the input that influences a single
    output neuron. Starting with RF = 1, for each layer:

        RF = RF + (kernel_size - 1) * cumulative_stride
        cumulative_stride *= stride

    Each layer dict must have:
        - "kernel_size": int
        - "stride": int

    Args:
        layers: List of dicts, each with "kernel_size" and "stride".

    Returns:
        The receptive field size in input pixels.
    """
    raise NotImplementedError("Implement compute_receptive_field")


def total_network_params(conv_configs: list, linear_configs: list) -> int:
    """Compute the total trainable parameters in a CNN.

    Args:
        conv_configs: List of dicts, each with keys:
            "in_channels", "out_channels", "kernel_size",
            "bias" (optional, default True)
        linear_configs: List of dicts, each with keys:
            "in_features", "out_features",
            "bias" (optional, default True)

    Returns:
        Total number of trainable parameters across all layers.
    """
    raise NotImplementedError("Implement total_network_params")
