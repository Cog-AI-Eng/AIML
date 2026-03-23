"""Milestone 4: Backpropagation and the Chain Rule

Implement gradient computations by hand for small neural networks.
These exercises reinforce how backpropagation applies the chain rule
to compute parameter gradients layer by layer.
"""

import numpy as np


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Squared Error loss.

    MSE = (1/n) * sum((predictions - targets)^2)

    Args:
        predictions: Model output, shape (n,) or (n, d).
        targets: Ground truth, same shape as predictions.

    Returns:
        Scalar MSE loss value (float).
    """
    # TODO: Implement MSE loss
    raise NotImplementedError("Implement mse_loss")


def mse_loss_gradient(predictions: np.ndarray,
                      targets: np.ndarray) -> np.ndarray:
    """Compute the gradient of MSE loss with respect to predictions.

    dL/d(predictions) = (2/n) * (predictions - targets)

    where n is the number of samples (first dimension).

    Args:
        predictions: Model output, shape (n,) or (n, d).
        targets: Ground truth, same shape as predictions.

    Returns:
        Gradient array, same shape as predictions.
    """
    # TODO: Implement MSE loss gradient
    raise NotImplementedError("Implement mse_loss_gradient")


def linear_backward(x: np.ndarray, w: np.ndarray,
                    upstream_grad: np.ndarray) -> dict:
    """Compute gradients for a single linear layer during backprop.

    For a linear layer computing z = x @ w + b, given the upstream
    gradient dL/dz, compute:
        dL/dw = x^T @ dL/dz
        dL/db = sum(dL/dz, axis=0)
        dL/dx = dL/dz @ w^T   (to pass to the layer below)

    Args:
        x: Layer input, shape (batch_size, in_dim).
        w: Weight matrix, shape (in_dim, out_dim).
        upstream_grad: dL/dz, shape (batch_size, out_dim).

    Returns:
        Dict with keys:
            "dw": gradient w.r.t. weights, shape (in_dim, out_dim)
            "db": gradient w.r.t. bias, shape (out_dim,)
            "dx": gradient w.r.t. input, shape (batch_size, in_dim)
    """
    # TODO: Implement linear layer backward pass
    raise NotImplementedError("Implement linear_backward")


def two_layer_gradients(x: np.ndarray, w1: np.ndarray, b1: np.ndarray,
                        w2: np.ndarray, b2: np.ndarray,
                        targets: np.ndarray) -> dict:
    """Compute all gradients for a 2-layer ReLU network with MSE loss.

    Network architecture:
        z1 = x @ w1 + b1
        a1 = relu(z1)
        z2 = a1 @ w2 + b2
        loss = MSE(z2, targets)

    Backward pass:
        1. dL/dz2 = mse_loss_gradient(z2, targets)
        2. Use linear_backward on layer 2 to get dw2, db2, da1
        3. dL/dz1 = da1 * relu'(z1)    (element-wise)
        4. Use linear_backward on layer 1 to get dw1, db1

    Args:
        x: Input, shape (batch_size, input_dim).
        w1: First layer weights, shape (input_dim, hidden_dim).
        b1: First layer bias, shape (hidden_dim,).
        w2: Second layer weights, shape (hidden_dim, output_dim).
        b2: Second layer bias, shape (output_dim,).
        targets: Target values, shape (batch_size, output_dim).

    Returns:
        Dict with keys:
            "loss": scalar float
            "dw1": gradient for w1
            "db1": gradient for b1
            "dw2": gradient for w2
            "db2": gradient for b2
            "z1": pre-activation of first layer
            "a1": post-activation (relu) of first layer
            "z2": output of second layer (predictions)
    """
    # TODO: Implement full forward and backward pass
    raise NotImplementedError("Implement two_layer_gradients")
