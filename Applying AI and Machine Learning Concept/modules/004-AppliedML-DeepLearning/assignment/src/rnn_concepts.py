"""
Conceptual exercises for Recurrent Neural Networks (RNN, LSTM, GRU).

Associates must implement functions that compute hidden state updates,
parameter counts, and analyze architecture tradeoffs.
"""

import numpy as np


def sigmoid(x):
    """Sigmoid activation function (provided for convenience)."""
    return 1.0 / (1.0 + np.exp(-x))


def vanilla_rnn_step(W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray,
                     x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    """Compute one time step of a vanilla RNN.

    Update rule:
        h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

    Args:
        W_xh: Weight matrix from input to hidden,
              shape (hidden_size, input_size).
        W_hh: Weight matrix from hidden to hidden,
              shape (hidden_size, hidden_size).
        b_h: Bias vector, shape (hidden_size,).
        x_t: Input vector at time t, shape (input_size,).
        h_prev: Hidden state from previous time step,
                shape (hidden_size,).

    Returns:
        h_t: New hidden state, shape (hidden_size,).
    """
    raise NotImplementedError("Implement vanilla_rnn_step")


def lstm_parameter_count(input_size: int, hidden_size: int,
                         bias: bool = True) -> int:
    """Compute the number of trainable parameters in a single LSTM layer.

    An LSTM has four gate-like operations (forget, input, cell candidate,
    output). Each has:
        - Input-to-hidden weights: input_size * hidden_size
        - Hidden-to-hidden weights: hidden_size * hidden_size
        - Bias (if bias=True): hidden_size

    Total = 4 * (input_size * hidden_size + hidden_size * hidden_size
                 + (hidden_size if bias else 0))

    Args:
        input_size: Dimensionality of the input at each time step.
        hidden_size: Number of hidden units.
        bias: Whether bias parameters are included.

    Returns:
        Total number of trainable parameters.
    """
    raise NotImplementedError("Implement lstm_parameter_count")


def gru_parameter_count(input_size: int, hidden_size: int,
                        bias: bool = True) -> int:
    """Compute the number of trainable parameters in a single GRU layer.

    A GRU has three gates (reset, update, candidate). Each gate has
    the same parameter structure as an LSTM gate.

    Total = 3 * (input_size * hidden_size + hidden_size * hidden_size
                 + (hidden_size if bias else 0))

    Args:
        input_size: Dimensionality of the input at each time step.
        hidden_size: Number of hidden units.
        bias: Whether bias parameters are included.

    Returns:
        Total number of trainable parameters.
    """
    raise NotImplementedError("Implement gru_parameter_count")


def lstm_cell_forward(x_t: np.ndarray, h_prev: np.ndarray,
                      c_prev: np.ndarray,
                      W_f: np.ndarray, W_i: np.ndarray,
                      W_g: np.ndarray, W_o: np.ndarray,
                      b_f: np.ndarray, b_i: np.ndarray,
                      b_g: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute one time step of an LSTM cell.

    First concatenate h_prev and x_t into a single vector:
        concat = [h_prev, x_t]   (shape: hidden_size + input_size)

    Gate computations:
        f_t = sigmoid(W_f @ concat + b_f)   (forget gate)
        i_t = sigmoid(W_i @ concat + b_i)   (input gate)
        g_t = tanh(W_g @ concat + b_g)      (cell candidate)
        o_t = sigmoid(W_o @ concat + b_o)   (output gate)

    State updates:
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)

    Args:
        x_t: Input vector, shape (input_size,).
        h_prev: Previous hidden state, shape (hidden_size,).
        c_prev: Previous cell state, shape (hidden_size,).
        W_f, W_i, W_g, W_o: Weight matrices,
            shape (hidden_size, hidden_size + input_size).
        b_f, b_i, b_g, b_o: Bias vectors, shape (hidden_size,).

    Returns:
        Tuple of (h_t, c_t), each of shape (hidden_size,).
    """
    raise NotImplementedError("Implement lstm_cell_forward")


def gru_vs_lstm_tradeoffs() -> dict:
    """Return a dictionary analyzing GRU vs LSTM architectural tradeoffs.

    The returned dict must have exactly these keys with correct values:

        "lstm_gate_count": int
            Number of gate-like operations in an LSTM cell.
            (4: forget, input, candidate, output)
        "gru_gate_count": int
            Number of gates in a GRU cell.
            (3: reset, update, candidate)
        "lstm_has_cell_state": bool
            Whether LSTM maintains a separate cell state.
        "gru_has_cell_state": bool
            Whether GRU maintains a separate cell state.
        "fewer_parameters": str
            Which architecture has fewer parameters ("GRU" or "LSTM").
        "gru_param_ratio": float
            Ratio of GRU parameters to LSTM parameters (3/4 = 0.75).

    Returns:
        Dictionary with the keys and types described above.
    """
    raise NotImplementedError("Implement gru_vs_lstm_tradeoffs")


def encoder_decoder_bottleneck(input_seq_len: int, input_dim: int,
                               hidden_dim: int) -> dict:
    """Analyze the information bottleneck in a basic encoder-decoder model.

    In a seq2seq model without attention, the encoder compresses the entire
    input sequence into a single fixed-size context vector (the final
    hidden state). This creates an information bottleneck.

    Compute:
        total_input_features = input_seq_len * input_dim
        bottleneck_size = hidden_dim
        compression_ratio = bottleneck_size / total_input_features

    Args:
        input_seq_len: Length of the input sequence.
        input_dim: Dimensionality of each input token/timestep.
        hidden_dim: Dimensionality of the encoder hidden state.

    Returns:
        Dict with keys:
            "total_input_features": int
            "bottleneck_size": int
            "compression_ratio": float
    """
    raise NotImplementedError("Implement encoder_decoder_bottleneck")
