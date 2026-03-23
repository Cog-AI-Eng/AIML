# AIML-AM-AppliedML-DeepLearning

**Activity Type:** Assignment  
**Duration:** 120 minutes  
**Module:** 004 -- Applied ML: Deep Learning

---

## Overview

In this assignment you will demonstrate your understanding of deep learning architecture concepts by implementing computational functions that analyze CNN and RNN architectures. No deep learning framework is required -- all exercises use NumPy and pure Python.

You will work through five milestones:

1. **CNN Dimension Calculations** -- Implement functions to compute convolution and pooling output sizes, and count layer parameters.
2. **CNN Architecture Tracing** -- Trace feature map shapes through a complete CNN, compute receptive fields, and calculate total network parameters.
3. **RNN Hidden State Computation** -- Implement vanilla RNN and LSTM cell forward passes using NumPy.
4. **RNN/LSTM/GRU Parameter Counts** -- Calculate parameter counts for LSTM and GRU layers.
5. **Architecture Tradeoffs & Analysis** -- Analyze GRU vs LSTM tradeoffs and encoder-decoder information bottlenecks.

---

## Learning Objectives

- Compute output spatial dimensions for convolution and pooling operations.
- Calculate the number of trainable parameters in convolutional and fully connected layers.
- Trace feature map shapes through a multi-layer CNN architecture.
- Compute receptive field sizes for stacked convolutional and pooling layers.
- Implement vanilla RNN and LSTM cell forward passes from first principles.
- Compare parameter counts and architectural tradeoffs between LSTM and GRU.
- Analyze the information bottleneck in encoder-decoder architectures.

---

## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| NumPy | latest stable |
| pytest | latest stable |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
assignment/
  README.md                # This file
  requirements.txt         # Python dependencies
  .gitignore               # Git ignore rules
  src/
    cnn_concepts.py        # TODO: Implement CNN conceptual exercises
    rnn_concepts.py        # TODO: Implement RNN conceptual exercises
  tests/
    test_cnn_concepts.py   # Tests for CNN exercises
    test_rnn_concepts.py   # Tests for RNN exercises
  solutions/
    cnn_concepts.py        # Reference solution
    rnn_concepts.py        # Reference solution
```

---

## Instructions

### Milestone 1: CNN Dimension Calculations (20 min)

Open `src/cnn_concepts.py`. Implement these three functions:

**`conv_output_size(input_size, kernel_size, stride, padding)`**

Apply the standard convolution output formula:

```
output = floor((input_size - kernel_size + 2 * padding) / stride) + 1
```

Example: `conv_output_size(32, 3, 1, 1)` should return `32` (a 3x3 conv with padding 1 preserves spatial size).

**`conv_layer_params(in_channels, out_channels, kernel_size, bias)`**

Count trainable parameters in a convolution layer:

```
params = in_channels * out_channels * kernel_size^2 + (out_channels if bias else 0)
```

Example: `conv_layer_params(3, 32, 3, True)` should return `896` (3 * 32 * 9 + 32).

**`pooling_output_size(input_size, pool_size, stride)`**

Apply the pooling output formula (stride defaults to pool\_size when None):

```
output = floor((input_size - pool_size) / stride) + 1
```

Example: `pooling_output_size(32, 2)` should return `16`.

Run tests:

```bash
pytest tests/test_cnn_concepts.py::TestConvOutputSize tests/test_cnn_concepts.py::TestConvLayerParams tests/test_cnn_concepts.py::TestPoolingOutputSize -v
```

### Milestone 2: CNN Architecture Tracing (25 min)

Continue in `src/cnn_concepts.py`. Implement:

**`trace_cnn_forward(input_shape, layers)`**

Given an input shape `(C, H, W)` and a list of layer config dicts, compute the output shape after each layer. Supported layer types:

- `"conv"`: `{"type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}` -- stride defaults to 1, padding defaults to 0.
- `"pool"`: `{"type": "pool", "pool_size": 2, "stride": 2}` -- stride defaults to pool\_size.
- `"flatten"`: `{"type": "flatten"}` -- converts `(C, H, W)` to `(C*H*W,)`.
- `"linear"`: `{"type": "linear", "out_features": 256}`.

Returns a list of output shape tuples, one per layer.

**`compute_receptive_field(layers)`**

Compute how many input pixels influence one output pixel. Starting at RF = 1, for each layer:

```
RF = RF + (kernel_size - 1) * cumulative_stride
cumulative_stride *= stride
```

Each layer dict must have `"kernel_size"` and `"stride"` keys.

**`total_network_params(conv_configs, linear_configs)`**

Sum all trainable parameters across conv and linear layers. Each conv config has `"in_channels"`, `"out_channels"`, `"kernel_size"`, `"bias"`. Each linear config has `"in_features"`, `"out_features"`, `"bias"`.

Run tests:

```bash
pytest tests/test_cnn_concepts.py::TestTraceCnnForward tests/test_cnn_concepts.py::TestComputeReceptiveField tests/test_cnn_concepts.py::TestTotalNetworkParams -v
```

### Milestone 3: RNN Hidden State Computation (25 min)

Open `src/rnn_concepts.py`. Implement:

**`vanilla_rnn_step(W_xh, W_hh, b_h, x_t, h_prev)`**

Compute one time step of a vanilla RNN:

```
h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
```

**`lstm_cell_forward(x_t, h_prev, c_prev, W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o)`**

Compute one LSTM time step. Concatenate `[h_prev, x_t]`, then:

```
f_t = sigmoid(W_f @ concat + b_f)     # forget gate
i_t = sigmoid(W_i @ concat + b_i)     # input gate
g_t = tanh(W_g @ concat + b_g)        # cell candidate
o_t = sigmoid(W_o @ concat + b_o)     # output gate
c_t = f_t * c_prev + i_t * g_t        # new cell state
h_t = o_t * tanh(c_t)                 # new hidden state
```

Returns `(h_t, c_t)`.

Run tests:

```bash
pytest tests/test_rnn_concepts.py::TestVanillaRnnStep tests/test_rnn_concepts.py::TestLstmCellForward -v
```

### Milestone 4: RNN/LSTM/GRU Parameter Counts (25 min)

Continue in `src/rnn_concepts.py`. Implement:

**`lstm_parameter_count(input_size, hidden_size, bias)`**

An LSTM has 4 gate-like operations (forget, input, candidate, output). Each has:
- Input-to-hidden weights: `input_size * hidden_size`
- Hidden-to-hidden weights: `hidden_size * hidden_size`
- Bias (if enabled): `hidden_size`

```
Total = 4 * (input_size * hidden_size + hidden_size * hidden_size + (hidden_size if bias else 0))
```

**`gru_parameter_count(input_size, hidden_size, bias)`**

A GRU has 3 gates (reset, update, candidate). Same per-gate structure as LSTM:

```
Total = 3 * (input_size * hidden_size + hidden_size * hidden_size + (hidden_size if bias else 0))
```

Run tests:

```bash
pytest tests/test_rnn_concepts.py::TestLstmParameterCount tests/test_rnn_concepts.py::TestGruParameterCount -v
```

### Milestone 5: Architecture Tradeoffs & Analysis (25 min)

Continue in `src/rnn_concepts.py`. Implement:

**`gru_vs_lstm_tradeoffs()`**

Return a dictionary with the following keys and correct values:

| Key | Type | Description |
|-----|------|-------------|
| `"lstm_gate_count"` | int | Number of gate-like operations in LSTM (4) |
| `"gru_gate_count"` | int | Number of gates in GRU (3) |
| `"lstm_has_cell_state"` | bool | Whether LSTM maintains a separate cell state (True) |
| `"gru_has_cell_state"` | bool | Whether GRU maintains a separate cell state (False) |
| `"fewer_parameters"` | str | Which architecture has fewer parameters (`"GRU"`) |
| `"gru_param_ratio"` | float | Ratio of GRU to LSTM parameters (`0.75`) |

**`encoder_decoder_bottleneck(input_seq_len, input_dim, hidden_dim)`**

Analyze the information bottleneck in a basic encoder-decoder (seq2seq) model without attention. Compute:

| Key | Formula |
|-----|---------|
| `"total_input_features"` | `input_seq_len * input_dim` |
| `"bottleneck_size"` | `hidden_dim` |
| `"compression_ratio"` | `hidden_dim / (input_seq_len * input_dim)` |

Run tests:

```bash
pytest tests/test_rnn_concepts.py::TestGruVsLstmTradeoffs tests/test_rnn_concepts.py::TestEncoderDecoderBottleneck -v
```

---

## Running All Tests

```bash
# All tests (should FAIL on starter code, PASS on completed code)
pytest tests/ -v

# Individual test files
pytest tests/test_cnn_concepts.py -v
pytest tests/test_rnn_concepts.py -v
```

---

## Submission Checklist

- [ ] All functions in `src/cnn_concepts.py` implemented and passing tests.
- [ ] All functions in `src/rnn_concepts.py` implemented and passing tests.
- [ ] `pytest tests/ -v` shows all tests passing.
- [ ] Code is clean, follows PEP 8, and contains no unnecessary comments.
