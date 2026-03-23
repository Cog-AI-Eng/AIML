# AIML-AM-AppliedML-NeuralNetworks

**Activity Type:** Assignment  
**Duration:** 120 minutes  
**Module:** 003 -- Applied ML: Neural Networks

---

## Overview

In this assignment you will demonstrate your understanding of neural network fundamentals through **conceptual exercises**. Instead of relying on a specific deep learning framework, you will implement core computations from scratch using only NumPy. You will progress through five milestones:

1. **Perceptron mathematics** -- Compute perceptron outputs, count parameters, and derive decision boundaries.
2. **Activation functions** -- Implement activation functions and their derivatives; classify their mathematical properties.
3. **MLP architecture design** -- Count parameters in multi-layer networks, trace forward passes by hand, and compare architectures.
4. **Backpropagation & chain rule** -- Compute gradients through a two-layer network using MSE loss and the chain rule.
5. **Optimizer comparison & learning rate scheduling** -- Implement SGD, SGD+momentum, and Adam update rules; compute learning rate schedules.

---

## Learning Objectives

- Compute a perceptron's output and parameter count by hand.
- Implement activation functions and their derivatives using only NumPy.
- Calculate the total trainable parameters in an MLP of arbitrary depth and width.
- Trace a forward pass and backpropagation through a multi-layer network manually.
- Apply the chain rule to derive weight and bias gradients layer by layer.
- Implement SGD, SGD with momentum, and Adam update rules from their mathematical definitions.
- Compute learning rate schedules (step decay, exponential decay).

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
  README.md                 # This file
  requirements.txt          # Python dependencies
  .gitignore                # Git ignore rules
  src/
    __init__.py
    perceptron.py            # TODO: Milestone 1 -- Perceptron math
    activations.py           # TODO: Milestone 2 -- Activation functions & properties
    architecture.py          # TODO: Milestone 3 -- MLP parameter counting & forward pass
    backpropagation.py       # TODO: Milestone 4 -- Gradient computation via chain rule
    optimizers.py            # TODO: Milestone 5 -- Optimizer update rules & LR schedules
  tests/
    __init__.py
    test_perceptron.py       # Tests for Milestone 1
    test_activations.py      # Tests for Milestone 2
    test_architecture.py     # Tests for Milestone 3
    test_backpropagation.py  # Tests for Milestone 4
    test_optimizers.py       # Tests for Milestone 5
  solutions/
    __init__.py
    perceptron.py            # Reference solution
    activations.py           # Reference solution
    architecture.py          # Reference solution
    backpropagation.py       # Reference solution
    optimizers.py            # Reference solution
```

---

## Instructions

### Milestone 1: Perceptron Mathematics (20 min)

Open `src/perceptron.py`. Implement three functions:

- **`perceptron_output(inputs, weights, bias)`** -- Compute the raw output z = X @ W + b using NumPy matrix multiplication.
- **`perceptron_parameter_count(input_dim, output_dim)`** -- Return the total number of trainable parameters (weights + biases).
- **`decision_boundary_2d(w1, w2, bias)`** -- For a 2D perceptron, compute the slope and y-intercept of the decision boundary line w1·x1 + w2·x2 + b = 0.

Run tests: `pytest tests/test_perceptron.py -v`

### Milestone 2: Activation Functions (20 min)

Open `src/activations.py`. Implement seven functions:

- **`sigmoid(x)`** -- Compute σ(x) = 1 / (1 + e^(−x)).
- **`sigmoid_derivative(x)`** -- Compute σ'(x) = σ(x) · (1 − σ(x)).
- **`relu(x)`** -- Compute ReLU(x) = max(0, x).
- **`relu_derivative(x)`** -- Compute ReLU'(x) = 1 if x > 0 else 0.
- **`tanh_activation(x)`** -- Compute tanh(x).
- **`tanh_derivative(x)`** -- Compute tanh'(x) = 1 − tanh²(x).
- **`classify_activation(name)`** -- Return a dict of properties for the named activation: `output_range`, `zero_centered`, `saturating`, and `dead_neuron_risk`.

Run tests: `pytest tests/test_activations.py -v`

### Milestone 3: MLP Architecture Design (25 min)

Open `src/architecture.py`. Implement three functions:

- **`count_parameters(layer_dims)`** -- Given a list like `[64, 128, 64, 10]` (input → hidden → output), compute total trainable parameters.
- **`forward_pass(X, weights, biases, activation)`** -- Manually compute a forward pass through an MLP using NumPy. Apply the activation after each hidden layer but **not** after the output layer. Return a list of all layer outputs.
- **`compare_architectures(configs)`** -- Given a dict of architecture configs mapping names to `layer_dims` lists, return a list of `(name, param_count)` tuples sorted by param count ascending.

Run tests: `pytest tests/test_architecture.py -v`

### Milestone 4: Backpropagation & Chain Rule (30 min)

Open `src/backpropagation.py`. Implement four functions:

- **`mse_loss(predictions, targets)`** -- Compute mean squared error.
- **`mse_loss_gradient(predictions, targets)`** -- Compute dL/d(predictions) = (2/n)·(predictions − targets).
- **`linear_backward(x, w, upstream_grad)`** -- Given upstream gradient dL/dz for a linear layer z = x @ w + b, compute gradients dL/dw, dL/db, and dL/dx.
- **`two_layer_gradients(x, w1, b1, w2, b2, targets)`** -- Full forward + backward pass through a 2-layer ReLU network with MSE loss. Return all intermediate values and gradients.

Run tests: `pytest tests/test_backpropagation.py -v`

### Milestone 5: Optimizers & Learning Rate Scheduling (25 min)

Open `src/optimizers.py`. Implement five functions:

- **`sgd_step(param, grad, lr)`** -- Vanilla SGD: param ← param − lr · grad.
- **`sgd_momentum_step(param, grad, velocity, lr, momentum)`** -- SGD with momentum: v ← μ·v − lr·grad, param ← param + v.
- **`adam_step(param, grad, m, v, t, lr, beta1, beta2, epsilon)`** -- One step of Adam with bias correction.
- **`step_decay_lr(initial_lr, epoch, drop_factor, drop_every)`** -- Step decay: lr = initial_lr · drop_factor^⌊epoch/drop_every⌋.
- **`exponential_decay_lr(initial_lr, epoch, decay_rate)`** -- Exponential decay: lr = initial_lr · decay_rate^epoch.

Run tests: `pytest tests/test_optimizers.py -v`

---

## Running All Tests

```bash
# All tests (should FAIL on starter code, PASS on completed code)
pytest tests/ -v

# Individual test files
pytest tests/test_perceptron.py -v
pytest tests/test_activations.py -v
pytest tests/test_architecture.py -v
pytest tests/test_backpropagation.py -v
pytest tests/test_optimizers.py -v
```

---

## Submission Checklist

- [ ] All functions in `src/perceptron.py` implemented and passing tests.
- [ ] All functions in `src/activations.py` implemented and passing tests.
- [ ] All functions in `src/architecture.py` implemented and passing tests.
- [ ] All functions in `src/backpropagation.py` implemented and passing tests.
- [ ] All functions in `src/optimizers.py` implemented and passing tests.
- [ ] `pytest tests/ -v` shows all tests passing.
- [ ] Code is clean, follows PEP 8, and contains no unnecessary comments.
