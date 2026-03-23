# Optimizers: SGD vs. Adam

**Estimated Time:** 10 Minutes

---

## Introduction

In the previous reading, you learned that gradient descent adjusts weights by stepping in the direction opposite to the gradient. But *how* you take that step -- how big, how adaptive, how much you remember from previous steps -- turns out to matter enormously. The choice of **optimizer** can be the difference between a model that converges quickly and one that trains for hours without improving.

This reading compares the two most important optimizers you will encounter: **Stochastic Gradient Descent (SGD)** and **Adam**. You will learn when to reach for each one, and how **learning rate schedulers** can further improve convergence by adjusting the step size during training.

---

## Core Concepts

### Stochastic Gradient Descent (SGD)

Vanilla gradient descent computes the gradient of the loss over the *entire* dataset before taking a single step. This is accurate but extremely slow for large datasets. **Stochastic Gradient Descent** solves this by computing the gradient on a small random **mini-batch** of data at each step.

The update rule is:

\[
w = w - \eta \cdot \nabla L
\]

The "stochastic" part refers to the noise introduced by using mini-batches instead of the full dataset. This noise is actually helpful: it can push the optimizer out of sharp local minima and toward flatter regions of the loss surface that tend to generalize better.

### SGD with Momentum

Plain SGD can be slow because it treats every step independently. **Momentum** adds a "memory" of previous gradients, so the optimizer builds up speed in directions where the gradient consistently points the same way (like a ball rolling downhill) and dampens oscillations in directions where the gradient keeps flipping.

The update becomes:

\[
v_t = \mu \cdot v_{t-1} + \nabla L
\]
\[
w = w - \eta \cdot v_t
\]

where \(\mu\) is the momentum coefficient (typically 0.9). SGD with momentum is the optimizer behind many state-of-the-art results in computer vision. It is simple, well-understood, and tends to find solutions that generalize well -- but it requires more careful tuning of the learning rate.

### Adam: Adaptive Moment Estimation

**Adam** combines two ideas:

1. **Momentum** (first moment): It keeps a running average of past gradients, just like SGD with momentum.
2. **Adaptive learning rates** (second moment): It also keeps a running average of past *squared* gradients. This allows each weight to effectively have its own learning rate -- weights with consistently large gradients get smaller effective learning rates, and weights with small or rare gradients get larger ones.

Formally, at each step Adam maintains:

\[
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla L \quad \text{(first moment estimate)}
\]
\[
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla L)^2 \quad \text{(second moment estimate)}
\]

After bias correction, the weight update is:

\[
w = w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

This adaptivity makes Adam much less sensitive to the initial learning rate choice and particularly effective for problems with sparse gradients or noisy data. The default learning rate for Adam is typically lower (0.001) than for SGD (0.01-0.1), because the adaptive scaling already amplifies small gradients.

### SGD vs. Adam: When to Use Which

| Factor | SGD (with momentum) | Adam |
|---|---|---|
| Ease of tuning | Requires careful learning rate selection | More forgiving; works well out of the box |
| Convergence speed | Slower to converge initially | Faster early convergence |
| Generalization | Often finds solutions that generalize better | Can sometimes converge to sharper minima that generalize worse |
| Memory usage | Lower (only stores velocity) | Higher (stores both first and second moment estimates per parameter) |
| Best for | Computer vision, well-tuned production models | NLP, transformers, rapid prototyping, sparse data |
| Research consensus | Preferred when you can afford the tuning time | Preferred when you need quick, reliable results |

**A practical rule of thumb:** Start with Adam for fast iteration during development and experimentation. If you are pushing for maximum performance on a well-understood problem (especially in computer vision), switch to SGD with momentum and invest time tuning the learning rate.

### Learning Rate Schedulers

No matter which optimizer you use, keeping the learning rate constant for the entire training run is rarely optimal. Early in training, you want larger steps to make rapid progress. Later, you want smaller steps to fine-tune the weights without overshooting the minimum.

A **learning rate scheduler** adjusts the learning rate according to a predefined policy during training. Common strategies include:

**Step decay** reduces the learning rate by a fixed factor at regular intervals. For example, cutting the rate in half every 10 epochs. It is simple and predictable.

**Cosine annealing** smoothly decreases the learning rate following a cosine curve from its initial value down to near zero over the training run. This often produces better results than abrupt step decreases because the smooth transition avoids sudden disruptions to the optimization trajectory.

**Reduce on plateau** monitors a metric (like validation loss) and reduces the learning rate only when the metric stops improving. This is adaptive and requires less guesswork about when to decay -- the data tells you.

### Choosing a Scheduler

| Scheduler | When to Use |
|---|---|
| Step decay | You know roughly when you want to decay (simple and predictable) |
| Cosine annealing | Standard choice for fixed-length training runs; smooth decay often outperforms step decay |
| Reduce on plateau | You want the schedule to be data-driven rather than predetermined |

For most projects, **cosine annealing** or **reduce on plateau** are strong defaults. Step decay is useful when you are reproducing a specific paper's training recipe.

---

## Connecting to Practice

In the hands-on exercises, you will train the same model with different optimizer and scheduler combinations and compare convergence curves. You will see firsthand how Adam tends to converge faster in early epochs, how SGD with momentum can catch up and sometimes surpass Adam by the end of training, and how a well-chosen scheduler can make a meaningful difference regardless of which optimizer you use.

The key takeaway for practice: do not treat the optimizer as an afterthought. It is a first-class hyperparameter. When a model is not training well, changing the optimizer or adding a scheduler is often more effective than changing the model architecture.

---

## Further Learning & Resources

### Documentation

1. [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) -- The original Adam paper. Sections 1-2 are accessible and explain the motivation clearly.
2. [Deep Learning Book, Chapter 8: Optimization](https://www.deeplearningbook.org/contents/optimization.html) -- Comprehensive coverage of SGD, momentum, adaptive methods, and learning rate strategies.
3. [An overview of gradient descent optimization algorithms (Sebastian Ruder)](https://ruder.io/optimizing-gradient-descent/) -- Well-known survey comparing SGD variants, Adam, and related optimizers with clear visuals.

### Interactive Resources

1. [Google Machine Learning Crash Course: Optimizers](https://developers.google.com/machine-learning/crash-course/reducing-loss/optimizing-learning-rate) -- Interactive exercises demonstrating how learning rate and optimizer choice affect training dynamics.
2. [Distill.pub: Why Momentum Really Works](https://distill.pub/2017/momentum/) -- A beautifully visualized, interactive exploration of momentum and adaptive learning rate methods.
3. [TensorFlow Playground](https://playground.tensorflow.org/) -- Experiment with learning rate settings and observe their effect on convergence in real time.
