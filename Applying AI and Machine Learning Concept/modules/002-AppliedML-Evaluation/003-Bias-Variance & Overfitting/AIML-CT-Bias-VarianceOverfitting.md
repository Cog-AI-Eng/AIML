# Bias-Variance & Overfitting

**Estimated Time:** 10 Minutes

## Introduction

Every model sits on a spectrum. On one side, it might be too simple to capture real patterns -- like using only "average rainfall" to predict crop yield and ignoring soil type. On the other side, it might memorize quirks of the sample -- like believing a stock goes up every Tuesday because it happened twice in your spreadsheet.

**Bias** is the error from overly simple assumptions. **Variance** is the error from being too sensitive to the particular training sample. The tension between them is the **bias-variance tradeoff**: more flexibility can reduce bias but increase variance, and vice versa.

**Overfitting** is the classic failure mode where the model fits training noise. **Underfitting** is when it misses signal. **Learning curves** help you see which story you are in without guessing.

## Core Concepts

**Bias vs variance (real-world analogy)**

Imagine studying for an exam:

- **High bias:** you memorized one generic template answer. You are consistently off-topic (systematic error).
- **High variance:** you memorized exact wording of three practice problems. You ace those, but any new question wrecks you (unstable generalization).

**Underfitting vs overfitting**

- **Underfitting:** poor on training *and* validation/test -- too simple or wrong features.
- **Overfitting:** great on training, worse on validation/test -- too complex relative to data, or not enough regularization/data.

**Learning curves as diagnostics**

A learning curve plots performance vs **training set size** (or vs training epochs in deep learning). You compare **training** and **validation** curves:

- **Gap widening with more data used for fitting:** often a sign the model is memorizing (variance/overfitting risk).
- **Both curves low and close together:** often underfitting -- more model capacity or better features may help.
- **Validation improves as data grows and gap shrinks:** a healthy sign that more data is helping generalization.

**When to look at them**

- You are choosing complexity (depth, polynomial degree, tree depth).
- You suspect label noise or small data.
- You want evidence that collecting more data is worth it.

## Connecting to Practice

You can sketch learning curves with scikit-learn and plot with matplotlib/seaborn.

```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    Ridge(), X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5
)
```

Curves turn vague feelings ("it memorized") into a pattern you can reason about -- whether the problem is "not enough signal," "too much capacity," or "needs more examples."

---

## Further Learning & Resources

**Documentation**

- **[scikit-learn: Learning curve](https://scikit-learn.org/stable/modules/learning_curve.html)** - *Docs*: Guide to plotting and interpreting learning and validation curves.
- **[scikit-learn: learning_curve API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)** - *Docs*: API reference with parameter explanations.
- **[scikit-learn: Validation curves](https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve)** - *Docs*: Complement to learning curves for hyperparameter analysis.

**Interactive**

- **[scikit-learn learning curve example gallery](https://scikit-learn.org/stable/auto_examples/index.html#model-selection)** - *Interactive*: Runnable examples for model selection diagnostics.
- **[Kaggle Learn: Underfitting and Overfitting](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Short exercises on recognizing model fit issues.
- **[Google Colab](https://colab.research.google.com/)** - *Interactive*: Plot learning curves on sklearn datasets in the browser.
