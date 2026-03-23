# Loss Functions: MSE & CrossEntropy

**Estimated Time:** 10 Minutes

## Introduction

A **loss function** is the scoreboard your learning algorithm tries to improve. It answers: "How wrong are we, and in what direction should we change?" Two models can both make mistakes, but not all mistakes are equal -- being off by 0.01 on a house price feels different from confidently predicting the wrong medical label.

**Mean Squared Error (MSE)** and **cross-entropy** are two common scoreboards, but they are built for different kinds of problems. Using the wrong one is like using a ruler to measure weight: the number might change, but it will not mean what you think.

## Core Concepts

**What a loss function does**

During training, the model produces predictions. The loss compares predictions to targets and returns a single number (usually nonnegative). Optimization (conceptually) nudges parameters to **reduce** that number. The choice of loss shapes what "better" means.

**MSE for regression**

MSE cares about **continuous** errors. Squaring does two intuitive things:

- Bigger errors are punished much more than small ones (outliers dominate).
- The math stays smooth and works nicely with many classical algorithms.

**When:** predicting numbers -- prices, demand, temperature -- where "closer is better" in a squared sense.

**Cross-entropy for classification**

Classification outputs **probabilities over classes** (or logits that become probabilities). Cross-entropy measures how surprised you would be if the true label were drawn from your predicted distribution. If you assign low probability to the true class, the loss spikes.

**When:** predicting categories or labels, especially with probabilistic models (logistic regression, softmax outputs in neural nets).

**Mathematical difference (plain language)**

- MSE compares **numeric distance** between prediction and a real-valued target.
- Cross-entropy compares **probability assignments** to a discrete true class.

**Practical application contrast**

- Regression metrics often pair naturally with **MSE** (or related losses like MAE/Huber depending on outlier tolerance).
- Multi-class classification with softmax typically uses **categorical cross-entropy**; binary classification often uses **binary cross-entropy** / log loss.

## Connecting to Practice

You will see these names in libraries and model APIs. The key habit is: **match the loss to the label type and model output**.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss

reg = LinearRegression().fit(X_train, y_train_cont)
pred = reg.predict(X_test)
print(mean_squared_error(y_test_cont, pred))

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train_cls)
proba = clf.predict_proba(X_test)
print(log_loss(y_test_cls, proba))
```

If you optimize the wrong objective, you can get a model that looks fine on a misleading metric -- or one that never learns the decision boundary you care about.

---

## Further Learning & Resources

**Documentation**

- **[scikit-learn: Mean squared error](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)** - *Docs*: Official reference for regression loss metrics.
- **[scikit-learn: Log loss](https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss)** - *Docs*: Official reference for classification loss metrics.
- **[scikit-learn: Linear models](https://scikit-learn.org/stable/modules/linear_model.html)** - *Docs*: How loss functions connect to model training objectives.

**Interactive**

- **[scikit-learn metrics user guide](https://scikit-learn.org/stable/modules/model_evaluation.html)** - *Interactive*: Work through metric examples in context.
- **[Kaggle Learn: Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)** - *Interactive*: Hands-on exercises covering metrics and evaluation.
- **[Google Colab](https://colab.research.google.com/)** - *Interactive*: Compare MSE vs log loss on toy data in the browser.
