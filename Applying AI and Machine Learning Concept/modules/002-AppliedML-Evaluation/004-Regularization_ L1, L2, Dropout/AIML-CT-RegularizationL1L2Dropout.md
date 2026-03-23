# Regularization: L1, L2, Dropout

**Estimated Time:** 10 Minutes

## Introduction

Complex models can fit training data extremely well -- including the random parts. **Regularization** is a deliberate penalty on complexity so the model prefers simpler, more stable solutions unless the data strongly justifies extra wiggles.

It is like choosing a travel itinerary: L2 says "avoid extreme detours everywhere," L1 says "prefer fewer stops even if some are bold," and **dropout** (in neural networks) says "do not let one road become the only route you know."

## Core Concepts

**Why regularization exists**

Without a penalty, many models will chase every fluctuation in the training set. Regularization adds a cost for large or numerous parameters, which typically **reduces overfitting** and can improve validation performance -- especially when features are noisy or correlated.

**L2 (Ridge): shrink weights smoothly**

L2 adds a penalty proportional to the **square** of coefficients. It tends to shrink many weights together rather than zeroing them out. It is often a good default when you believe many features contribute a little.

**When:** regression/classification with many correlated features; you want stable coefficients and smoother decision boundaries.

**L1 (Lasso): encourage sparsity**

L1 adds a penalty proportional to the **absolute value** of coefficients. It can drive some coefficients exactly to zero, performing **feature selection** implicitly.

**When:** you suspect only a subset of features truly matter, or you want an interpretable sparse model.

**Elastic Net (briefly)**

Sometimes practitioners combine L1 and L2 to get both shrinkage and selection -- useful when features are correlated and pure L1 behaves erratically.

**Dropout (neural networks)**

Dropout randomly "turns off" a fraction of neurons during training so the network cannot rely on a few co-adapted paths. At inference, outputs are scaled to match expected behavior. It reduces overfitting by forcing redundant, robust representations. Applied depth lives in your neural network module.

## Connecting to Practice

In scikit-learn, Ridge/Lasso/ElasticNet are first-class tools for linear models.

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.01, max_iter=5000).fit(X_train, y_train)

print(ridge.coef_.shape)
print((lasso.coef_ != 0).sum())
```

**Choosing between L1 and L2 (practical intuition)**

- Need **interpretable feature selection** and can tolerate some instability with correlated features: lean **Lasso** (or Elastic Net).
- Need **stable coefficients** and smooth shrinkage: lean **Ridge**.
- Neural nets: use **Dropout** layers; pair with other regularizers and good validation discipline. The specific API differs across frameworks, but the idea is the same: during each training step, randomly zero out a fraction of neuron outputs (e.g., 30%), then scale the remaining outputs so the expected sum stays the same. At inference time, all neurons are active.

---

## Further Learning & Resources

**Documentation**

- **[scikit-learn: Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)** - *Docs*: API reference for L2-regularized linear model.
- **[scikit-learn: Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)** - *Docs*: API reference for L1-regularized linear model.
- **[Deep Learning Book, Chapter 7: Regularization](https://www.deeplearningbook.org/contents/regularization.html)** - *Docs*: Rigorous treatment of L1, L2, dropout, and other regularization strategies.

**Interactive**

- **[scikit-learn linear model examples](https://scikit-learn.org/stable/modules/linear_model.html)** - *Interactive*: Comprehensive examples of regularized models.
- **[Kaggle Learn: Regularization](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Exercises on ridge/lasso intuition.
- **[Google Colab](https://colab.research.google.com/)** - *Interactive*: Sweep alpha for Ridge/Lasso on a small dataset.
