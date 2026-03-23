# Train, Validation & Test Splits

**Estimated Time:** 10 Minutes

## Introduction

You already know how to hold out a test set so you can report "how well the model does on new data." That is still the right instinct. The missing piece is *what happens in the middle* between "fit the model" and "final grade." If you tune hyperparameters or pick a model using the same data you later use to score it, you have accidentally **peeked at the exam** while studying. Train, validation, and test splits exist so you can learn, adjust, and finally judge -- without cheating.

Think of building a recipe: the **training** set is your kitchen experiments; the **validation** set is a friendly taste-test before the dinner party; the **test** set is the actual dinner -- you only get one honest shot to see if guests like it.

## Core Concepts

**Why three ways, not two?**

- **Training:** Data the model's parameters actually learn from.
- **Validation:** Data used to compare choices -- model family, hyperparameters, early stopping -- without touching the test set.
- **Test:** Data locked away until the end, used once (or rarely) for an unbiased estimate of generalization.

If you pick "the best model" on the test set, you have turned the test set into a validation set. Your reported score stops being a fair estimate of future performance.

**Stratification**

When classes are imbalanced (e.g., 95% "no fraud" and 5% "fraud"), a random split might put almost all rare cases in one fold by bad luck. **Stratified splitting** keeps each split's class proportions close to the full dataset, so every subset is representative. That matters whenever the minority class is what you care about most.

**K-fold cross-validation (concept)**

Instead of one validation slice, **k-fold** rotates: split data into *k* parts, train on *k*-1, validate on the remaining 1, repeat *k* times, average the scores. It is a more stable estimate of how sensitive your pipeline is to which rows were in the training bucket -- especially when data is scarce. **Stratified k-fold** does the same but preserves class ratios in each fold.

**When to use what**

- Simple baseline: train / validation / test (or train+val with a final test).
- Small data: prefer k-fold (often stratified) to reduce variance in your metric.
- Final reporting: still keep a **held-out test** that was not used for any tuning if you need an unbiased number.

## Connecting to Practice

In Python with scikit-learn, you typically split first, then only use `fit` on training (and optionally validation for model selection). Stratification is common for classification.

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    ...
```

The **why** here: stratification reduces unlucky splits; k-fold averages out noise so your "best model" decision is less fragile.

---

## Further Learning & Resources

**Documentation**

- **[scikit-learn User Guide: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)** - *Docs*: Comprehensive guide to validation strategies and iterators.
- **[scikit-learn: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)** - *Docs*: API reference for basic splitting with stratification support.
- **[scikit-learn: StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)** - *Docs*: API reference for stratified cross-validation.

**Interactive**

- **[Kaggle Learn: Model Validation](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Short exercises on validation strategies.
- **[scikit-learn: Cross-validation iterators (examples)](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators)** - *Interactive*: Runnable examples of different CV strategies.
- **[Google Colab](https://colab.research.google.com/)** - *Interactive*: Try sklearn splits on a small dataset in the browser.
