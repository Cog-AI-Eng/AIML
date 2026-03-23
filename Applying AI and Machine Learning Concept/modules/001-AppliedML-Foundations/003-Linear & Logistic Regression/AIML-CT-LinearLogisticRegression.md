# Linear & Logistic Regression

**Estimated Time:** 10 Minutes

## Introduction

When people talk about "teaching a machine to learn from examples," supervised learning is often what they mean: you show the model inputs and the correct answers, and it learns a rule that maps new inputs to outputs. Two of the most widely used supervised models are **linear regression** and **logistic regression**. Despite the word "regression" in both names, they solve different problems: one predicts **numbers** (like price or temperature), and the other predicts **categories** (like "yes/no" or "spam/not spam").

Think of linear regression as drawing the **best straight trend line** through scattered points on a graph. Think of logistic regression as drawing a **decision boundary** that separates two groups, then turning a score into a **probability** between 0 and 1. Both are simple to explain, fast to train, and easy to interpret -- which is why they remain staples in real projects, even beside more complex algorithms.

In this reading, you will focus on **why** each model exists, **when** to choose it, and **what** each API in scikit-learn is asking for. You will build full models end-to-end in your upcoming lecture and assignment; here, the goal is a clear mental model so that code feels like filling in a well-labeled form, not memorizing magic.

## Core Concepts

### What problem does linear regression solve?

**Linear regression** predicts a **continuous** target: anything you can reasonably measure on a scale (dollars, hours, temperature, scores). The model assumes the target can be approximated as a **weighted sum** of the input features, plus a constant (the intercept). In plain language: "If I increase this feature a little, the prediction moves in a predictable direction, proportional to a learned weight."

**Why it matters:** Many business questions are naturally numeric: forecast demand, estimate house prices, or predict time to complete a task. Linear regression gives a **baseline** that is easy to explain to stakeholders ("each extra year of experience adds this much to the predicted salary, holding other factors constant").

**When it is a good fit:**

- The relationship between features and target is **roughly linear**, or you are willing to add transformations (e.g., squared terms) later.
- You want **interpretability**: coefficients describe direction and relative impact under the model's assumptions.
- You need something **fast and stable** on small or medium tabular data.

**When to be cautious:** Strong non-linear patterns, heavy interaction effects, or messy data may need richer models or feature engineering -- but linear regression is still often the first model you try for continuous targets.

**Scikit-learn hook (conceptual):** You fit on `X` (features) and `y` (continuous labels). The idea is "find weights that minimize prediction error," typically **mean squared error** in ordinary least squares settings.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

You will use libraries like **NumPy** and **pandas** to prepare `X` and `y`, and **matplotlib** or **seaborn** to visualize relationships and residuals -- those steps belong in your hands-on work.

### What problem does logistic regression solve?

**Logistic regression** is used for **classification**, especially **binary** classification (two classes: positive/negative, churn/not churn, pass/fail). It still uses a **linear combination** of features, but instead of outputting an unbounded number, it passes that score through a **logistic (sigmoid) function** so the output behaves like a **probability** between 0 and 1.

**Why the sigmoid:** Raw scores can be any real number; probabilities cannot. The sigmoid squashes the score into (0, 1), which matches how we talk about uncertainty: "There is a 78% chance this email is spam."

**When it is a good fit:**

- You need a **probabilistic** view of class membership, not just a label.
- The **decision boundary** is roughly linear in the feature space (or can be made so with feature engineering).
- You want a **fast, interpretable** classifier for tabular data, often as a strong baseline before trying more complex models.

**When to be cautious:** If classes overlap in complex ways or interactions dominate, a linear boundary may underfit -- but logistic regression remains a standard first step for binary problems.

**Scikit-learn hook (conceptual):** You still pass `X`, but `y` contains **class labels** (often 0 and 1 for binary tasks). The model learns weights that separate the classes and outputs probabilities for the positive class.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
p_pos = clf.predict_proba(X_test)[:, 1]
y_hat = clf.predict(X_test)
```

For **binary** tasks, `predict` returns class labels; `predict_proba` returns probabilities per class -- useful for ranking, thresholds, and evaluation beyond simple accuracy.

### How the two models relate (and how they differ)

Both models start from the same idea: **combine features with learned weights**. The critical difference is the **type of output**:

| Idea | Linear regression | Logistic regression |
|------|-------------------|---------------------|
| Typical target | Continuous numeric value | Class label (often binary) |
| Output nature | Unbounded prediction | Probability-like score in (0, 1) for positive class |
| Common use | Regression baselines, interpretation | Classification baselines, calibrated probabilities |

**Analogy:** Linear regression answers "**How much?**" Logistic regression answers "**Which side of the line am I on, and how confident are we?**"

### Evaluation: what "good" means (at a glance)

You will deepen metrics in later activities; for now, know the intent:

- **Linear regression:** Errors are usually measured with **mean squared error** or **mean absolute error** -- how far predictions are from true numeric values.
- **Logistic regression:** Common measures include **accuracy**, **precision/recall**, **ROC-AUC**, and **log-loss**, depending on whether you care about pure labels or the quality of **probabilities**.

The right metric depends on the business cost of different mistakes -- another reason logistic regression's **probabilities** are valuable.

## Connecting to Practice

In professional workflows, linear and logistic regression often appear as **first models** because they are quick to train, easy to debug, and straightforward to explain. They also help you sanity-check data issues: if a simple model behaves strangely, the problem may be features, leakage, or labeling -- not the lack of a fancier algorithm.

You should now recognize that the target `y` is numeric for `LinearRegression` and that `predict` returns continuous values. For `LogisticRegression`, `y` contains discrete class labels, and the model outputs class predictions and/or probabilities. Binary setup is the common pattern for churn, fraud flags, pass/fail, and similar problems.

Your upcoming **video**, **Algorithm Selection Framework**, **live lecture**, and **assignment** are where you will turn these ideas into full pipelines -- loading data with **pandas**, shaping `X` and `y`, splitting data, fitting models, and visualizing results with **matplotlib** and **seaborn**. This reading is the conceptual map; the hands-on work is where the map becomes a route.

---

## Further Learning & Resources

**Reading and documentation**

- **[scikit-learn User Guide: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)** - *Docs*: Official overview of linear and logistic regression in the library, including regularization and key parameters.
- **[scikit-learn: LinearRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)** - *Docs*: API reference for the continuous regression estimator.
- **[scikit-learn: LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)** - *Docs*: API reference for classification, including `predict_proba` and solver options.

**Interactive practice**

- **[Google ML Crash Course: Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression)** - *Interactive*: Short, text-focused explanation of logistic regression within Google's ML course structure.
- **[scikit-learn: Classification examples gallery](https://scikit-learn.org/stable/auto_examples/classification/index.html)** - *Interactive*: Runnable examples you can explore and adapt, focusing on linear classifiers.
- **[Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Interactive notebooks covering modeling basics with evaluation exercises.
