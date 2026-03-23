# Algorithm Selection Framework

**Estimated Time:** 10 Minutes

## Introduction

Choosing a machine learning algorithm can feel like picking a vehicle for a trip. A compact car, a delivery van, and a long-haul truck all move you forward, but they excel under different loads, distances, and road conditions. In the same way, algorithms differ in what they assume about your data, how much data they need, and how interpretable or flexible they are.

This lesson gives you a **systematic way to narrow the field**: start from the problem you are solving, describe your data honestly, respect real-world constraints, and only then compare a small set of sensible candidates. The goal is not to memorize every algorithm name on day one, but to build a repeatable decision path so you are not guessing when you open a notebook.

## Core Concepts

### 1. Anchor on the problem type

The first fork in the road is **what you are trying to predict or discover**:

- **Supervised learning** means you have labeled examples (inputs paired with known outcomes). Typical tasks include **classification** (predicting a category) and **regression** (predicting a number).
- **Unsupervised learning** means you are looking for structure without predefined labels, such as **clustering** or **dimensionality reduction**.

If you already studied supervised versus unsupervised learning, treat this step as your compass: it tells you which family of tools is even on the table.

### 2. Describe your data in plain terms

For **tabular data** (rows and columns, like a spreadsheet), a few characteristics drive sensible choices:

- **Size:** Do you have hundreds of rows or millions? Some methods need more examples to stabilize; others can work with less but may underfit or overfit if you are not careful.
- **Feature types:** Mostly numeric, mostly categorical, or a mix? Some algorithms handle mixed types smoothly after preprocessing; others expect numeric inputs and rely on encoding choices.
- **Sparsity and noise:** Many empty or rare categories, measurement error, or inconsistent labeling can make flexible models look good on training data and worse in production.
- **Linear versus nonlinear patterns:** If relationships are roughly linear after scaling, simpler models can be strong. If interactions and curves dominate, you may need more expressive models or feature engineering.

You are not judging whether your data is "good" or "bad" here. You are listing facts that steer you toward algorithms that match those facts.

### 3. Apply constraints early

Real projects rarely optimize only for leaderboard accuracy. Before you fall in love with a complex model, list constraints such as:

- **Interpretability:** Do stakeholders need to see coefficients, rules, or clear feature importance?
- **Latency and cost:** Must predictions be fast on a CPU? Is retraining expensive?
- **Regulatory or audit needs:** Some settings favor simpler, more explainable baselines even when complexity could squeeze out extra points of accuracy.
- **Maintenance:** Will the team reliably monitor and refresh a complex pipeline?

Constraints do not block machine learning; they **trim the search space** so you invest effort where it matters.

### 4. The "baselines first" rule

Before you reach for sophisticated algorithms, establish **simple, trustworthy baselines** on the same data split and evaluation protocol you will use for everything else. For tabular problems, common baselines include:

- A **dummy** model that follows a trivial rule (for example, always predict the most frequent class). It answers: "How hard is this problem if I barely try?"
- A **linear** model such as logistic regression for classification or linear regression for regression. It answers: "What can a straightforward linear decision surface capture?"

Conceptually, this mirrors the idea behind resources like the **scikit-learn algorithm selection cheat sheet**: start from the simplest reasonable choices for your task, then move outward when evidence says you should. Baselines turn model selection from opinion into measurement.

### 5. Shortlist candidates, then compare fairly

Once problem type, data facts, and constraints are clear, pick **a small set** of algorithms that plausibly fit. For example, on tabular supervised learning you might compare linear models, tree-based ensembles, and perhaps a distance-based or kernel method depending on scale and noise. The lesson's focus is **why each might belong on the list**, not the full implementation details.

When you compare candidates:

- Use the **same validation strategy** (for example, cross-validation or a held-out set).
- Track **multiple metrics** aligned with the business question, not only one number that looks impressive in isolation.
- Watch for **instability**: large swings across folds or splits often signal data issues or a poor match between algorithm and data size.

### 6. Decide when to escalate complexity

Move toward more complex algorithms when baselines plateau and your diagnostics suggest **underfitting** (the model cannot capture the signal) or when constraints allow and the uplift justifies the operational cost. **Resist** adding complexity when the main issue is **data quality, leakage, or evaluation mistakes**, because a fancier algorithm usually amplifies those problems rather than fixing them.

## Connecting to Practice

The following snippets are **illustrative only**; full pipelines belong in labs and lectures. They show the *shape* of baseline-first thinking in Python with common libraries.

**Problem framing (supervised classification on tabular data):**

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
```

`DummyClassifier` gives you a trivial baseline. `LogisticRegression` is a strong, fast linear baseline for many tabular classification tasks, especially after scaling numeric features and thoughtful encoding of categories.

**Data handling (typical stack):**

```python
import pandas as pd
import numpy as np
```

Tabular work usually starts in `pandas` for inspection and splitting, then converts to `numpy` arrays or uses sklearn-compatible structures for modeling.

**Evaluation mindset (conceptual):**

```python
from sklearn.model_selection import cross_val_score
```

You are not declaring a winner from one lucky split. Cross-validation helps you see whether an algorithm's performance is stable across different slices of the data.

To **apply an algorithm selection framework to choose base models for tabular data**, state explicitly:

1. Task type (classification or regression, supervised path).
2. Data sketch (size, feature types, noise, approximate linearity).
3. Constraints (interpretability, speed, cost).
4. Baselines (dummy plus a linear model at minimum).
5. A shortlist justified by the above, compared with shared validation and metrics.

If you can write those five bullets for a dataset, you are using the framework, not just browsing algorithm names.

---

## Further Learning & Resources

**Reading and documentation**

- **[scikit-learn: Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html)** - *Docs*: The official "cheat sheet" style map for narrowing estimators by task and data type.
- **[scikit-learn User Guide: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)** - *Docs*: Overview of estimators and when they are commonly used.
- **[pandas User Guide: Intro to data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html)** - *Docs*: Grounding for describing tabular data before modeling.

**Interactive practice**

- **[scikit-learn: Tutorials with notebooks](https://scikit-learn.org/stable/tutorial/index.html)** - *Interactive*: Hands-on examples aligned with sklearn workflows.
- **[Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** - *Interactive*: Short exercises focused on core ideas and sklearn-style workflows.
- **[Google Colab](https://colab.research.google.com/)** - *Interactive*: Run small sklearn experiments in a browser notebook without local setup.
