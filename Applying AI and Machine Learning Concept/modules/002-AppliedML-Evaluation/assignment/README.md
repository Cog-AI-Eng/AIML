# AppliedML-Evaluation Assignment

**Activity ID:** AIML-AM-AppliedML-Evaluation
**Display Name:** AppliedML-Evaluation Assignment
**Duration:** 120 minutes

---

## Overview

You are a data scientist at a hospital analytics company. Your task is to build and
evaluate a **patient readmission prediction** pipeline. The dataset exhibits significant
**class imbalance** (far fewer readmissions than non-readmissions), which makes naive
accuracy a poor metric and demands careful evaluation methodology.

You will implement data splitting strategies, train regularized models, compute
classification metrics, and generate explainability outputs -- all from scratch using
scikit-learn primitives and numpy.

---

## Learning Objectives

By the end of this assignment you will be able to:

- Split data reproducibly using train/validation/test splits with stratification
- Implement stratified k-fold cross-validation
- Train Ridge (L2) and Lasso (L1) regularized logistic regression models
- Calculate precision, recall, F1-score, AUC-ROC, and confusion matrices
- Build an early stopping mechanism for iterative models
- Generate SHAP values for model explainability

---

## Prerequisites

Associates should have completed the readings, videos, and live lecture covering:

1. Train, Validation, and Test Splits (stratified k-fold)
2. Loss Functions: MSE and CrossEntropy
3. Bias-Variance Tradeoff and Overfitting (learning curves)
4. Regularization: L1, L2, Dropout
5. Metrics: Precision, Recall, F1
6. AUC-ROC and Confusion Matrix
7. Early Stopping Logic
8. Explainability and SHAP Values

---

## Project Structure

```
assignment/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- raw/           # Raw synthetic data (generated)
|   |-- processed/     # Processed splits
|-- notebooks/
|   |-- exploration.ipynb
|-- src/
|   |-- __init__.py
|   |-- data_utils.py  # Synthetic data generation (provided)
|   |-- splits.py      # TODO: Data splitting functions
|   |-- models.py      # TODO: Model training functions
|   |-- evaluate.py    # TODO: Evaluation metrics
|   |-- explain.py     # TODO: SHAP explainability
|-- tests/
|   |-- __init__.py
|   |-- test_splits.py
|   |-- test_models.py
|   |-- test_evaluate.py
|   |-- test_explain.py
|-- solutions/
|   |-- splits.py
|   |-- models.py
|   |-- evaluate.py
|   |-- explain.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset:

```python
from src.data_utils import generate_readmission_data
df = generate_readmission_data(n_samples=2000, imbalance_ratio=0.15, random_state=42)
df.to_csv("data/raw/readmission_data.csv", index=False)
```

---

## Milestones

### Milestone 1 -- Reproducibility and Data Splitting (Required)

**File:** `src/splits.py`

- Implement `train_val_test_split()` with stratification support
- Ensure reproducibility via `random_state`
- Validate split proportions sum to 1.0

### Milestone 2 -- Cross-Validation (Preferred)

**File:** `src/splits.py`

- Implement `stratified_kfold_split()` using `StratifiedKFold`
- Return fold indices for downstream use

### Milestone 3 -- Regularized Models (Required)

**File:** `src/models.py`

- Implement `train_ridge_model()` (L2 regularization via LogisticRegression)
- Implement `train_lasso_model()` (L1 regularization via LogisticRegression)
- Implement `train_model_with_early_stopping()` (Preferred)

### Milestone 4 -- Evaluation Metrics (Required)

**File:** `src/evaluate.py`

- Implement `compute_precision_recall_f1()` returning a dict
- Implement `compute_auc_roc()` returning the AUC score
- Implement `compute_confusion_matrix()` returning the matrix
- Implement `plot_roc_curve()` saving a figure to disk

### Milestone 5 -- Model Comparison Pipeline (Required)

**File:** `src/evaluate.py`

- Implement `compare_models()` that trains multiple models and returns
  a comparison DataFrame with metrics for each

### Milestone 6 -- SHAP Explainability (Preferred)

**File:** `src/explain.py`

- Implement `compute_shap_values()` using the SHAP library
- Implement `plot_shap_summary()` saving a SHAP summary plot

---

## Running Tests

```bash
pytest tests/ -v
```

All tests will **FAIL** on the starter code. Your goal is to make them **PASS** by
completing the TODO sections in each source file.

Reference solutions are in `solutions/` -- use them only after attempting the work
yourself.

---

## Exit Criteria

### Required (must pass to complete the assignment)

- [ ] `train_val_test_split()` works with stratification and reproducibility
- [ ] `train_ridge_model()` and `train_lasso_model()` return fitted models
- [ ] `compute_precision_recall_f1()` returns correct precision, recall, and F1
- [ ] `compute_auc_roc()` returns a valid AUC score
- [ ] `compute_confusion_matrix()` returns a valid confusion matrix
- [ ] `plot_roc_curve()` generates and saves a ROC curve figure
- [ ] `compare_models()` returns a DataFrame comparing model metrics
- [ ] All required tests pass

### Preferred (stretch goals)

- [ ] `stratified_kfold_split()` returns correct fold indices
- [ ] `train_model_with_early_stopping()` implements early stopping logic
- [ ] `compute_shap_values()` returns valid SHAP values
- [ ] `plot_shap_summary()` generates a SHAP summary plot
- [ ] All preferred tests pass

---

## Grading

| Category | Weight |
|---|---|
| Required tests passing | 70% |
| Preferred tests passing | 20% |
| Code quality and documentation | 10% |

---

## Tips

- Use `random_state` everywhere for reproducibility.
- The dataset is imbalanced (~15% positive class). Stratification matters.
- When computing AUC-ROC, you need predicted probabilities, not class labels.
- SHAP requires a fitted model and a background dataset.
- Check the test files to understand exactly what each function should return.
