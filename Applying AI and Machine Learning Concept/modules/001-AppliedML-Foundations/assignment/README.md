# AppliedML-Foundations Assignment

## Scenario

You are a junior data scientist at **TechPulse Analytics**, a consulting firm that helps businesses make data-driven decisions. You have been assigned two projects:

**Project A -- Housing Price Prediction (Regression)**
A real estate company wants to estimate the market value of residential properties based on their physical characteristics and location quality. Your task is to build a Linear Regression model that predicts house prices from features such as square footage, number of bedrooms, lot size, and neighborhood quality score.

**Project B -- Customer Churn Classification (Classification)**
A telecom provider wants to identify customers who are likely to cancel their subscription. Your task is to build a Logistic Regression model that classifies whether a customer will churn based on their tenure, contract type, monthly charges, and support history.

Both projects must follow reproducibility best practices and demonstrate a principled approach to algorithm selection.

---

## Learning Objectives

By completing this assignment you will demonstrate the ability to:

1. Apply code reproducibility strategies using random seeds and deterministic pipelines
2. Build a Linear Regression model to predict a continuous target variable
3. Build a Logistic Regression model for binary classification
4. Prevent data leakage by splitting data before fitting transformers
5. Evaluate models using multiple metrics (never relying on a single metric)
6. Apply an algorithm selection framework to justify model choices for tabular data

---

## Prerequisites

- Python 3.10 or higher
- Familiarity with numpy, pandas, and scikit-learn
- Completion of the AppliedML-Foundations readings and lecture

---

## Setup

1. Clone this repository.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Verify your setup by running the test suite:
   ```
   pytest -v
   ```
   All tests should FAIL at this point. Your goal is to make them pass.

---

## Project Structure

```
/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # Data generation (provided, do not modify)
│   ├── features.py           # TODO: Feature engineering and preprocessing
│   ├── train.py              # TODO: Model training and algorithm selection
│   └── evaluate.py           # TODO: Model evaluation and comparison
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared test fixtures
│   ├── test_reproducibility.py
│   ├── test_features.py
│   ├── test_train.py
│   └── test_evaluate.py
├── solutions/                # Instructor reference (do not distribute)
├── requirements.txt
└── README.md
```

---

## Milestones

Work through these milestones in order. Run `pytest -v` after each milestone to track your progress.

### Milestone 1 -- Reproducibility Setup (~15 minutes)

File: `src/features.py`

Implement `set_random_seed(seed)` to set deterministic seeds for both `numpy` and Python's built-in `random` module. All downstream operations must use `random_state=42`.

**Tests:** `test_reproducibility.py`

### Milestone 2 -- Feature Preparation (~15 minutes)

File: `src/features.py`

Implement `prepare_housing_features(df)` and `prepare_churn_features(df)` to separate feature matrices from target vectors.

**Tests:** `test_features.py::test_prepare_housing_features`, `test_features.py::test_prepare_churn_features`

### Milestone 3 -- Data Splitting and Scaling (~20 minutes)

File: `src/features.py`

Implement `split_data(X, y, ...)` and `scale_features(X_train, X_test)`.

CRITICAL: You must fit the scaler on the training set ONLY, then transform both sets. Fitting on the combined data constitutes data leakage and the tests will catch it.

**Tests:** `test_features.py::test_split_data_*`, `test_features.py::test_scale_*`

### Milestone 4 -- Linear Regression for Housing Prices (~25 minutes)

File: `src/train.py`

Implement `train_dummy_regressor(...)` as a baseline, then `train_linear_regression(...)`. The Linear Regression model must outperform the dummy baseline on R-squared.

**Tests:** `test_train.py::test_train_dummy_regressor`, `test_train.py::test_train_linear_regression*`

### Milestone 5 -- Logistic Regression for Customer Churn (~25 minutes)

File: `src/train.py`

Implement `train_dummy_classifier(...)` as a baseline, then `train_logistic_regression(...)`. The Logistic Regression model must outperform the dummy baseline on F1 score.

**Tests:** `test_train.py::test_train_dummy_classifier`, `test_train.py::test_train_logistic_regression*`

### Milestone 6 -- Model Evaluation (~20 minutes)

File: `src/evaluate.py`

Implement `evaluate_regression(...)`, `evaluate_classification(...)`, and `compare_models(...)`. You must compute multiple metrics for every model. Using accuracy alone for classification is not acceptable.

**Tests:** `test_evaluate.py`

### Milestone 7 -- Algorithm Selection Framework (~15 minutes)

File: `src/train.py`

Implement `select_algorithm(task_type, data_characteristics)` to recommend an appropriate algorithm based on the task type (regression vs. classification), data format, number of features, and target type. Return a structured recommendation with rationale.

**Tests:** `test_train.py::test_select_algorithm_*`

---

## Grading

| Milestone | Points | Criteria |
|-----------|--------|----------|
| Milestone 1 -- Reproducibility | 10 | Seeds set correctly; deterministic outputs |
| Milestone 2 -- Feature Prep | 10 | Correct feature/target separation |
| Milestone 3 -- Split & Scale | 20 | Correct split; no data leakage |
| Milestone 4 -- Linear Regression | 15 | Baseline + model; beats dummy on R2 |
| Milestone 5 -- Logistic Regression | 15 | Baseline + model; beats dummy on F1 |
| Milestone 6 -- Evaluation | 20 | Multiple metrics; correct comparison logic |
| Milestone 7 -- Algorithm Selection | 10 | Structured recommendation with rationale |
| **Total** | **100** | |

All grading is automated through the pytest suite. A passing test earns full credit for that component.

---

## Submission

Push your completed code to the repository. The autograder will run `pytest -v` and report your score based on the number of passing tests.

---

## Key Constraints

- ALL random states must be set to 42
- NEVER fit a scaler, encoder, or any transformer on test data
- NEVER use accuracy as your sole classification metric
- ALWAYS build a dummy/baseline model before a real model
- Do NOT modify `src/data_utils.py` or any file in `tests/`
