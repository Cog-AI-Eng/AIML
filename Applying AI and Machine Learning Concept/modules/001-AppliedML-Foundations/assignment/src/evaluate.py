"""
Model evaluation with multiple metrics.

Complete every function marked with TODO. Do not change function signatures.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_regression(y_true, y_pred):
    """Evaluate a regression model using multiple metrics.

    You must compute all three metrics. Relying on a single metric is
    insufficient.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        dict: Dictionary with keys 'mae', 'rmse', and 'r2'.
    """
    # TODO: Compute Mean Absolute Error
    # TODO: Compute Root Mean Squared Error (square root of MSE)
    # TODO: Compute R-squared
    # TODO: Return a dict with keys 'mae', 'rmse', 'r2'
    raise NotImplementedError("TODO: Implement evaluate_regression")


def evaluate_classification(y_true, y_pred, y_prob=None):
    """Evaluate a classification model using multiple metrics.

    You must compute accuracy, precision, recall, and F1. If predicted
    probabilities are provided, also compute AUC-ROC.

    IMPORTANT: Do NOT rely exclusively on accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        dict: Dictionary with keys 'accuracy', 'precision', 'recall', 'f1'.
              If y_prob is not None, also include 'auc_roc'.
    """
    # TODO: Compute accuracy, precision, recall, and F1
    # TODO: If y_prob is provided, also compute AUC-ROC
    # TODO: Return the results dictionary
    raise NotImplementedError("TODO: Implement evaluate_classification")


def compare_models(results_dict):
    """Compare multiple models and identify the best one.

    For regression results (detected by the presence of the 'r2' key), use
    R-squared as the primary metric (higher is better).

    For classification results, use F1 as the primary metric (higher is better).

    Args:
        results_dict (dict): Maps model names (str) to metric dicts.
            Example:
                {
                    'DummyRegressor': {'mae': 100000, 'rmse': 120000, 'r2': 0.0},
                    'LinearRegression': {'mae': 20000, 'rmse': 25000, 'r2': 0.85},
                }

    Returns:
        dict: Dictionary with keys:
            - 'best_model' (str): Name of the best-performing model
            - 'comparison' (dict): The full results_dict passed in
            - 'primary_metric' (str): The metric used to rank models
    """
    # TODO: Detect whether this is regression or classification results
    # TODO: Select the appropriate primary metric
    # TODO: Find the model with the highest value of the primary metric
    # TODO: Return the structured comparison result
    raise NotImplementedError("TODO: Implement compare_models")
