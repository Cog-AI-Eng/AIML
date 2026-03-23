"""
Evaluation metrics and model comparison utilities.

Complete the TODO sections to implement classification metrics,
ROC curve plotting, and a model comparison pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute precision, recall, and F1-score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys "precision", "recall", and "f1" mapped to
        their respective float values.
    """
    # TODO: Use sklearn.metrics.precision_score, recall_score, and
    # f1_score to compute the metrics. Return them in a dict with keys
    # "precision", "recall", and "f1".

    raise NotImplementedError("Complete the compute_precision_recall_f1 function")


def compute_auc_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Compute the Area Under the ROC Curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_scores : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        The AUC-ROC score.
    """
    # TODO: Use sklearn.metrics.roc_auc_score to compute and return
    # the AUC-ROC score.

    raise NotImplementedError("Complete the compute_auc_roc function")


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (2, 2).
    """
    # TODO: Use sklearn.metrics.confusion_matrix to compute and return
    # the confusion matrix.

    raise NotImplementedError("Complete the compute_confusion_matrix function")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = "roc_curve.png",
    title: str = "ROC Curve",
) -> str:
    """Plot the ROC curve and save it to a file.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_scores : np.ndarray
        Predicted probabilities for the positive class.
    save_path : str
        File path to save the plot.
    title : str
        Title for the plot.

    Returns
    -------
    str
        The file path where the plot was saved.
    """
    # TODO:
    #   1. Use sklearn.metrics.roc_curve to get fpr, tpr, thresholds.
    #   2. Compute AUC using sklearn.metrics.auc.
    #   3. Create a matplotlib figure and plot fpr vs tpr.
    #   4. Add a diagonal reference line (random classifier).
    #   5. Label axes, set title (include AUC in title or legend).
    #   6. Save the figure to save_path and close it.
    #   7. Return save_path.

    raise NotImplementedError("Complete the plot_roc_curve function")


def compare_models(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compare multiple fitted models on the test set.

    Parameters
    ----------
    models : Dict[str, object]
        Dictionary mapping model names to fitted sklearn model objects.
        Each model must support predict() and predict_proba().
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: "model", "precision", "recall", "f1",
        "auc_roc". One row per model.
    """
    # TODO:
    #   1. Iterate over models dict.
    #   2. For each model, get predictions and predicted probabilities.
    #   3. Compute precision, recall, f1 (use compute_precision_recall_f1).
    #   4. Compute AUC-ROC (use compute_auc_roc).
    #   5. Collect results into a list of dicts.
    #   6. Return a DataFrame from that list.

    raise NotImplementedError("Complete the compare_models function")
