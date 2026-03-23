"""
Data splitting utilities for model evaluation.

Complete the TODO sections to implement train/val/test splitting and
stratified k-fold cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    train_size : float
        Proportion of data for training (default 0.7).
    val_size : float
        Proportion of data for validation (default 0.15).
    test_size : float
        Proportion of data for testing (default 0.15).
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        Whether to preserve class proportions in each split.

    Returns
    -------
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        Each is a numpy array.

    Raises
    ------
    ValueError
        If train_size + val_size + test_size does not equal 1.0
        (within floating point tolerance).
    """
    # TODO: Validate that split proportions sum to 1.0 (use np.isclose).
    # Raise ValueError if they do not.

    # TODO: Use sklearn.model_selection.train_test_split twice:
    #   1. First split: separate out the test set.
    #   2. Second split: separate the remaining data into train and validation.
    #   Handle the `stratify` parameter appropriately.

    # TODO: Return (X_train, X_val, X_test, y_train, y_val, y_test)

    raise NotImplementedError("Complete the train_val_test_split function")


def stratified_kfold_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold cross-validation indices.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    n_splits : int
        Number of folds (default 5).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        A list of (train_indices, test_indices) tuples, one per fold.
    """
    # TODO: Use sklearn.model_selection.StratifiedKFold to generate
    # train/test index pairs for each fold. Return them as a list.

    raise NotImplementedError("Complete the stratified_kfold_split function")
