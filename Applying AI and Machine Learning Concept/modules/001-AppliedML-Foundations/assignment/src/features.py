"""
Feature engineering and preprocessing pipeline.

Complete every function marked with TODO. Do not change function signatures.
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_random_seed(seed=42):
    """Set random seeds for reproducibility.

    Must seed both numpy and Python's built-in random module so that all
    downstream operations produce deterministic results.

    Args:
        seed (int): The seed value to use.
    """
    # TODO: Set the numpy random seed
    # TODO: Set the built-in random module seed
    raise NotImplementedError("TODO: Implement set_random_seed")


def prepare_housing_features(df):
    """Separate features and target for the housing dataset.

    Args:
        df (pd.DataFrame): Raw housing DataFrame from data_utils.

    Returns:
        tuple: (X, y) where X is a DataFrame of features (all columns except
               'price') and y is a Series containing the 'price' column.
    """
    # TODO: Drop the target column to create X
    # TODO: Extract the target column as y
    # TODO: Return (X, y)
    raise NotImplementedError("TODO: Implement prepare_housing_features")


def prepare_churn_features(df):
    """Separate features and target for the churn dataset.

    Args:
        df (pd.DataFrame): Raw churn DataFrame from data_utils.

    Returns:
        tuple: (X, y) where X is a DataFrame of features (all columns except
               'churn') and y is a Series containing the 'churn' column.
    """
    # TODO: Drop the target column to create X
    # TODO: Extract the target column as y
    # TODO: Return (X, y)
    raise NotImplementedError("TODO: Implement prepare_churn_features")


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size (float): Proportion of data reserved for testing.
        random_state (int): Seed for reproducible splits.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: Use sklearn's train_test_split with the provided parameters
    # TODO: Return the four resulting arrays
    raise NotImplementedError("TODO: Implement split_data")


def scale_features(X_train, X_test):
    """Scale features using StandardScaler.

    IMPORTANT: Fit the scaler on X_train ONLY, then transform both X_train
    and X_test. Fitting on the combined data is data leakage and will cause
    tests to fail.

    Args:
        X_train: Training feature matrix.
        X_test:  Test feature matrix.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
            - X_train_scaled (ndarray): Scaled training features
            - X_test_scaled (ndarray): Scaled test features
            - scaler (StandardScaler): The fitted scaler instance
    """
    # TODO: Create a StandardScaler instance
    # TODO: Fit the scaler on X_train ONLY
    # TODO: Transform both X_train and X_test using the fitted scaler
    # TODO: Return (X_train_scaled, X_test_scaled, scaler)
    raise NotImplementedError("TODO: Implement scale_features")
