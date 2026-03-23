"""
Model training and algorithm selection.

Complete every function marked with TODO. Do not change function signatures.
"""

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


def train_dummy_regressor(X_train, y_train, strategy="mean"):
    """Train a DummyRegressor baseline.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        strategy (str): Strategy for the dummy regressor.

    Returns:
        DummyRegressor: The fitted model.
    """
    # TODO: Create a DummyRegressor with the given strategy
    # TODO: Fit it on the training data
    # TODO: Return the fitted model
    raise NotImplementedError("TODO: Implement train_dummy_regressor")


def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        LinearRegression: The fitted model.
    """
    # TODO: Create a LinearRegression instance
    # TODO: Fit it on the training data
    # TODO: Return the fitted model
    raise NotImplementedError("TODO: Implement train_linear_regression")


def train_dummy_classifier(X_train, y_train, strategy="most_frequent"):
    """Train a DummyClassifier baseline.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        strategy (str): Strategy for the dummy classifier.

    Returns:
        DummyClassifier: The fitted model.
    """
    # TODO: Create a DummyClassifier with the given strategy and random_state=42
    # TODO: Fit it on the training data
    # TODO: Return the fitted model
    raise NotImplementedError("TODO: Implement train_dummy_classifier")


def train_logistic_regression(X_train, y_train, random_state=42, max_iter=1000):
    """Train a Logistic Regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        random_state (int): Random seed for reproducibility.
        max_iter (int): Maximum iterations for the solver.

    Returns:
        LogisticRegression: The fitted model.
    """
    # TODO: Create a LogisticRegression with the given random_state and max_iter
    # TODO: Fit it on the training data
    # TODO: Return the fitted model
    raise NotImplementedError("TODO: Implement train_logistic_regression")


def select_algorithm(task_type, data_characteristics):
    """Apply an algorithm selection framework to recommend a model.

    Given the task type and characteristics of the dataset, return a structured
    recommendation explaining which algorithm to use and why.

    Args:
        task_type (str): Either "regression" or "classification".
        data_characteristics (dict): Dictionary with keys:
            - 'n_samples' (int): Number of training samples
            - 'n_features' (int): Number of features
            - 'data_type' (str): One of 'tabular', 'text', 'image'
            - 'target_type' (str): One of 'continuous', 'binary', 'multiclass'

    Returns:
        dict: A dictionary with exactly these keys:
            - 'recommended_algorithm' (str): Name of the recommended algorithm
            - 'rationale' (str): A multi-sentence explanation of why this
              algorithm is appropriate given the task and data
            - 'alternatives' (list[str]): At least two alternative algorithms
    """
    # TODO: Examine the task_type and data_characteristics
    # TODO: For regression on tabular data, consider LinearRegression, Ridge, Lasso
    # TODO: For binary classification on tabular data, consider LogisticRegression
    # TODO: Build and return the recommendation dictionary
    raise NotImplementedError("TODO: Implement select_algorithm")
