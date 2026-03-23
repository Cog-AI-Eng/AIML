"""Shared fixtures for all test modules."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import generate_readmission_data, get_feature_columns


@pytest.fixture(scope="session")
def dataset():
    """Generate the synthetic dataset once for all tests."""
    df = generate_readmission_data(n_samples=800, imbalance_ratio=0.15, random_state=42)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["readmitted"].values
    return X, y, feature_cols


@pytest.fixture(scope="session")
def small_dataset():
    """A smaller dataset for faster tests (e.g., SHAP)."""
    df = generate_readmission_data(n_samples=200, imbalance_ratio=0.15, random_state=42)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["readmitted"].values
    return X, y, feature_cols
