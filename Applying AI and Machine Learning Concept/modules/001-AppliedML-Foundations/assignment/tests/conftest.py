import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import generate_housing_data, generate_churn_data


@pytest.fixture
def housing_df():
    """Raw housing DataFrame."""
    return generate_housing_data(n_samples=500, random_state=42)


@pytest.fixture
def churn_df():
    """Raw churn DataFrame."""
    return generate_churn_data(n_samples=800, random_state=42)


@pytest.fixture
def housing_split():
    """Pre-processed housing data split for model training tests."""
    df = generate_housing_data(n_samples=500, random_state=42)
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


@pytest.fixture
def churn_split():
    """Pre-processed churn data split for model training tests."""
    df = generate_churn_data(n_samples=800, random_state=42)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values
