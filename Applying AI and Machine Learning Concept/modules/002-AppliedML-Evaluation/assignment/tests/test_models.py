"""Tests for src/models.py -- model training utilities."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.splits import train_val_test_split
from src.models import train_ridge_model, train_lasso_model, train_model_with_early_stopping


@pytest.fixture(scope="module")
def split_data(dataset):
    X, y, _ = dataset
    return train_val_test_split(X, y, random_state=42)


class TestRidgeModel:
    """Tests for train_ridge_model (Required)."""

    def test_returns_logistic_regression(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_ridge_model(X_train, y_train)
        assert isinstance(model, LogisticRegression)

    def test_uses_l2_penalty(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_ridge_model(X_train, y_train)
        assert model.penalty == "l2", "Ridge model must use L2 penalty"

    def test_model_is_fitted(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_ridge_model(X_train, y_train)
        assert hasattr(model, "classes_"), "Model must be fitted"
        assert hasattr(model, "coef_"), "Model must have learned coefficients"

    def test_can_predict(self, split_data):
        X_train, X_val, _, y_train, _, _ = split_data
        model = train_ridge_model(X_train, y_train)
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)

    def test_respects_C_parameter(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_ridge_model(X_train, y_train, C=0.01)
        assert model.C == 0.01


class TestLassoModel:
    """Tests for train_lasso_model (Required)."""

    def test_returns_logistic_regression(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_lasso_model(X_train, y_train)
        assert isinstance(model, LogisticRegression)

    def test_uses_l1_penalty(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_lasso_model(X_train, y_train)
        assert model.penalty == "l1", "Lasso model must use L1 penalty"

    def test_model_is_fitted(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_lasso_model(X_train, y_train)
        assert hasattr(model, "classes_"), "Model must be fitted"
        assert hasattr(model, "coef_"), "Model must have learned coefficients"

    def test_can_predict_proba(self, split_data):
        X_train, X_val, _, y_train, _, _ = split_data
        model = train_lasso_model(X_train, y_train)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

    def test_respects_C_parameter(self, split_data):
        X_train, _, _, y_train, _, _ = split_data
        model = train_lasso_model(X_train, y_train, C=0.1)
        assert model.C == 0.1


class TestEarlyStopping:
    """Tests for train_model_with_early_stopping (Preferred)."""

    def test_returns_correct_dict_keys(self, split_data):
        X_train, X_val, _, y_train, y_val, _ = split_data
        result = train_model_with_early_stopping(
            X_train, y_train, X_val, y_val, max_epochs=30, patience=5
        )
        assert isinstance(result, dict)
        for key in ["model", "best_epoch", "train_losses", "val_losses"]:
            assert key in result, f"Missing key: {key}"

    def test_model_can_predict(self, split_data):
        X_train, X_val, _, y_train, y_val, _ = split_data
        result = train_model_with_early_stopping(
            X_train, y_train, X_val, y_val, max_epochs=30, patience=5
        )
        preds = result["model"].predict(X_val)
        assert len(preds) == len(X_val)

    def test_losses_are_recorded(self, split_data):
        X_train, X_val, _, y_train, y_val, _ = split_data
        result = train_model_with_early_stopping(
            X_train, y_train, X_val, y_val, max_epochs=50, patience=5
        )
        assert len(result["train_losses"]) > 0
        assert len(result["val_losses"]) > 0
        assert len(result["train_losses"]) == len(result["val_losses"])

    def test_early_stopping_triggers(self, split_data):
        X_train, X_val, _, y_train, y_val, _ = split_data
        result = train_model_with_early_stopping(
            X_train, y_train, X_val, y_val, max_epochs=200, patience=5
        )
        assert len(result["train_losses"]) < 200, (
            "Early stopping should trigger before max_epochs"
        )

    def test_best_epoch_is_valid(self, split_data):
        X_train, X_val, _, y_train, y_val, _ = split_data
        result = train_model_with_early_stopping(
            X_train, y_train, X_val, y_val, max_epochs=50, patience=5
        )
        assert 0 <= result["best_epoch"] < len(result["val_losses"])
