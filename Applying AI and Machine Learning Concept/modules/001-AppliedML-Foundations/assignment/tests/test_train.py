"""Tests for Milestones 4, 5, and 7: Model training and algorithm selection."""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, f1_score


class TestDummyRegressor:

    def test_returns_fitted_model(self, housing_split):
        from src.train import train_dummy_regressor
        X_train, _, y_train, _ = housing_split
        model = train_dummy_regressor(X_train, y_train)
        assert isinstance(model, DummyRegressor)

    def test_can_predict(self, housing_split):
        from src.train import train_dummy_regressor
        X_train, X_test, y_train, _ = housing_split
        model = train_dummy_regressor(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]


class TestLinearRegression:

    def test_returns_fitted_model(self, housing_split):
        from src.train import train_linear_regression
        X_train, _, y_train, _ = housing_split
        model = train_linear_regression(X_train, y_train)
        assert isinstance(model, LinearRegression)

    def test_r2_above_threshold(self, housing_split):
        from src.train import train_linear_regression
        X_train, X_test, y_train, y_test = housing_split
        model = train_linear_regression(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        assert r2 > 0.5, f"R2={r2:.3f} is below the minimum threshold of 0.5"

    def test_beats_dummy(self, housing_split):
        from src.train import train_linear_regression, train_dummy_regressor
        X_train, X_test, y_train, y_test = housing_split
        dummy = train_dummy_regressor(X_train, y_train)
        model = train_linear_regression(X_train, y_train)
        dummy_r2 = r2_score(y_test, dummy.predict(X_test))
        model_r2 = r2_score(y_test, model.predict(X_test))
        assert model_r2 > dummy_r2, (
            f"LinearRegression R2={model_r2:.3f} must exceed "
            f"DummyRegressor R2={dummy_r2:.3f}"
        )


class TestDummyClassifier:

    def test_returns_fitted_model(self, churn_split):
        from src.train import train_dummy_classifier
        X_train, _, y_train, _ = churn_split
        model = train_dummy_classifier(X_train, y_train)
        assert isinstance(model, DummyClassifier)

    def test_can_predict(self, churn_split):
        from src.train import train_dummy_classifier
        X_train, X_test, y_train, _ = churn_split
        model = train_dummy_classifier(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]


class TestLogisticRegression:

    def test_returns_fitted_model(self, churn_split):
        from src.train import train_logistic_regression
        X_train, _, y_train, _ = churn_split
        model = train_logistic_regression(X_train, y_train)
        assert isinstance(model, LogisticRegression)

    def test_f1_above_threshold(self, churn_split):
        from src.train import train_logistic_regression
        X_train, X_test, y_train, y_test = churn_split
        model = train_logistic_regression(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, zero_division=0)
        assert f1 > 0.1, f"F1={f1:.3f} is below the minimum threshold of 0.1"

    def test_beats_dummy(self, churn_split):
        from src.train import train_logistic_regression, train_dummy_classifier
        X_train, X_test, y_train, y_test = churn_split
        dummy = train_dummy_classifier(X_train, y_train)
        model = train_logistic_regression(X_train, y_train)
        dummy_f1 = f1_score(y_test, dummy.predict(X_test), zero_division=0)
        model_f1 = f1_score(y_test, model.predict(X_test), zero_division=0)
        assert model_f1 > dummy_f1, (
            f"LogisticRegression F1={model_f1:.3f} must exceed "
            f"DummyClassifier F1={dummy_f1:.3f}"
        )

    def test_reproducibility(self, churn_split):
        from src.train import train_logistic_regression
        X_train, X_test, y_train, _ = churn_split
        model_a = train_logistic_regression(X_train, y_train, random_state=42)
        model_b = train_logistic_regression(X_train, y_train, random_state=42)
        np.testing.assert_array_equal(
            model_a.predict(X_test), model_b.predict(X_test)
        )


class TestSelectAlgorithm:

    def test_regression_returns_valid_structure(self):
        from src.train import select_algorithm
        result = select_algorithm("regression", {
            "n_samples": 500,
            "n_features": 7,
            "data_type": "tabular",
            "target_type": "continuous",
        })
        assert isinstance(result, dict)
        assert "recommended_algorithm" in result
        assert "rationale" in result
        assert "alternatives" in result
        assert isinstance(result["recommended_algorithm"], str)
        assert len(result["recommended_algorithm"]) > 0
        assert len(result["rationale"]) > 10
        assert isinstance(result["alternatives"], list)
        assert len(result["alternatives"]) >= 2

    def test_classification_returns_valid_structure(self):
        from src.train import select_algorithm
        result = select_algorithm("classification", {
            "n_samples": 800,
            "n_features": 7,
            "data_type": "tabular",
            "target_type": "binary",
        })
        assert isinstance(result, dict)
        assert "recommended_algorithm" in result
        assert "rationale" in result
        assert "alternatives" in result
        assert len(result["rationale"]) > 10
        assert len(result["alternatives"]) >= 2

    def test_different_tasks_give_different_recommendations(self):
        from src.train import select_algorithm
        reg = select_algorithm("regression", {
            "n_samples": 500,
            "n_features": 7,
            "data_type": "tabular",
            "target_type": "continuous",
        })
        clf = select_algorithm("classification", {
            "n_samples": 800,
            "n_features": 7,
            "data_type": "tabular",
            "target_type": "binary",
        })
        assert reg["recommended_algorithm"] != clf["recommended_algorithm"], (
            "Regression and classification tasks should yield different algorithms"
        )
