"""Tests for Milestone 6: Evaluation metrics and model comparison."""

import numpy as np
import pytest


class TestEvaluateRegression:

    def test_returns_required_keys(self):
        from src.evaluate import evaluate_regression
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = evaluate_regression(y_true, y_pred)
        assert isinstance(result, dict)
        for key in ("mae", "rmse", "r2"):
            assert key in result, f"Missing key: {key}"

    def test_perfect_predictions(self):
        from src.evaluate import evaluate_regression
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = evaluate_regression(y, y)
        assert result["mae"] == pytest.approx(0.0, abs=1e-9)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert result["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_known_values(self):
        from src.evaluate import evaluate_regression
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        result = evaluate_regression(y_true, y_pred)
        assert result["mae"] == pytest.approx(0.5, abs=1e-5)
        expected_rmse = np.sqrt(np.mean([0.25, 0.25, 0.0, 1.0]))
        assert result["rmse"] == pytest.approx(expected_rmse, abs=1e-5)


class TestEvaluateClassification:

    def test_returns_required_keys(self):
        from src.evaluate import evaluate_classification
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        result = evaluate_classification(y_true, y_pred)
        assert isinstance(result, dict)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in result, f"Missing key: {key}"

    def test_auc_roc_included_when_proba_given(self):
        from src.evaluate import evaluate_classification
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.4, 0.2])
        result = evaluate_classification(y_true, y_pred, y_prob=y_prob)
        assert "auc_roc" in result

    def test_auc_roc_excluded_when_no_proba(self):
        from src.evaluate import evaluate_classification
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        result = evaluate_classification(y_true, y_pred)
        assert "auc_roc" not in result

    def test_perfect_predictions(self):
        from src.evaluate import evaluate_classification
        y = np.array([0, 1, 1, 0, 1])
        result = evaluate_classification(y, y)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_known_values(self):
        from src.evaluate import evaluate_classification
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1])
        result = evaluate_classification(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(4.0 / 6.0, abs=1e-5)
        assert result["precision"] == pytest.approx(2.0 / 3.0, abs=1e-5)
        assert result["recall"] == pytest.approx(2.0 / 3.0, abs=1e-5)


class TestCompareModels:

    def test_regression_comparison(self):
        from src.evaluate import compare_models
        results = {
            "DummyRegressor": {"mae": 100000, "rmse": 120000, "r2": 0.0},
            "LinearRegression": {"mae": 20000, "rmse": 25000, "r2": 0.85},
        }
        comparison = compare_models(results)
        assert isinstance(comparison, dict)
        assert comparison["best_model"] == "LinearRegression"
        assert comparison["primary_metric"] == "r2"
        assert comparison["comparison"] is results

    def test_classification_comparison(self):
        from src.evaluate import compare_models
        results = {
            "DummyClassifier": {
                "accuracy": 0.65,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "LogisticRegression": {
                "accuracy": 0.78,
                "precision": 0.72,
                "recall": 0.68,
                "f1": 0.70,
            },
        }
        comparison = compare_models(results)
        assert comparison["best_model"] == "LogisticRegression"
        assert comparison["primary_metric"] == "f1"

    def test_returns_required_keys(self):
        from src.evaluate import compare_models
        results = {
            "A": {"mae": 10, "rmse": 12, "r2": 0.9},
            "B": {"mae": 20, "rmse": 25, "r2": 0.7},
        }
        comparison = compare_models(results)
        for key in ("best_model", "comparison", "primary_metric"):
            assert key in comparison, f"Missing key: {key}"
