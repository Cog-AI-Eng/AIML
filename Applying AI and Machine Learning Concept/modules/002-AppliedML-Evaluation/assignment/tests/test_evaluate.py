"""Tests for src/evaluate.py -- evaluation metrics and comparison."""

import os
import numpy as np
import pandas as pd
import pytest

from src.splits import train_val_test_split
from src.models import train_ridge_model, train_lasso_model
from src.evaluate import (
    compute_precision_recall_f1,
    compute_auc_roc,
    compute_confusion_matrix,
    plot_roc_curve,
    compare_models,
)


@pytest.fixture(scope="module")
def trained_models(dataset):
    X, y, _ = dataset
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_state=42
    )
    ridge = train_ridge_model(X_train, y_train)
    lasso = train_lasso_model(X_train, y_train)
    return {
        "X_test": X_test,
        "y_test": y_test,
        "ridge": ridge,
        "lasso": lasso,
    }


class TestPrecisionRecallF1:
    """Tests for compute_precision_recall_f1 (Required)."""

    def test_returns_dict_with_correct_keys(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = compute_precision_recall_f1(y_true, y_pred)
        assert isinstance(result, dict)
        for key in ["precision", "recall", "f1"]:
            assert key in result, f"Missing key: {key}"

    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        result = compute_precision_recall_f1(y, y)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_values_are_floats_between_0_and_1(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        result = compute_precision_recall_f1(y_true, y_pred)
        for key in ["precision", "recall", "f1"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_known_values(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        result = compute_precision_recall_f1(y_true, y_pred)
        assert abs(result["precision"] - 2 / 3) < 0.01
        assert abs(result["recall"] - 2 / 3) < 0.01


class TestAUCROC:
    """Tests for compute_auc_roc (Required)."""

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.9, 0.95])
        auc = compute_auc_roc(y_true, y_scores)
        assert auc == 1.0

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_scores = np.array([0.3, 0.7, 0.4, 0.8, 0.2])
        auc = compute_auc_roc(y_true, y_scores)
        assert isinstance(auc, float)

    def test_value_between_0_and_1(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_scores = np.array([0.2, 0.6, 0.8, 0.3, 0.7, 0.1, 0.5, 0.4])
        auc = compute_auc_roc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0

    def test_with_model_predictions(self, trained_models):
        model = trained_models["ridge"]
        X_test = trained_models["X_test"]
        y_test = trained_models["y_test"]
        y_scores = model.predict_proba(X_test)[:, 1]
        auc = compute_auc_roc(y_test, y_scores)
        assert 0.5 < auc <= 1.0, "Trained model should do better than random"


class TestConfusionMatrix:
    """Tests for compute_confusion_matrix (Required)."""

    def test_returns_2x2_array(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        cm = compute_confusion_matrix(y, y)
        assert cm[0, 1] == 0, "No false positives expected"
        assert cm[1, 0] == 0, "No false negatives expected"

    def test_sums_to_total_samples(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.sum() == len(y_true)


class TestPlotROCCurve:
    """Tests for plot_roc_curve (Required)."""

    def test_saves_file(self, tmp_path):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])
        save_path = str(tmp_path / "test_roc.png")
        result = plot_roc_curve(y_true, y_scores, save_path=save_path)
        assert os.path.exists(save_path), "ROC curve plot was not saved"
        assert result == save_path

    def test_returns_path_string(self, tmp_path):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.2, 0.8, 0.3, 0.9])
        save_path = str(tmp_path / "roc.png")
        result = plot_roc_curve(y_true, y_scores, save_path=save_path)
        assert isinstance(result, str)


class TestCompareModels:
    """Tests for compare_models (Required)."""

    def test_returns_dataframe(self, trained_models):
        models = {
            "Ridge": trained_models["ridge"],
            "Lasso": trained_models["lasso"],
        }
        df = compare_models(models, trained_models["X_test"], trained_models["y_test"])
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, trained_models):
        models = {"Ridge": trained_models["ridge"]}
        df = compare_models(models, trained_models["X_test"], trained_models["y_test"])
        for col in ["model", "precision", "recall", "f1", "auc_roc"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_correct_number_of_rows(self, trained_models):
        models = {
            "Ridge": trained_models["ridge"],
            "Lasso": trained_models["lasso"],
        }
        df = compare_models(models, trained_models["X_test"], trained_models["y_test"])
        assert len(df) == 2

    def test_metrics_are_valid(self, trained_models):
        models = {"Ridge": trained_models["ridge"]}
        df = compare_models(models, trained_models["X_test"], trained_models["y_test"])
        row = df.iloc[0]
        for metric in ["precision", "recall", "f1", "auc_roc"]:
            assert 0.0 <= row[metric] <= 1.0, f"{metric} out of [0,1] range"
