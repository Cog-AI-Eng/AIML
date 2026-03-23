"""Tests for src/explain.py -- SHAP explainability (Preferred)."""

import os
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.explain import compute_shap_values, plot_shap_summary


@pytest.fixture(scope="module")
def fitted_model_and_data(small_dataset):
    X, y, feature_names = small_dataset
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
    )
    model.fit(X, y)
    return model, X, y, feature_names


class TestComputeShapValues:
    """Tests for compute_shap_values (Preferred)."""

    def test_returns_ndarray(self, fitted_model_and_data):
        model, X, _, feature_names = fitted_model_and_data
        X_bg = X[:30]
        X_explain = X[:10]
        shap_vals = compute_shap_values(model, X_bg, X_explain, feature_names)
        assert isinstance(shap_vals, np.ndarray)

    def test_correct_shape(self, fitted_model_and_data):
        model, X, _, feature_names = fitted_model_and_data
        X_bg = X[:30]
        X_explain = X[:10]
        shap_vals = compute_shap_values(model, X_bg, X_explain, feature_names)
        assert shap_vals.shape == (10, X.shape[1]), (
            f"Expected shape (10, {X.shape[1]}), got {shap_vals.shape}"
        )

    def test_shap_values_are_finite(self, fitted_model_and_data):
        model, X, _, feature_names = fitted_model_and_data
        X_bg = X[:30]
        X_explain = X[:10]
        shap_vals = compute_shap_values(model, X_bg, X_explain, feature_names)
        assert np.all(np.isfinite(shap_vals)), "SHAP values must be finite"


class TestPlotShapSummary:
    """Tests for plot_shap_summary (Preferred)."""

    def test_saves_file(self, fitted_model_and_data, tmp_path):
        model, X, _, feature_names = fitted_model_and_data
        X_bg = X[:30]
        X_explain = X[:10]
        shap_vals = compute_shap_values(model, X_bg, X_explain, feature_names)

        save_path = str(tmp_path / "shap_test.png")
        result = plot_shap_summary(
            shap_vals, X_explain, feature_names=feature_names, save_path=save_path
        )
        assert os.path.exists(save_path), "SHAP summary plot was not saved"
        assert result == save_path

    def test_returns_path_string(self, fitted_model_and_data, tmp_path):
        model, X, _, feature_names = fitted_model_and_data
        X_bg = X[:30]
        X_explain = X[:5]
        shap_vals = compute_shap_values(model, X_bg, X_explain, feature_names)

        save_path = str(tmp_path / "shap_summary.png")
        result = plot_shap_summary(shap_vals, X_explain, save_path=save_path)
        assert isinstance(result, str)
