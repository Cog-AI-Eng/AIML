"""Tests for Milestones 2 and 3: Feature preparation, splitting, and scaling."""

import numpy as np
import pandas as pd
import pytest


class TestPrepareHousingFeatures:

    def test_returns_tuple(self, housing_df):
        from src.features import prepare_housing_features
        result = prepare_housing_features(housing_df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_target_excluded_from_features(self, housing_df):
        from src.features import prepare_housing_features
        X, y = prepare_housing_features(housing_df)
        assert "price" not in X.columns

    def test_correct_target(self, housing_df):
        from src.features import prepare_housing_features
        X, y = prepare_housing_features(housing_df)
        assert y.name == "price"
        assert len(y) == len(housing_df)

    def test_feature_count(self, housing_df):
        from src.features import prepare_housing_features
        X, y = prepare_housing_features(housing_df)
        assert X.shape[1] == len(housing_df.columns) - 1


class TestPrepareChurnFeatures:

    def test_returns_tuple(self, churn_df):
        from src.features import prepare_churn_features
        result = prepare_churn_features(churn_df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_target_excluded_from_features(self, churn_df):
        from src.features import prepare_churn_features
        X, y = prepare_churn_features(churn_df)
        assert "churn" not in X.columns

    def test_correct_target(self, churn_df):
        from src.features import prepare_churn_features
        X, y = prepare_churn_features(churn_df)
        assert y.name == "churn"
        assert set(y.unique()).issubset({0, 1})


class TestSplitData:

    def test_output_shapes(self, housing_df):
        from src.features import prepare_housing_features, split_data
        X, y = prepare_housing_features(housing_df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) == 400
        assert len(X_test) == 100
        assert len(y_train) == 400
        assert len(y_test) == 100

    def test_reproducibility(self, housing_df):
        from src.features import prepare_housing_features, split_data
        X, y = prepare_housing_features(housing_df)
        _, X_test_a, _, _ = split_data(X, y, test_size=0.2, random_state=42)
        _, X_test_b, _, _ = split_data(X, y, test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(X_test_a, X_test_b)

    def test_no_overlap(self, housing_df):
        from src.features import prepare_housing_features, split_data
        X, y = prepare_housing_features(housing_df)
        X_train, X_test, _, _ = split_data(X, y, test_size=0.2)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx & test_idx) == 0


class TestScaleFeatures:

    def test_output_types(self):
        from src.features import scale_features
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"a": [7.0], "b": [8.0]})
        X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)
        assert isinstance(X_tr_s, np.ndarray)
        assert isinstance(X_te_s, np.ndarray)

    def test_output_shapes(self):
        from src.features import scale_features
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"a": [7.0, 8.0], "b": [9.0, 10.0]})
        X_tr_s, X_te_s, _ = scale_features(X_train, X_test)
        assert X_tr_s.shape == (3, 2)
        assert X_te_s.shape == (2, 2)

    def test_train_is_standardized(self):
        from src.features import scale_features
        X_train = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        X_test = pd.DataFrame({"a": [6.0, 7.0], "b": [60.0, 70.0]})
        X_tr_s, _, _ = scale_features(X_train, X_test)
        np.testing.assert_array_almost_equal(
            X_tr_s.mean(axis=0), [0.0, 0.0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            X_tr_s.std(axis=0), [1.0, 1.0], decimal=3
        )

    def test_no_data_leakage(self):
        """Scaler must be fit on training data only."""
        from src.features import scale_features
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        X_test = pd.DataFrame({"a": [100.0, 200.0], "b": [1000.0, 2000.0]})
        _, X_te_s, scaler = scale_features(X_train, X_test)
        np.testing.assert_array_almost_equal(scaler.mean_, [2.0, 20.0], decimal=5)
        assert np.all(np.abs(X_te_s) > 2.0), (
            "Scaled test values should be far from zero when test distribution "
            "differs greatly from train distribution"
        )
