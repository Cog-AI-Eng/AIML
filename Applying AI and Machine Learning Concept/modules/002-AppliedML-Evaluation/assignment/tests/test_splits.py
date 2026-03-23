"""Tests for src/splits.py -- data splitting utilities."""

import numpy as np
import pytest

from src.splits import train_val_test_split, stratified_kfold_split


class TestTrainValTestSplit:
    """Tests for the train_val_test_split function (Required)."""

    def test_returns_six_arrays(self, dataset):
        X, y, _ = dataset
        result = train_val_test_split(X, y)
        assert len(result) == 6, "Should return 6 arrays"

    def test_correct_split_proportions(self, dataset):
        X, y, _ = dataset
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15
        )
        n = len(y)
        assert abs(len(y_train) / n - 0.7) < 0.05, "Train split proportion off"
        assert abs(len(y_val) / n - 0.15) < 0.05, "Val split proportion off"
        assert abs(len(y_test) / n - 0.15) < 0.05, "Test split proportion off"

    def test_no_data_leakage(self, dataset):
        X, y, _ = dataset
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        total = len(y_train) + len(y_val) + len(y_test)
        assert total == len(y), "Total samples must equal original"

    def test_stratification_preserves_class_ratio(self, dataset):
        X, y, _ = dataset
        _, _, _, y_train, y_val, y_test = train_val_test_split(
            X, y, stratify=True
        )
        original_ratio = y.mean()
        for split_y, name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
            split_ratio = split_y.mean()
            assert abs(split_ratio - original_ratio) < 0.05, (
                f"Stratification failed in {name} split: "
                f"expected ~{original_ratio:.3f}, got {split_ratio:.3f}"
            )

    def test_reproducibility(self, dataset):
        X, y, _ = dataset
        result1 = train_val_test_split(X, y, random_state=99)
        result2 = train_val_test_split(X, y, random_state=99)
        for a, b in zip(result1, result2):
            np.testing.assert_array_equal(a, b)

    def test_invalid_proportions_raises_error(self, dataset):
        X, y, _ = dataset
        with pytest.raises(ValueError):
            train_val_test_split(X, y, train_size=0.5, val_size=0.3, test_size=0.3)

    def test_features_and_labels_aligned(self, dataset):
        X, y, _ = dataset
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


class TestStratifiedKFoldSplit:
    """Tests for the stratified_kfold_split function (Preferred)."""

    def test_returns_correct_number_of_folds(self, dataset):
        X, y, _ = dataset
        folds = stratified_kfold_split(X, y, n_splits=5)
        assert len(folds) == 5, "Should return 5 folds"

    def test_each_fold_has_train_and_test(self, dataset):
        X, y, _ = dataset
        folds = stratified_kfold_split(X, y, n_splits=5)
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0, "Train indices must not be empty"
            assert len(test_idx) > 0, "Test indices must not be empty"

    def test_no_overlap_between_train_and_test(self, dataset):
        X, y, _ = dataset
        folds = stratified_kfold_split(X, y, n_splits=5)
        for train_idx, test_idx in folds:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, "Train and test indices must not overlap"

    def test_all_indices_covered(self, dataset):
        X, y, _ = dataset
        folds = stratified_kfold_split(X, y, n_splits=5)
        all_test = np.concatenate([test_idx for _, test_idx in folds])
        assert len(np.unique(all_test)) == len(y), "All samples must appear in test exactly once"

    def test_stratification_in_folds(self, dataset):
        X, y, _ = dataset
        folds = stratified_kfold_split(X, y, n_splits=5)
        original_ratio = y.mean()
        for train_idx, test_idx in folds:
            fold_ratio = y[test_idx].mean()
            assert abs(fold_ratio - original_ratio) < 0.08, (
                f"Fold class ratio {fold_ratio:.3f} deviates from "
                f"original {original_ratio:.3f}"
            )
