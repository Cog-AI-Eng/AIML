"""Tests for Milestone 1: Reproducibility."""

import numpy as np
import random as stdlib_random


class TestSetRandomSeed:

    def test_numpy_seed_is_set(self):
        from src.features import set_random_seed
        set_random_seed(42)
        first = np.random.rand(5)
        set_random_seed(42)
        second = np.random.rand(5)
        np.testing.assert_array_equal(first, second)

    def test_stdlib_random_seed_is_set(self):
        from src.features import set_random_seed
        set_random_seed(42)
        first = [stdlib_random.random() for _ in range(5)]
        set_random_seed(42)
        second = [stdlib_random.random() for _ in range(5)]
        assert first == second

    def test_different_seeds_produce_different_results(self):
        from src.features import set_random_seed
        set_random_seed(42)
        result_a = np.random.rand(5)
        set_random_seed(99)
        result_b = np.random.rand(5)
        assert not np.array_equal(result_a, result_b)
