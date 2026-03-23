"""Shared fixtures for transformer assignment tests."""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
