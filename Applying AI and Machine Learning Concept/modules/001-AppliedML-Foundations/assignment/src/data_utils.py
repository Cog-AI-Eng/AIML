"""
Synthetic data generation for the AppliedML-Foundations assignment.

DO NOT MODIFY THIS FILE. The test suite depends on deterministic data generation.
"""

import numpy as np
import pandas as pd


def generate_housing_data(n_samples=500, random_state=42):
    """Generate a synthetic housing dataset for regression.

    Features:
        square_feet, bedrooms, bathrooms, year_built,
        garage_size, lot_size, neighborhood_quality

    Target:
        price (continuous, in USD)
    """
    rng = np.random.RandomState(random_state)

    square_feet = rng.randint(600, 4500, n_samples)
    bedrooms = rng.randint(1, 7, n_samples)
    bathrooms = rng.randint(1, 5, n_samples)
    year_built = rng.randint(1950, 2024, n_samples)
    garage_size = rng.randint(0, 4, n_samples)
    lot_size = rng.randint(2000, 20000, n_samples)
    neighborhood_quality = rng.randint(1, 11, n_samples)

    price = (
        50 * square_feet
        + 10000 * bedrooms
        + 15000 * bathrooms
        + 500 * (year_built - 1950)
        + 20000 * garage_size
        + 5 * lot_size
        + 30000 * neighborhood_quality
        + rng.normal(0, 25000, n_samples)
    )
    price = np.maximum(price, 50000).astype(int)

    return pd.DataFrame({
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "garage_size": garage_size,
        "lot_size": lot_size,
        "neighborhood_quality": neighborhood_quality,
        "price": price,
    })


def generate_churn_data(n_samples=800, random_state=42):
    """Generate a synthetic customer churn dataset for binary classification.

    Features:
        tenure_months, monthly_charges, total_charges,
        contract_type (0=month-to-month, 1=one-year, 2=two-year),
        num_support_tickets, has_internet, has_phone

    Target:
        churn (0 or 1)
    """
    rng = np.random.RandomState(random_state)

    tenure_months = rng.randint(1, 73, n_samples)
    monthly_charges = rng.uniform(20, 120, n_samples).round(2)
    contract_type = rng.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    num_support_tickets = rng.poisson(2, n_samples)
    has_internet = rng.choice([0, 1], n_samples, p=[0.3, 0.7])
    has_phone = rng.choice([0, 1], n_samples, p=[0.2, 0.8])

    total_charges = (tenure_months * monthly_charges).round(2)

    logit = (
        -1.5
        - 0.04 * tenure_months
        + 0.02 * monthly_charges
        + 1.0 * (contract_type == 0).astype(float)
        + 0.2 * num_support_tickets
        - 0.3 * has_phone
        + rng.normal(0, 0.5, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "num_support_tickets": num_support_tickets,
        "has_internet": has_internet,
        "has_phone": has_phone,
        "churn": churn,
    })
