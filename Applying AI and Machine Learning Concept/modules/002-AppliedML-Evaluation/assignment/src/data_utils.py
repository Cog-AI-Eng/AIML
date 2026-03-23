"""
Synthetic data generation for patient readmission prediction.

This module is COMPLETE -- no TODOs here. Use it to generate the dataset
for the assignment.
"""

import numpy as np
import pandas as pd


def generate_readmission_data(
    n_samples: int = 2000,
    imbalance_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic patient readmission dataset with class imbalance.

    The dataset simulates hospital patient records with clinical features
    that correlate with 30-day readmission risk. The positive class
    (readmitted=1) is the minority class.

    Parameters
    ----------
    n_samples : int
        Total number of patient records to generate.
    imbalance_ratio : float
        Fraction of positive class (readmitted) samples. For example,
        0.15 means roughly 15% of patients are readmitted.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with clinical features and a binary 'readmitted' target.
    """
    rng = np.random.RandomState(random_state)

    n_positive = int(n_samples * imbalance_ratio)
    n_negative = n_samples - n_positive

    age_neg = rng.normal(55, 15, n_negative).clip(18, 100)
    age_pos = rng.normal(65, 12, n_positive).clip(18, 100)
    age = np.concatenate([age_neg, age_pos])

    num_medications_neg = rng.poisson(5, n_negative)
    num_medications_pos = rng.poisson(9, n_positive)
    num_medications = np.concatenate([num_medications_neg, num_medications_pos])

    num_procedures_neg = rng.poisson(1, n_negative)
    num_procedures_pos = rng.poisson(3, n_positive)
    num_procedures = np.concatenate([num_procedures_neg, num_procedures_pos])

    num_diagnoses_neg = rng.poisson(3, n_negative)
    num_diagnoses_pos = rng.poisson(6, n_positive)
    num_diagnoses = np.concatenate([num_diagnoses_neg, num_diagnoses_pos])

    time_in_hospital_neg = rng.exponential(3, n_negative).clip(1, 14)
    time_in_hospital_pos = rng.exponential(6, n_positive).clip(1, 14)
    time_in_hospital = np.concatenate([time_in_hospital_neg, time_in_hospital_pos])

    num_lab_procedures_neg = rng.poisson(40, n_negative)
    num_lab_procedures_pos = rng.poisson(55, n_positive)
    num_lab_procedures = np.concatenate([num_lab_procedures_neg, num_lab_procedures_pos])

    num_outpatient_neg = rng.poisson(0.3, n_negative)
    num_outpatient_pos = rng.poisson(1.2, n_positive)
    num_outpatient = np.concatenate([num_outpatient_neg, num_outpatient_pos])

    num_emergency_neg = rng.poisson(0.2, n_negative)
    num_emergency_pos = rng.poisson(0.8, n_positive)
    num_emergency = np.concatenate([num_emergency_neg, num_emergency_pos])

    num_inpatient_neg = rng.poisson(0.3, n_negative)
    num_inpatient_pos = rng.poisson(1.5, n_positive)
    num_inpatient = np.concatenate([num_inpatient_neg, num_inpatient_pos])

    hba1c_neg = rng.normal(5.5, 0.8, n_negative).clip(4, 14)
    hba1c_pos = rng.normal(7.5, 1.5, n_positive).clip(4, 14)
    hba1c = np.concatenate([hba1c_neg, hba1c_pos])

    target = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])

    df = pd.DataFrame({
        "age": np.round(age, 1),
        "num_medications": num_medications,
        "num_procedures": num_procedures,
        "num_diagnoses": num_diagnoses,
        "time_in_hospital": np.round(time_in_hospital, 1),
        "num_lab_procedures": num_lab_procedures,
        "num_outpatient": num_outpatient,
        "num_emergency": num_emergency,
        "num_inpatient": num_inpatient,
        "hba1c_level": np.round(hba1c, 2),
        "readmitted": target.astype(int),
    })

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature column names (everything except target)."""
    return [c for c in df.columns if c != "readmitted"]


def get_target_column() -> str:
    """Return the name of the target column."""
    return "readmitted"
