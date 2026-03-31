"""Preprocessing script for the FraudShield SageMaker Pipeline.

Queries the Feature Store offline store via Athena, splits data into
train/validation sets, and writes CSVs to the configured output paths.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-group", type=str, default="fraudshield-transactions")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = "/opt/ml/processing/input"
    train_output = "/opt/ml/processing/output/train"
    val_output = "/opt/ml/processing/output/validation"

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    input_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith(".csv")
    ]
    df = pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)

    drop_cols = [c for c in df.columns if c.startswith("write_time") or c in ("api_invocation_time", "is_deleted", "event_time")]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    train_df, val_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state, stratify=df["target"])

    train_df.to_csv(os.path.join(train_output, "train.csv"), index=False)
    val_df.to_csv(os.path.join(val_output, "validation.csv"), index=False)

    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")
