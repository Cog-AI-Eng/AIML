"""Training script for the FraudShield SageMaker Pipeline.

Trains a RandomForestClassifier on the preprocessed Feature Store data
and persists the model artifact for registration.
"""

import argparse
import os
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = os.path.join(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"), "train.csv")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    df = pd.read_csv(train_path)
    y = df.pop("target")
    X = df.select_dtypes(include="number")

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    clf.fit(X, y)

    print(classification_report(y, clf.predict(X), target_names=["legit", "fraud"]))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
    print(f"Model saved to {model_dir}/model.joblib")
