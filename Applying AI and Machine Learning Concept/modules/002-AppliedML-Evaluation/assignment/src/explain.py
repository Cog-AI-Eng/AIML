"""
Solution: Model explainability using SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from typing import Optional


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[list] = None,
) -> np.ndarray:
    """Compute SHAP values for a fitted model."""
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values[:, :, 1]


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[list] = None,
    save_path: str = "shap_summary.png",
) -> str:
    """Generate and save a SHAP summary (beeswarm) plot."""
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
