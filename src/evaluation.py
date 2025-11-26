"""
evaluation.py

Loads trained models and test data, evaluates performance, and prints metrics.

Metrics:
- Accuracy
- ROC-AUC
- Brier Score
"""

import os
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)

MODELS_DIR = "models"
X_TEST_PATH = "data/processed/X_test.npy"
Y_TEST_PATH = "data/processed/y_test.npy"


def load_test_data():
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError("Test data not found. Did you run models.py first?")
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    return X_test, y_test


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a classifier with probability output.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model {model_name} does not support predict_proba.")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_test, y_proba)

    print(f"📊 Results for {model_name}:")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"  ROC-AUC:      {auc:.3f}")
    print(f"  Brier Score:  {brier:.3f}")
    print("-" * 40)

def main():
    X_test, y_test = load_test_data()

    # Must match the name used in models.py
    model_names = [
        "random_forest_target_pts_over",
        # If you later change TARGET_COL and retrain, update names here too:
        # "random_forest_target_trb_over",
        # "random_forest_target_ast_over",
    ]

    for name in model_names:
        model = load_model(name)
        evaluate_model(model, X_test, y_test, model_name=name)

if __name__ == "__main__":
    main()
