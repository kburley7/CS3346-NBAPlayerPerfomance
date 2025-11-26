"""
evaluation.py

Loads trained models and test data, evaluates performance, and prints metrics.

Metrics:
- Accuracy
- ROC-AUC
- Brier Score
- Feature Importance
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)

MODELS_DIR = "models"
X_TEST_PATH = "data/processed/X_test.npy"
Y_TEST_PATH = "data/processed/y_test.npy"
FEATURE_NAMES_PATH = "data/processed/feature_names.npy"


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


def plot_feature_importance(model, feature_names, model_name: str, top_n: int = 15):
    """
    Plot and save feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️  Model {model_name} does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Take top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_names)), top_importances, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save plot
    os.makedirs("models", exist_ok=True)
    plot_path = os.path.join("models", f"{model_name}_feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Feature importance plot saved to {plot_path}")
    plt.close()

    # Print top features
    print(f"\n🔍 Top {min(10, len(top_names))} Features:")
    for i, (name, importance) in enumerate(zip(top_names[:10], top_importances[:10]), 1):
        print(f"  {i}. {name:20s} {importance:.4f}")


def evaluate_model(model, X_test, y_test, model_name: str, feature_names=None):
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

    # Plot feature importance if available
    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name)

    print("-" * 40)

def main():
    X_test, y_test = load_test_data()

    # Load feature names if available
    feature_names = None
    if os.path.exists(FEATURE_NAMES_PATH):
        feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)

    # Must match the name used in models.py
    model_names = [
        "random_forest_target_pts_over",
        # If you later change TARGET_COL and retrain, update names here too:
        # "random_forest_target_trb_over",
        # "random_forest_target_ast_over",
    ]

    for name in model_names:
        model = load_model(name)
        evaluate_model(model, X_test, y_test, model_name=name, feature_names=feature_names)

if __name__ == "__main__":
    main()
