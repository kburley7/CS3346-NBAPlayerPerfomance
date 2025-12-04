"""
evaluation.py

Loads trained regression models and test data, evaluates performance, and prints metrics.

Metrics (regression):
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² score
- Feature Importance plot (for tree-based models)
"""

import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = "models"
X_TEST_PATH = "data/processed/X_test.npy"
Y_TEST_PATH = "data/processed/y_test.npy"
FEATURE_NAMES_PATH = "data/processed/feature_names.npy"


def load_test_data():
    """Load test feature and label arrays."""
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError("Test data not found. Did you run models.py first?")
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    return X_test, y_test


def load_model(name: str):
    """Load a trained model by name."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def plot_feature_importance(
    model, feature_names, model_name: str, top_n: int = 15
) -> None:
    """Plot and save feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️  Model {model_name} does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_names)), top_importances, align="center")
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(MODELS_DIR, f"{model_name}_feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"📈 Feature importance plot saved to {plot_path}")
    plt.close()

    print(f"\n🔍 Top {min(10, len(top_names))} Features:")
    for i, (name, importance) in enumerate(
        zip(top_names[:10], top_importances[:10]), 1
    ):
        print(f"  {i}. {name:25s} {importance:.4f}")


def evaluate_model(model, X_test, y_test, model_name: str, feature_names=None) -> None:
    """Evaluate a regression model."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Results for {model_name} (regression):")
    print(f"  MAE (mean abs error):   {mae:.3f}")
    print(f"  MSE (mean sq error):    {mse:.3f}")
    print(f"  R² score:               {r2:.3f}")

    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name)

    print("-" * 50)


def main():
    X_test, y_test = load_test_data()

    feature_names = None
    if os.path.exists(FEATURE_NAMES_PATH):
        feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)

    model_names = [
        "random_forest_target_pts",
        # If you later train models for rebounds/assists:
        # "random_forest_target_trb",
        # "random_forest_target_ast",
    ]

    for name in model_names:
        model = load_model(name)
        evaluate_model(model, X_test, y_test, model_name=name, feature_names=feature_names)


if __name__ == "__main__":
    main()
