"""
accuracy_breakdown.py

Generates a bar chart showing the percentage of predictions within ±3, ±5,
and ±10 points (plus beyond ±10) using the saved model/test split.

Usage:
    python accuracy_breakdown.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "models/random_forest_target_pts.joblib"
X_PATH = "data/processed/X_test.npy"
Y_PATH = "data/processed/y_test.npy"
OUTPUT_DIR = "reports"
OUTPUT_FILE = "accuracy_breakdown.png"


def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError("Test split files missing.")

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_PATH)
    y_test = np.load(Y_PATH)
    return model, X_test, y_test


def main():
    model, X_test, y_test = load_resources()
    y_pred = model.predict(X_test)
    errors = np.abs(y_pred - y_test)

    within_3 = np.mean(errors < 3) * 100
    within_5 = np.mean(errors < 5) * 100
    within_10 = np.mean(errors < 10) * 100
    beyond_10 = 100 - within_10

    categories = ["Within 3 pts", "Within 5 pts", "Within 10 pts", "Beyond 10 pts"]
    values = [within_3, within_5 - within_3, within_10 - within_5, beyond_10]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(categories, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage of Predictions", fontsize=12)
    ax.set_title("Prediction Accuracy Breakdown", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.7)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Accuracy breakdown saved to {output_path}")


if __name__ == "__main__":
    main()
