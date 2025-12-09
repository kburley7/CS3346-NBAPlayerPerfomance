"""
plot_error_vs_actual.py

Creates a scatter plot of prediction error versus actual points scored.
Saves the figure in reports/error_vs_actual.png.

Usage:
    python plot_error_vs_actual.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "models/random_forest_target_pts.joblib"
X_PATH = "data/processed/X_test.npy"
Y_PATH = "data/processed/y_test.npy"
OUTPUT_DIR = "reports"
OUTPUT_FILE = "error_vs_actual.png"


def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(
            f"Test data missing: {X_PATH} and/or {Y_PATH} not found"
        )

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_PATH)
    y_test = np.load(Y_PATH)

    return model, X_test, y_test


def build_plot(actual, errors):
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        actual,
        errors,
        c=np.abs(errors),
        cmap="coolwarm",
        alpha=0.65,
        edgecolor="k",
        linewidth=0.25,
        s=40,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.25)
    ax.set_title("Prediction Error vs Actual Points", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Points Scored", fontsize=12)
    ax.set_ylabel("Prediction Error (Predicted - Actual)", fontsize=12)
    fig.colorbar(scatter, ax=ax, label="Absolute Error")

    mae = np.mean(np.abs(errors))
    ax.text(
        0.98,
        0.02,
        f"MAE: {mae:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def main():
    model, X_test, y_test = load_resources()
    predictions = model.predict(X_test)
    errors = predictions - y_test

    fig = build_plot(y_test, errors)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Error vs Actual plot saved to {output_path}")


if __name__ == "__main__":
    main()
