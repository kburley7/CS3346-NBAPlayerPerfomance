"""
error_distribution.py

Creates a histogram of prediction errors and saves it to reports/error_distribution.png.

Usage:
    python error_distribution.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "models/random_forest_target_pts.joblib"
X_PATH = "data/processed/X_test.npy"
Y_PATH = "data/processed/y_test.npy"
OUTPUT_DIR = "reports"
OUTPUT_FILE = "error_distribution.png"


def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError("Test data files are missing.")

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_PATH)
    y_test = np.load(Y_PATH)
    return model, X_test, y_test


def main():
    model, X_test, y_test = load_resources()
    y_pred = model.predict(X_test)
    errors = y_pred - y_test

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=40, edgecolor="black", color="#5a8ddb", alpha=0.75)
    ax.set_title("Prediction Error Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.axvline(errors.mean(), color="red", linewidth=2, linestyle="--", label="Mean Error")
    ax.axvline(0, color="black", linewidth=1, linestyle=":")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Error distribution histogram saved to {output_path}")


if __name__ == "__main__":
    main()
