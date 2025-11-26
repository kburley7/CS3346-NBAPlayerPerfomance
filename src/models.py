"""
models.py

Trains a Random Forest classifier to predict whether a player will exceed
a chosen threshold for points, rebounds, or assists.

Target options (set TARGET_COL below):
- "target_pts_over"
- "target_trb_over"
- "target_ast_over"

Outputs:
- models/random_forest_<TARGET_COL>.joblib
- data/processed/X_test.npy
- data/processed/y_test.npy
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES_PATH = "data/processed/features.csv"
MODELS_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Choose which target to model:
# "target_pts_over", "target_trb_over", or "target_ast_over"
TARGET_COL = "target_pts_over"


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load feature dataframe with engineered features + targets."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found at: {path}")
    return pd.read_csv(path, parse_dates=["game_date"])


def train_test_split_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Features used:
    - all numeric engineered columns, excluding:
        IDs, dates, raw stats, and target columns
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found in data.")

    exclude_cols = [
        "Player",
        "Tm",
        "Opp",
        "game_date",
        "PTS",
        "TRB",
        "AST",
        "target_pts_over",
        "target_trb_over",
        "target_ast_over",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    You can tweak hyperparameters here if you want to experiment later.
    """
    model = RandomForestClassifier(
        n_estimators=300,      # number of trees
        max_depth=None,       # let trees grow until pure / min_samples constraints
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,            # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str) -> None:
    """Save a trained model to the models/ directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"✅ Saved model {name} to {path}")


def main():
    df = load_features()
    X_train, X_test, y_train, y_test = train_test_split_xy(df)

    # Save the split for evaluation later
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    # Train Random Forest
    rf = train_random_forest(X_train, y_train)

    # Name includes target for clarity
    model_name = f"random_forest_{TARGET_COL}"
    save_model(rf, model_name)


if __name__ == "__main__":
    main()
