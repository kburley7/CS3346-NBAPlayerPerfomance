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
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit

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
    Split data into train and test sets using TIME-BASED split.

    Uses chronological split (first 80% of games for training, last 20% for testing)
    to prevent data leakage and simulate real-world prediction scenario.

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

    # Sort by date to ensure chronological order
    df_sorted = df.sort_values("game_date").reset_index(drop=True)

    # Calculate split index for time-based split (80/20)
    split_idx = int(len(df_sorted) * (1 - TEST_SIZE))

    # Split chronologically
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    # Extract features and target
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values

    print(f"📅 Time-based split:")
    print(f"   Training: {train_df['game_date'].min()} to {train_df['game_date'].max()} ({len(train_df)} games)")
    print(f"   Testing:  {test_df['game_date'].min()} to {test_df['game_date'].max()} ({len(test_df)} games)")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, tune_hyperparameters: bool = False) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: If True, performs GridSearchCV to find best hyperparameters

    Returns:
        Trained RandomForestClassifier
    """
    if tune_hyperparameters:
        print("🔧 Tuning hyperparameters with GridSearchCV...")

        # Base model
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

        # Parameter grid - focused on key parameters for quick tuning
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }

        # Use TimeSeriesSplit for time-aware cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Grid search with ROC-AUC scoring
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV ROC-AUC: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    else:
        # Default hyperparameters (good baseline)
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
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

    # Save feature names for later analysis
    exclude_cols = ["Player", "Tm", "Opp", "game_date", "PTS", "TRB", "AST",
                    "target_pts_over", "target_trb_over", "target_ast_over"]
    feature_names = [c for c in df.columns if c not in exclude_cols]
    np.save("data/processed/feature_names.npy", np.array(feature_names))

    # Train Random Forest with optional hyperparameter tuning
    # Set tune_hyperparameters=True to enable GridSearchCV (takes longer but better results)
    TUNE_HYPERPARAMETERS = True  # Change to False for quick training with defaults
    rf = train_random_forest(X_train, y_train, tune_hyperparameters=TUNE_HYPERPARAMETERS)

    # Name includes target for clarity
    model_name = f"random_forest_{TARGET_COL}"
    save_model(rf, model_name)


if __name__ == "__main__":
    main()
