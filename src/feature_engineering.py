"""
feature_engineering.py

Takes cleaned NBA game logs and builds:
- Rolling performance features (PTS, TRB, AST, MP)
- Simple schedule features (days since last game, back-to-back flag)
- Regression targets for actual stats:
    - target_pts: points scored in the game
    - target_trb: rebounds in the game
    - target_ast: assists in the game

Critically, we DROP all same-game box score stats from the feature set
(FG, FGA, 3P, 3PA, FT, FTA, FG%, 3P%, FT%, etc.) to prevent data leakage.

Outputs:
- data/processed/features.csv
"""

import os
from typing import List

import pandas as pd

CLEAN_DATA_PATH = "data/processed/clean_games.csv"
FEATURES_PATH = "data/processed/features.csv"


def load_clean_data(path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load cleaned game logs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data file not found at: {path}")
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["Player"],
    stat_cols: List[str] = ["PTS", "TRB", "AST", "MP"],
    windows: List[int] = [5, 10],
) -> pd.DataFrame:
    """
    For each player, compute rolling averages for the given stats over the last N games.

    Important:
    - We shift by 1 game so that the current game's features only use PAST games.
    """
    df = df.sort_values(group_cols + ["game_date"]).copy()
    grouped = df.groupby(group_cols, group_keys=False)

    for stat in stat_cols:
        stat_lower = stat.lower()
        # Previous game's raw value (safe: comes from PRIOR game)
        df[f"{stat_lower}_prev"] = grouped[stat].shift(1)

        for w in windows:
            df[f"{stat_lower}_avg_{w}"] = (
                grouped[stat]
                .shift(1)  # exclude current game
                .rolling(window=w, min_periods=1)
                .mean()
            )

    return df


def add_schedule_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["Player"],
) -> pd.DataFrame:
    """
    Add simple schedule-based features:
    - days_since_last_game: difference in days between this game and previous game
    - is_back_to_back: 1 if days_since_last_game == 1, else 0
    """
    df = df.sort_values(group_cols + ["game_date"]).copy()
    grouped = df.groupby(group_cols, group_keys=False)

    df["days_since_last_game"] = grouped["game_date"].diff().dt.days
    df["is_back_to_back"] = (df["days_since_last_game"] == 1).astype(int)

    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regression targets:
    - target_pts: actual points scored
    - target_trb: actual rebounds
    - target_ast: actual assists
    """
    df["target_pts"] = df["PTS"].astype(float)
    df["target_trb"] = df["TRB"].astype(float)
    df["target_ast"] = df["AST"].astype(float)
    return df


def drop_leaky_box_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop same-game box score stats that directly encode the outcome of the game.
    These MUST NOT be used as features, or the model will "cheat".

    We already used PTS/TRB/AST/MP to build rolling and target columns,
    so it's safe to drop them here.
    """
    leaky_cols = [
        # Core box score
        "PTS", "TRB", "AST", "MP",
        "FG", "FGA", "FG%",
        "3P", "3PA", "3P%",
        "FT", "FTA", "FT%",
        # Other per-game stats (if present)
        "ORB", "DRB", "STL", "BLK", "TOV", "PF",
        "GmSc", "+/-",
    ]
    return df.drop(columns=leaky_cols, errors="ignore")


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that do not have enough history (e.g., first game for a player),
    and keep only rows that have all feature columns and targets available.

    We:
    - drop leaky box-score columns
    - keep Player/Tm/Opp/game_date for reference (they're excluded in models.py)
    - use only numeric, non-target columns as features
    """
    # First, remove leaky columns entirely
    df = drop_leaky_box_score_columns(df)

    target_cols = ["target_pts", "target_trb", "target_ast"]

    # Identify numeric feature columns (non-target)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in target_cols]

    # Drop rows where any feature or target is NaN (early games, missing data)
    df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    return df


def save_features(df: pd.DataFrame, path: str = FEATURES_PATH) -> None:
    """Save engineered features + targets to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Features saved to {path}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")


def main():
    # 1. Load cleaned data
    df = load_clean_data()

    # 2. Add rolling performance features (history-based)
    df = add_rolling_features(df)

    # 3. Add schedule features
    df = add_schedule_features(df)

    # 4. Add regression targets
    df = add_target_columns(df)

    # 5. Remove leaky columns & rows without full history
    df = finalize_features(df)

    # 6. Save
    save_features(df)


if __name__ == "__main__":
    main()
