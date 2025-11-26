"""
feature_engineering.py

Creates model-ready features and target labels from cleaned game logs.

Targets:
- target_pts_over: 1 if PTS > POINTS_THRESHOLD else 0
- target_trb_over: 1 if TRB > REBOUNDS_THRESHOLD else 0
- target_ast_over: 1 if AST > ASSISTS_THRESHOLD else 0

Outputs:
- data/processed/features.csv
"""

import os
import pandas as pd

CLEAN_DATA_PATH = "data/processed/clean_games.csv"
FEATURES_PATH = "data/processed/features.csv"

# Thresholds (change to whatever lines you care about)
POINTS_THRESHOLD = 22.5
REBOUNDS_THRESHOLD = 8.5
ASSISTS_THRESHOLD = 5.5


def load_clean_data(path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data not found at: {path}")
    return pd.read_csv(path, parse_dates=["game_date"])


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling stats per player (only past games used).

    For each player:
    - last_3_pts_avg, last_5_pts_avg
    - last_3_trb_avg, last_5_trb_avg
    - last_3_ast_avg, last_5_ast_avg
    - last_5_mp_avg
    - season_pts_avg (up to that game)
    """
    df = df.sort_values(["Player", "game_date"]).copy()

    def _add_for_player(group: pd.DataFrame) -> pd.DataFrame:
        for stat in ["PTS", "TRB", "AST"]:
            group[f"last_3_{stat.lower()}_avg"] = (
                group[stat].shift(1).rolling(window=3, min_periods=1).mean()
            )
            group[f"last_5_{stat.lower()}_avg"] = (
                group[stat].shift(1).rolling(window=5, min_periods=1).mean()
            )

        group["last_5_mp_avg"] = (
            group["MP"].shift(1).rolling(window=5, min_periods=1).mean()
        )

        group["season_pts_avg"] = (
            group["PTS"].shift(1).expanding(min_periods=1).mean()
        )

        return group

    df = df.groupby("Player", group_keys=False).apply(_add_for_player)

    return df


def add_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game context features:

    - days_rest: days since player's last game
    - is_back_to_back: 1 if game is played with only 1 day rest (fatigue indicator)
    - is_home_game: 1 if playing at home, 0 if away (inferred from opponent format)
    """
    df = df.sort_values(["Player", "game_date"]).copy()

    def _add_rest(group: pd.DataFrame) -> pd.DataFrame:
        group["prev_game_date"] = group["game_date"].shift(1)
        group["days_rest"] = (group["game_date"] - group["prev_game_date"]).dt.days
        group["days_rest"] = group["days_rest"].fillna(3)  # neutral default
        return group

    df = df.groupby("Player", group_keys=False).apply(_add_rest)
    df = df.drop(columns=["prev_game_date"])

    # Add back-to-back indicator (games with exactly 1 day rest)
    df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)

    # Add home/away indicator
    # In NBA game logs, away games typically have "@" in opponent or team designation
    # If Opp contains "@", it's an away game; otherwise home
    # This is a heuristic - adjust based on actual data format
    if df["Opp"].dtype == object:
        df["is_home_game"] = (~df["Opp"].str.contains("@", na=False)).astype(int)
    else:
        # Fallback: assume 50/50 distribution if format unclear
        # Better: check actual data format and adjust
        df["is_home_game"] = 1  # Default to home if unclear

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary over/under-style targets for PTS, TRB, AST.
    """
    df["target_pts_over"] = (df["PTS"] > POINTS_THRESHOLD).astype(int)
    df["target_trb_over"] = (df["TRB"] > REBOUNDS_THRESHOLD).astype(int)
    df["target_ast_over"] = (df["AST"] > ASSISTS_THRESHOLD).astype(int)
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns needed for modeling + targets.
    """
    feature_cols = [
        "Player",
        "Tm",
        "Opp",
        "game_date",
        "MP",
        "PTS",
        "TRB",
        "AST",
        "last_3_pts_avg",
        "last_5_pts_avg",
        "last_3_trb_avg",
        "last_5_trb_avg",
        "last_3_ast_avg",
        "last_5_ast_avg",
        "last_5_mp_avg",
        "season_pts_avg",
        "days_rest",
        "is_back_to_back",
        "is_home_game",
        "target_pts_over",
        "target_trb_over",
        "target_ast_over",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return df[feature_cols].copy()


def save_features(df: pd.DataFrame, path: str = FEATURES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Features saved to {path}")


def main():
    df = load_clean_data()
    df = add_rolling_features(df)
    df = add_game_context_features(df)
    df = add_targets(df)
    df_features = select_model_features(df)
    save_features(df_features)


if __name__ == "__main__":
    main()
