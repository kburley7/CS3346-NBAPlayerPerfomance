"""
player_prediction_history.py

Prints (and optionally exports) a player's recent predicted vs actual scoring history.

Usage:
    python player_prediction_history.py "LeBron James" --games 10 --save reports/lebron_history.csv
"""

import argparse
import os
from textwrap import dedent

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PATH = "models/random_forest_target_pts.joblib"
FEATURES_PATH = "data/processed/features.csv"
FEATURE_NAMES_PATH = "data/processed/feature_names.npy"


def parse_args():
    parser = argparse.ArgumentParser(description="Show recent prediction history for a player.")
    parser.add_argument("player", help="Full player name as it appears in the data.")
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of most recent games to display (default: 10).",
    )
    parser.add_argument("--save", help="Optional path to save the output as CSV.")
    parser.add_argument(
        "--figure",
        help="Optional path to save the player history as an image (PNG).",
    )
    return parser.parse_args()


def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file missing: {FEATURES_PATH}")
    if not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(f"Feature names missing: {FEATURE_NAMES_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"])
    feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()
    return model, df, feature_names


def build_history(df, feature_names, model, player_name, games):
    # Numeric feature columns match saved feature names
    feature_cols = [c for c in feature_names if c in df.columns]
    player_df = (
        df[df["Player"].str.lower() == player_name.lower()]
        .sort_values("game_date", ascending=False)
        .head(games)
        .copy()
    )

    if player_df.empty:
        raise ValueError(f"No games found for player '{player_name}'.")

    X = player_df[feature_cols].values
    preds = model.predict(X)
    player_df["predicted_pts"] = preds
    player_df["error"] = player_df["predicted_pts"] - player_df["target_pts"]
    player_df["abs_error"] = player_df["error"].abs()
    player_df["within_5"] = player_df["abs_error"] < 5

    columns = [
        "game_date",
        "Tm",
        "Opp",
        "pts_avg_5",
        "mp_prev",
        "predicted_pts",
        "target_pts",
        "error",
        "abs_error",
        "within_5",
    ]

    return player_df[columns]


def save_history_figure(history, player_name, path):
    """Render the player history as a table figure."""
    fig_height = max(2.5, 0.4 * len(history) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    columns = [
        "game_date",
        "Tm",
        "Opp",
        "predicted_pts",
        "target_pts",
        "error",
        "within_5",
    ]
    headers = ["Date", "Tm", "Opp", "Predicted", "Actual", "Error", "Within ±5"]
    cell_data = []
    for _, row in history.iterrows():
        cell_data.append(
            [
                row["game_date"].strftime("%Y-%m-%d"),
                row["Tm"],
                row["Opp"],
                f"{row['predicted_pts']:.1f}",
                f"{row['target_pts']:.1f}",
                f"{row['error']:+.1f}",
                "Yes" if row["within_5"] else "No",
            ]
        )

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    ax.set_title(f"{player_name} Prediction History", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved player history figure to {path}")


def main():
    args = parse_args()
    model, df, feature_names = load_resources()
    history = build_history(df, feature_names, model, args.player, args.games)

    print("\n" + "=" * 70)
    print(f"Prediction history for {args.player} (last {len(history)} games)")
    print("=" * 70)
    print(
        dedent(
            """
            Date       | Team | Opp | Predicted | Actual | Error | Abs Error | Within ±5
            -----------------------------------------------------------------------
            """
        ).strip()
    )

    for _, row in history.iterrows():
        print(
            f"{row['game_date'].strftime('%Y-%m-%d')} | {row['Tm']:3s}   | "
            f"{row['Opp']:3s} | {row['predicted_pts']:8.1f} | {row['target_pts']:6.1f} | "
            f"{row['error']:6.1f} | {row['abs_error']:9.1f} | {str(row['within_5'])}"
        )

    if args.save:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        history.to_csv(args.save, index=False)
        print(f"\nSaved player history to {args.save}")

    if args.figure:
        save_history_figure(history, args.player, args.figure)


if __name__ == "__main__":
    main()
