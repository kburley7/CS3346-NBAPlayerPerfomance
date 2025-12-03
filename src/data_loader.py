"""
data_loader.py

Loads raw NBA player game data, performs basic cleaning, and saves a cleaned CSV.

Expected raw dataset columns (at least):
- Player: player name
- Tm: team abbreviation
- Opp: opponent abbreviation
- Date/Data: date of game YYYY-MM-DD
- MP: minutes played (float)
- PTS: points
- TRB: total rebounds
- AST: assists
"""

import os
import pandas as pd

RAW_DATA_PATH = "data/raw/games.csv"
CLEAN_DATA_PATH = "data/processed/clean_games.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw game logs from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop rows with missing key fields
    - Convert Date/Data -> datetime (game_date)
    - Convert MP to numeric
    - Filter out games with 0 or NaN minutes (DNP)
    - Sort by player and date
    """

    # Support either "Date" or "Data" from the raw CSV
    date_col = None
    for candidate in ["Date", "Data", "GAME_DATE"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(
            "No date column found. Expected one of: 'Date', 'Data', 'GAME_DATE'."
        )

    required_cols = [
        "Player",
        "Tm",
        "Opp",
        date_col,
        "MP",
        "PTS",
        "TRB",
        "AST",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing required values
    df = df.dropna(subset=required_cols)

    # Convert date column -> standardized 'game_date'
    df["game_date"] = pd.to_datetime(df[date_col])

    # Convert MP to numeric
    df["MP"] = pd.to_numeric(df["MP"], errors="coerce")

    # Filter out DNP (0 or NaN minutes)
    df = df[df["MP"] > 0].copy()

    # Sort by player and date
    df = df.sort_values(["Player", "game_date"]).reset_index(drop=True)

    return df


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    """Save cleaned data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Clean data saved to {path}")


def main():
    df_raw = load_raw_data()
    df_clean = basic_cleaning(df_raw)
    save_clean_data(df_clean)


if __name__ == "__main__":
    main()
