"""
View model predictions for specific NBA players

Usage:
    python view_predictions.py
"""

import joblib
import numpy as np
import pandas as pd

# Load model and data
model = joblib.load('models/random_forest_target_pts.joblib')
df = pd.read_csv('data/processed/features.csv', parse_dates=['game_date'])
feature_names = np.load('data/processed/feature_names.npy', allow_pickle=True)

# Get feature columns
exclude_cols = ['Player', 'Tm', 'Opp', 'Res', 'Data', 'game_date',
                'target_pts', 'target_trb', 'target_ast']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

# Sort by date to get recent games
df_sorted = df.sort_values('game_date', ascending=False)

def show_predictions_for_player(player_name, n_games=10):
    """Show recent predictions vs actual results for a player"""
    player_data = df_sorted[df_sorted['Player'] == player_name].head(n_games)

    if len(player_data) == 0:
        print(f"Player '{player_name}' not found!")
        return

    print(f"\n{'='*80}")
    print(f"Predictions for {player_name} (Most Recent {len(player_data)} Games)")
    print(f"{'='*80}")

    for idx, row in player_data.iterrows():
        # Get features for this game
        X = row[feature_cols].values.reshape(1, -1)
        prediction = model.predict(X)[0]
        actual = row['target_pts']
        error = prediction - actual

        print(f"\nDate: {row['game_date'].strftime('%Y-%m-%d')} | {row['Tm']} vs {row['Opp']}")
        print(f"  5-game avg: {row['pts_avg_5']:.1f} pts | Minutes prev: {row['mp_prev']:.1f}")
        print(f"  PREDICTED: {prediction:.1f} pts | ACTUAL: {actual:.1f} pts | Error: {error:+.1f}")

        # Show accuracy
        accuracy = "✓ Great!" if abs(error) < 3 else "✓ Good" if abs(error) < 6 else "✗ Off"
        print(f"  {accuracy}")

def show_model_summary():
    """Show overall model behavior"""
    print("\n" + "="*80)
    print("MODEL SUMMARY - How It Works")
    print("="*80)

    print("\nTop 5 Most Important Features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:5]

    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:20s} - {importances[idx]*100:.1f}% importance")

    print("\nWhat this means:")
    print("  - The model relies HEAVILY on recent scoring (pts_avg_5)")
    print("  - Minutes played is the 2nd most important factor")
    print("  - Other stats (rebounds, assists) have smaller effects")

    # Show some statistics
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    y_pred = model.predict(X_test)

    print(f"\nOverall Test Set Performance:")
    print(f"  Average error: {np.mean(np.abs(y_pred - y_test)):.2f} points")
    print(f"  Predictions within 5 pts: {np.mean(np.abs(y_pred - y_test) < 5)*100:.1f}%")
    print(f"  Predictions within 10 pts: {np.mean(np.abs(y_pred - y_test) < 10)*100:.1f}%")

if __name__ == "__main__":
    # Show model summary first
    show_model_summary()

    # Show predictions for some star players
    players = ['LeBron James', 'Stephen Curry', 'Giannis Antetokounmpo', 'Luka Dončić']

    for player in players:
        show_predictions_for_player(player, n_games=5)

    # Interactive mode
    print("\n" + "="*80)
    print("Try it yourself! Enter a player name (or 'quit' to exit)")
    print("="*80)

    while True:
        player = input("\nPlayer name: ").strip()
        if player.lower() in ['quit', 'exit', 'q', '']:
            break
        show_predictions_for_player(player, n_games=10)
