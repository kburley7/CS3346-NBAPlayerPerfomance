NBA Player Performance Prediction

The dataset can be found here: https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425

This project predicts how many points an NBA player will score in their next game using machine learning.
The system uses historical game logs, rolling averages, and schedule features while avoiding data leakage.

Features

 - Cleans raw NBA game data

 - Builds rolling statistical features (5-game / 10-game averages)

 - Adds schedule context (days since last game, back-to-back)

 - Trains a Random Forest regression model

 - Evaluates using MAE, MSE, and R²

 - Provides feature importance visualization

How to Run

Run each module in order:

 - python -m src.data_loader
 - python -m src.feature_engineering
 - python -m src.models
 - python -m src.evaluation

Explore Model Predictions

After training, you can explore what the model does:

View predictions for specific players:

python view_predictions.py

This shows:
 - Recent game predictions vs actual results for star players
 - Which features influenced each prediction
 - Interactive mode to query any player

Generate performance visualizations:

python visualize_model.py

This creates:
 - Predicted vs Actual scatter plot
 - Error distribution histogram
 - Accuracy breakdown charts
 - Saved to models/model_performance_visualization.png

Model Performance
 - MAE: 4.78 points (average prediction error)
 - MSE: 37.93
 - R²: 0.525 (explains 52.5% of variance)
 - 61.2% of predictions within 5 points
 - 90.0% of predictions within 10 points

These are realistic results for NBA player scoring prediction.

Key Features Used

 - pts_avg_5

 - pts_avg_10

 - mp_prev

 - mp_avg_5

 - mp_avg_10

 - Previous game stats (pts_prev, etc.)

Output Files

After running the pipeline, you will have:

 - data/processed/clean_games.csv
 - data/processed/features.csv
 - data/processed/X_test.npy
 - data/processed/y_test.npy
 - data/processed/feature_names.npy
 - models/random_forest_target_pts.joblib
 - models/random_forest_target_pts_feature_importance.png
 - models/model_performance_visualization.png (after running visualize_model.py)

Requirements

Install dependencies:

pip install -r requirements.txt
