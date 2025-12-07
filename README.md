NBA Player Performance Prediction

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

python -m src.data_loader
python -m src.feature_engineering
python -m src.models
python -m src.evaluation

Model Performance
MAE: 4.78
MSE: 37.93
R²: 0.525


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

data/processed/clean_games.csv
data/processed/features.csv
data/processed/X_test.npy
data/processed/y_test.npy
data/processed/feature_names.npy
models/random_forest_target_pts.joblib

Requirements

Install dependencies:

pip install -r requirements.txt
