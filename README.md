# CS3346-NBAPlayerPerfomance

# NBA Player Performance Prediction Model

## Overview

This project builds a **machine learning system** that predicts whether an NBA player will exceed a specific statistical threshold in an upcoming game (e.g., scoring more than 22.5 points).  

These thresholds are commonly used in proposition contexts, but **this project focuses purely on statistical forecasting and model evaluation**, not betting or wagering.

The goals of this project are:

- Collect and preprocess NBA player game data
- Engineer meaningful performance and context features
- Train machine learning models to predict game-level outcomes
- Evaluate prediction accuracy and probability calibration
- Visualize and interpret model performance

## Features

- Game-level player performance dataset
- Rolling statistical features (last 5 / last 10 games)
- Opponent strength metrics
- Rest and schedule factors
- Multiple ML models:
  - Baseline rolling average
  - Logistic Regression
  - Random Forest / Gradient Boosting
  - (Optional) Neural Network
- Evaluation metrics:
  - Accuracy
  - ROC-AUC
  - Brier Score
  - Calibration curves
- Model insights and feature importance

## How to Run and Train the Model

This project processes NBA game data, builds features, trains a Random Forest classifier, and evaluates its performance.  
Follow the steps below to run the full pipeline.

---

### 1. Install Requirements

If you have `requirements.txt`:

```bash
pip install -r requirements.txt

```

### Run the Pipline 

Run these comands in succession 

python -m src.data_loader

python -m src.feature_engineering

python -m src.models

python -m src.evaluation

---

## Model Improvement Roadmap

Based on analysis of the current implementation, here are recommended improvements organized by priority:

### Quick Wins (1-2 Hours)

These 5 improvements offer the highest value with minimal implementation time:

1. **Time-Based Train/Test Split** (20 min)
   - Fix critical data leakage by using chronological split instead of random
   - Ensures model can't "cheat" by learning from future games
   - Files: `src/models.py`, `src/evaluation.py`

2. **Home/Away Encoding** (15 min)
   - Add `is_home_game` binary feature
   - Players typically perform ~5% better at home
   - Files: `src/feature_engineering.py`

3. **Back-to-Back Game Indicator** (10 min)
   - Add `is_back_to_back` flag for games with 1 day rest
   - Captures fatigue effects
   - Files: `src/feature_engineering.py`

4. **Feature Importance Analysis** (15 min)
   - Plot which features drive predictions
   - Validates model behavior and guides future engineering
   - Files: `src/evaluation.py`

5. **Basic Hyperparameter Tuning** (30-40 min)
   - GridSearch on key Random Forest parameters
   - Expected 2-5% accuracy improvement
   - Files: `src/models.py`

**Expected Result**: 3-5 percentage point accuracy improvement

### Medium Priority Improvements

6. **Opponent Strength Features**
   - Add opponent defensive ratings, points allowed, pace
   - High impact: playing vs strong/weak defense matters
   - Files: `src/feature_engineering.py`, `src/data_loader.py`

7. **Alternative Model Architectures**
   - Implement XGBoost, LightGBM, Logistic Regression
   - Create ensemble combining multiple models
   - Files: `src/models.py`, `src/evaluation.py`, `requirements.txt`

8. **Time-Series Cross-Validation**
   - Use rolling window validation for robust performance estimates
   - Detects if model degrades over time
   - Files: `src/evaluation.py`, `src/models.py`

### Advanced Improvements

9. **Improve Missing Data Handling**
   - Explicit imputation for early-season games
   - Use player/position-specific priors
   - Files: `src/feature_engineering.py`, `src/data_loader.py`

10. **Handle Class Imbalance**
    - Add `class_weight='balanced'` or SMOTE
    - Optimize decision threshold
    - Files: `src/models.py`, `src/evaluation.py`

11. **Advanced Rolling Features**
    - Add variance, trends, weighted averages
    - Efficiency metrics (PTS/MP, TRB/MP)
    - Files: `src/feature_engineering.py`

12. **Probability Calibration**
    - Apply isotonic/Platt scaling
    - Better probability estimates for decision-making
    - Files: `src/evaluation.py`, `src/models.py`

### Expected Performance by Tier

- **Current baseline**: ~55-60% accuracy
- **After Quick Wins**: ~62-67% accuracy
- **After Medium Priority**: ~65-70% accuracy
- **After Advanced**: ~67-72% accuracy

For detailed implementation guidance, see the full plan at `.claude/plans/effervescent-finding-reef.md`

