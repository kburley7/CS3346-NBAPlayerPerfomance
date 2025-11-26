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

## Repository Structure

