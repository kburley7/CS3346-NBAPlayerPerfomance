"""
Visualize model predictions vs actual results

Usage:
    python visualize_model.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and test data
model = joblib.load('models/random_forest_target_pts.joblib')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Make predictions
y_pred = model.predict(X_test)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Predicted vs Actual scatter plot
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.3, s=20)
ax1.plot([0, 50], [0, 50], 'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Points', fontsize=12)
ax1.set_ylabel('Predicted Points', fontsize=12)
ax1.set_title('Predicted vs Actual Points', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Error distribution
ax2 = axes[0, 1]
errors = y_pred - y_test
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect (0 error)')
ax2.set_xlabel('Prediction Error (points)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Error by actual points scored
ax3 = axes[1, 0]
ax3.scatter(y_test, errors, alpha=0.3, s=20)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Actual Points', fontsize=12)
ax3.set_ylabel('Prediction Error', fontsize=12)
ax3.set_title('Prediction Error by Point Total', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Model accuracy breakdown
ax4 = axes[1, 1]
within_3 = np.mean(np.abs(errors) < 3) * 100
within_5 = np.mean(np.abs(errors) < 5) * 100
within_10 = np.mean(np.abs(errors) < 10) * 100
beyond_10 = 100 - within_10

categories = ['Within\n3 pts', 'Within\n5 pts', 'Within\n10 pts', 'Beyond\n10 pts']
values = [within_3, within_5 - within_3, within_10 - within_5, beyond_10]
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

ax4.bar(categories, values, color=colors, edgecolor='black')
ax4.set_ylabel('Percentage of Predictions', fontsize=12)
ax4.set_title('Prediction Accuracy Breakdown', fontsize=14, fontweight='bold')
ax4.set_ylim(0, 70)

# Add value labels on bars
for i, (cat, val) in enumerate(zip(categories, values)):
    ax4.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('models/model_performance_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: models/model_performance_visualization.png")

# Print summary statistics
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"\nTotal predictions analyzed: {len(y_test):,}")
print(f"\nAverage actual points: {np.mean(y_test):.2f}")
print(f"Average predicted points: {np.mean(y_pred):.2f}")
print(f"\nMean Absolute Error: {np.mean(np.abs(errors)):.2f} points")
print(f"Standard deviation of errors: {np.std(errors):.2f} points")
print(f"\nAccuracy breakdown:")
print(f"  Within 3 points:  {within_3:.1f}%")
print(f"  Within 5 points:  {within_5:.1f}%")
print(f"  Within 10 points: {within_10:.1f}%")

# Show best and worst predictions
print(f"\n{'='*60}")
print("BEST PREDICTIONS (Closest to actual)")
print(f"{'='*60}")
best_indices = np.argsort(np.abs(errors))[:5]
for i, idx in enumerate(best_indices, 1):
    print(f"{i}. Predicted: {y_pred[idx]:.1f} | Actual: {y_test[idx]:.1f} | Error: {errors[idx]:+.1f}")

print(f"\n{'='*60}")
print("WORST PREDICTIONS (Furthest from actual)")
print(f"{'='*60}")
worst_indices = np.argsort(np.abs(errors))[-5:][::-1]
for i, idx in enumerate(worst_indices, 1):
    print(f"{i}. Predicted: {y_pred[idx]:.1f} | Actual: {y_test[idx]:.1f} | Error: {errors[idx]:+.1f}")

print("\n" + "="*60)
