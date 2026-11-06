"""
Model Training and Evaluation Pipeline
Reproduces the best-performing model identified in notebooks/deliveryeta.ipynb:
- Tuned Random Forest Regressor (300 estimators, max_depth=15, min_samples_split=2, min_samples_leaf=2)
- Saves model.pkl and scaler.pkl to models/
- Generates performance visualizations and metrics summary to docs/results/
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.preprocessing import build_preprocessor


def run_pipeline():
    print("=" * 60)
    print("DeliveryETA: Training & Evaluation Pipeline")
    print("=" * 60)

    data_path = project_root / 'data' / 'processed' / 'cleaned_dataset.csv'
    models_dir = project_root / 'models'
    results_dir = project_root / 'docs' / 'results'

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    y = df['Time_taken(min)']
    X = df.drop(columns=['Time_taken(min)'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")

    print("\nFitting ColumnTransformer preprocessor on training data...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Preprocessed feature dimensions: {X_train_processed.shape[1]}")

    # Benchmark all models compared in the project notebook
    print("\nTraining and evaluating baseline models...")
    models_to_compare = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Random Forest (Default)': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Random Forest (Tuned)': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=None,
            random_state=42,
            n_jobs=-1
        )
    }

    benchmark_results = []
    trained_models = {}

    for name, model_instance in models_to_compare.items():
        print(f"  Training {name}...")
        model_instance.fit(X_train_processed, y_train)
        y_test_pred = model_instance.predict(X_test_processed)
        y_train_pred = model_instance.predict(X_train_processed)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        metrics = {
            'model': name,
            'test_mae': round(float(test_mae), 3),
            'test_mse': round(float(test_mse), 3),
            'test_rmse': round(float(test_rmse), 3),
            'test_r2': round(float(test_r2), 4),
            'train_mae': round(float(train_mae), 3),
            'train_rmse': round(float(train_rmse), 3),
            'train_r2': round(float(train_r2), 4)
        }
        benchmark_results.append(metrics)
        trained_models[name] = model_instance
        print(f"    -> Test MAE: {test_mae:.2f} min | RMSE: {test_rmse:.2f} min | R2: {test_r2:.3f}")

    best_model = trained_models['Random Forest (Tuned)']

    # Save model and preprocessor artifacts
    print(f"\nSaving production model and preprocessor to {models_dir}...")
    joblib.dump(best_model, models_dir / 'model.pkl')
    joblib.dump(preprocessor, models_dir / 'scaler.pkl')
    print("  Saved model.pkl and scaler.pkl successfully.")

    # Save metrics JSON
    metrics_file = results_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Saved metrics summary to {metrics_file}")

    # Generate Visualizations
    print("\nGenerating performance evaluation charts...")

    # 1. Model comparison chart
    plt.figure(figsize=(10, 5), dpi=300)
    models_names = [m['model'] for m in benchmark_results]
    r2_scores = [m['test_r2'] for m in benchmark_results]
    maes = [m['test_mae'] for m in benchmark_results]

    x = np.arange(len(models_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=300)
    color = '#1f77b4'
    ax1.set_xlabel('Model Architecture', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Mean Absolute Error (min) [Lower is better]', color=color, fontweight='bold')
    bars1 = ax1.bar(x - width/2, maes, width, label='Test MAE (min)', color='#3b82f6', edgecolor='#1d4ed8')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(maes) * 1.25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}m', ha='center', va='bottom', fontsize=9)

    ax2 = ax1.twinx()
    color = '#059669'
    ax2.set_ylabel('R² Score [Higher is better]', color=color, fontweight='bold')
    bars2 = ax2.bar(x + width/2, r2_scores, width, label='Test R²', color='#10b981', edgecolor='#047857')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(x, models_names, rotation=15, ha='right')
    plt.title('Benchmark: Regression Models for Delivery Time Prediction', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison.png')
    plt.close()
    print("  Generated model_comparison.png")

    # 2. Actual vs Predicted for Tuned Random Forest
    y_pred_tuned = best_model.predict(X_test_processed)
    plt.figure(figsize=(7, 7), dpi=300)
    plt.scatter(y_test, y_pred_tuned, alpha=0.25, color='#2563eb', edgecolors='none', s=20)
    min_val = min(y_test.min(), y_pred_tuned.min())
    max_val = max(y_test.max(), y_pred_tuned.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction ($y = \\hat{y}$)')
    plt.xlabel('Actual Delivery Time (minutes)', fontweight='bold')
    plt.ylabel('Predicted Delivery Time (minutes)', fontweight='bold')
    plt.title('Tuned Random Forest: Actual vs. Predicted Delivery Time', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    r2_val = r2_score(y_test, y_pred_tuned)
    mae_val = mean_absolute_error(y_test, y_pred_tuned)
    rmse_val = root_mean_squared_error(y_test, y_pred_tuned)
    plt.text(0.05, 0.85, f'Test $R^2$: {r2_val:.3f}\nTest MAE: {mae_val:.2f} min\nTest RMSE: {rmse_val:.2f} min',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cbd5e1'))
    plt.tight_layout()
    plt.savefig(results_dir / 'actual_vs_predicted.png')
    plt.close()
    print("  Generated actual_vs_predicted.png")

    # 3. Residual Distribution Plot
    residuals = y_test - y_pred_tuned
    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(residuals, bins=50, color='#6366f1', edgecolor='black', alpha=0.7, density=True)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.axvline(residuals.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'Mean Error: {residuals.mean():.2f} min')
    plt.xlabel('Prediction Residual (Actual - Predicted, minutes)', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('Distribution of Prediction Residuals (Tuned Random Forest)', fontsize=12, fontweight='bold', pad=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / 'residual_distribution.png')
    plt.close()
    print("  Generated residual_distribution.png")

    # 4. Feature Importance
    try:
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [f.replace('num__', '').replace('cat__', '').replace('bin__', '') for f in feature_names]
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(10, 6), dpi=300)
        plt.barh(range(len(indices)), importances[indices][::-1], color='#0ea5e9', edgecolor='#0284c7')
        plt.yticks(range(len(indices)), [clean_names[i] for i in indices][::-1])
        plt.xlabel('Feature Importance (Gini Impurity Reduction)', fontweight='bold')
        plt.title('Top 15 Most Important Features for Delivery ETA Prediction', fontsize=12, fontweight='bold', pad=12)
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(results_dir / 'feature_importance.png')
        plt.close()
        print("  Generated feature_importance.png")
    except Exception as e:
        print(f"  Note: Feature importance chart skipped: {e}")

    print("\nTraining and evaluation pipeline finished successfully!")
    print("=" * 60)


if __name__ == '__main__':
    run_pipeline()
