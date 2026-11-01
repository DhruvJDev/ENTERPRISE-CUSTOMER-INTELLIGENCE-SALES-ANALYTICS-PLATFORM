"""
regression_module.py
Linear Regression pipeline on the unified Customer Sales & Feedback dataset.
Uses:
- math, numpy, pandas
- matplotlib.pyplot, seaborn
- scipy.stats
- sklearn.model_selection.train_test_split
- sklearn.linear_model.LinearRegression
- sklearn.metrics.mean_squared_error
- sklearn.preprocessing.LabelEncoder, StandardScaler
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import display_table, compute_statistics, print_section_header

def run_regression_pipeline(data_path="data/customer_sales_analytics.csv", output_dir="output_charts"):
    os.makedirs(output_dir, exist_ok=True)
    print_section_header("1. Linear Regression: Predicting Annual Customer Spend")

    # Step 1: Load and display data table
    df = pd.read_csv(data_path)
    display_table(
        df[['Customer_ID', 'Age', 'Membership_Tier', 'Annual_Income_kUSD', 'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD', 'Total_Annual_Spend_USD']],
        title="Unified Customer Dataset: Sales & Behavioral Features (Sample)"
    )

    # Step 2: Categorical Preprocessing with LabelEncoder
    label_encoder = LabelEncoder()
    df['Membership_Tier_Code'] = label_encoder.fit_transform(df['Membership_Tier'])
    
    tier_mapping = pd.DataFrame({
        'Original_Tier': label_encoder.classes_,
        'Encoded_Code': label_encoder.transform(label_encoder.classes_)
    })
    display_table(tier_mapping, title="LabelEncoder Mapping for Membership Tier", max_rows=5)

    # Step 3: Statistical Analysis using Math and SciPy
    stat_results = compute_statistics(df['Annual_Income_kUSD'].values, df['Total_Annual_Spend_USD'].values)
    print(f"• SciPy Pearson Correlation (Income vs Spend): r = {stat_results['pearson_r']:.4f} (p-value: {stat_results['p_value']:.2e})")
    print(f"• Math-calculated Income Mean: ${stat_results['mean_x']:.2f}k, StdDev: ${stat_results['std_x']:.2f}k")
    print(f"• Average Annual Customer Spend: ${stat_results['mean_y']:.2f} | Outliers Detected: {stat_results['outlier_count_y']}\n")

    # Step 4: Correlation Matrix & Heatmap
    num_cols = [
        'Age', 'Membership_Tier_Code', 'Annual_Income_kUSD',
        'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD',
        'App_Weekly_Hours', 'Discount_Usage_Pct', 'Total_Annual_Spend_USD'
    ]
    corr_matrix = df[num_cols].corr()
    display_table(corr_matrix.round(3), title="Feature Correlation Matrix (Tabular)", max_rows=8)

    # Chart: Correlation Heatmap
    plt.figure(figsize=(9.5, 7.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Customer Features & Spend Correlation Heatmap", fontsize=14, weight='bold')
    heatmap_path = os.path.join(output_dir, "02_correlation_matrix.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Correlation Heatmap Chart to: {heatmap_path}")

    # Step 5: Train-Test Split and Model Fitting
    feature_cols = [
        'Age', 'Membership_Tier_Code', 'Annual_Income_kUSD',
        'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD',
        'App_Weekly_Hours', 'Discount_Usage_Pct'
    ]
    X = df[feature_cols]
    y = df['Total_Annual_Spend_USD']

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    # Using Python math module for RMSE
    rmse = math.sqrt(mse)
    r2_score = model.score(X_test, y_test)

    # Tabular comparison of Actual vs Predicted
    comparison_df = pd.DataFrame({
        'Actual_Spend ($)': y_test.values.round(2),
        'Predicted_Spend ($)': y_pred.round(2),
        'Residual_Error ($)': (y_test.values - y_pred).round(2),
        'Abs_Pct_Error (%)': (np.abs(y_test.values - y_pred) / y_test.values * 100).round(2)
    })
    display_table(comparison_df, title="Regression Test Set: Actual vs Predicted Customer Spend", max_rows=7)

    # Metrics summary table
    metrics_summary = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R² Score (Coefficient of Determination)'],
        'Value': [f"{mse:.2f}", f"${rmse:.2f}", f"{r2_score:.4f}"]
    })
    display_table(metrics_summary, title="Linear Regression Performance Metrics", max_rows=5)

    # Visualizations: Fit & Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot A: Actual vs Predicted scatter
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='royalblue', alpha=0.8, edgecolor='black', s=50)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction (y = x)")
    axes[0].set_title(f"Actual vs Predicted Annual Spend\nR² = {r2_score:.3f}, RMSE = ${rmse:.2f}", fontsize=12, weight='bold')
    axes[0].set_xlabel("Actual Annual Spend ($)")
    axes[0].set_ylabel("Predicted Annual Spend ($)")
    axes[0].legend()

    # Subplot B: Residuals Distribution
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='teal', bins=18)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=12, weight='bold')
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Frequency")

    chart_path = os.path.join(output_dir, "01_regression_analysis.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Regression Analysis Chart to: {chart_path}")

    return {
        "model": model,
        "scaler": scaler,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score
    }

if __name__ == "__main__":
    run_regression_pipeline()
