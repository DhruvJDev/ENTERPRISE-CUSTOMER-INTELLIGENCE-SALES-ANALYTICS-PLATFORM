"""
classification_module.py
Supervised Multi-Model Classification Pipeline on the unified Customer dataset.
Predicts whether a customer is a High-Value VIP Customer ('High_Value_Customer': 0 or 1).
Uses:
- numpy, pandas, matplotlib.pyplot, seaborn
- sklearn.model_selection.train_test_split
- sklearn.preprocessing.StandardScaler, LabelEncoder
- sklearn.linear_model.LogisticRegression
- sklearn.tree.DecisionTreeClassifier
- sklearn.neighbors.KNeighborsClassifier
- sklearn.naive_bayes.GaussianNB
- sklearn.svm.SVC
- sklearn.metrics.confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
- (Optional) sklearn.datasets.load_breast_cancer for lab syllabus verification
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

from src.utils import display_table, print_section_header

def run_classification_pipeline(data_path="data/customer_sales_analytics.csv", output_dir="output_charts"):
    os.makedirs(output_dir, exist_ok=True)
    print_section_header("2. Supervised Classification: Predicting High-Value VIP Customers")

    # Step 1: Load and tabular inspect the dataset
    df = pd.read_csv(data_path)
    
    # Preprocess categorical tier
    if 'Membership_Tier_Code' not in df.columns:
        le = LabelEncoder()
        df['Membership_Tier_Code'] = le.fit_transform(df['Membership_Tier'])

    df['Customer_Status'] = df['High_Value_Customer'].map({0: 'Standard Customer', 1: 'VIP High-Value'})
    
    display_table(
        df[['Customer_ID', 'Membership_Tier', 'Annual_Income_kUSD', 'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD', 'Customer_Status']],
        title="Customer Features & VIP Target Status (Tabular Preview)"
    )

    # Class balance table
    balance_df = df['Customer_Status'].value_counts().reset_index()
    balance_df.columns = ['Status Class', 'Customer Count']
    balance_df['Percentage (%)'] = (balance_df['Customer Count'] / len(df) * 100).round(2)
    display_table(balance_df, title="Target Class Distribution (High_Value_Customer)", max_rows=5)

    # Step 2: Feature Matrix and Train-Test Split
    feature_cols = [
        'Age', 'Membership_Tier_Code', 'Annual_Income_kUSD',
        'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD',
        'App_Weekly_Hours', 'Discount_Usage_Pct'
    ]
    X = df[feature_cols]
    y = df['High_Value_Customer']
    target_names = ['Standard (0)', 'VIP (1)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Step 3: Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Model Dictionary with all 5 algorithms
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine (SVC)": SVC(kernel='rbf', random_state=42)
    }

    # Step 5: Benchmark Loop
    evaluation_records = []
    confusion_matrices = {}

    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        confusion_matrices[name] = cm
        evaluation_records.append({
            "Model Name": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    # Benchmark Summary Table
    benchmark_df = pd.DataFrame(evaluation_records)
    benchmark_df_formatted = benchmark_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        benchmark_df_formatted[col] = (benchmark_df_formatted[col] * 100).round(2).astype(str) + "%"
    display_table(benchmark_df_formatted, title="Classification Benchmark: 5-Model Performance Comparison", max_rows=10)

    # Top Model Identification
    best_row = benchmark_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    print(f"★ Top Performing Classifier: {best_row['Model Name']} "
          f"(F1-Score: {best_row['F1-Score']*100:.2f}%, Accuracy: {best_row['Accuracy']*100:.2f}%)\n")

    top_clf = models[best_row['Model Name']]
    top_pred = top_clf.predict(X_test_scaled)
    print("--- Detailed Classification Report for Best Model ---")
    print(classification_report(y_test, top_pred, target_names=target_names))

    # Step 6: Visualizations
    # Chart 1: Grouped Bar Chart
    melted_df = benchmark_df.melt(id_vars="Model Name", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_df, x="Model Name", y="Score", hue="Metric", palette="mako")
    plt.title("Comparative Performance Across 5 Classification Algorithms (VIP Prediction)", fontsize=14, weight='bold')
    plt.xlabel("Classifier Model", fontsize=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(0.70, 1.02)
    plt.legend(loc='lower right', frameon=True)
    plt.xticks(rotation=15)
    
    chart1_path = os.path.join(output_dir, "03_classification_metrics_comparison.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Classification Comparison Chart to: {chart1_path}")

    # Chart 2: 2x3 Grid of Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Standard', 'VIP'],
                    yticklabels=['Standard', 'VIP'])
        acc_val = benchmark_df.loc[benchmark_df['Model Name']==name, 'Accuracy'].values[0]
        ax.set_title(f"{name}\nAccuracy: {acc_val*100:.1f}%", fontsize=11, weight='bold')
        ax.set_xlabel("Predicted Status")
        ax.set_ylabel("Actual Status")

    # Hide 6th empty subplot
    axes[5].axis('off')

    chart2_path = os.path.join(output_dir, "04_confusion_matrices_grid.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Confusion Matrices Grid to: {chart2_path}")

    return {
        "models": models,
        "scaler": scaler,
        "benchmark_df": benchmark_df,
        "confusion_matrices": confusion_matrices
    }

if __name__ == "__main__":
    run_classification_pipeline()
