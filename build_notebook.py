"""
build_notebook.py
Generates the updated, comprehensive Jupyter Notebook 'ml_mini_project.ipynb'
based entirely on the SINGLE UNIFIED DATASET: 'data/customer_sales_analytics.csv'.
"""
import json
import os

def create_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    md("""
# Enterprise Customer Intelligence & Sales Analytics Platform
### A Unified Machine Learning Mini Project on a Single E-Commerce Dataset

This project implements an end-to-end business intelligence machine learning system using a **single source of truth dataset**: `data/customer_sales_analytics.csv`.

Across this single dataset, we solve four core business problems:
1. **Regression & Statistical Analysis**: Predict customer annual spend (`Total_Annual_Spend_USD`) using `LinearRegression`, evaluate with `mean_squared_error` and `math.sqrt(mse)` (RMSE), and compute SciPy correlation metrics ($r$, $p$-value).
2. **Supervised Multi-Model Classification**: Benchmark 5 classifiers (`LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `GaussianNB`, `SVC`) to predict `High_Value_Customer` (VIP vs. Standard) using `StandardScaler`, `confusion_matrix`, `precision_score`, `recall_score`, and `f1_score`.
3. **Unsupervised Clustering**: Segment customers into behavioral market personas using `KMeans` with the Elbow Method and 2D centroid visualization.
4. **Natural Language Processing & Model Persistence**: Automatically classify incoming text reviews (`Customer_Review`) into Positive vs. Negative sentiment using `TfidfVectorizer` and `MultinomialNB`, and serialize the complete pipeline with `joblib` for real-time inference.
""")

    # Cell 1: Environment & Imports
    md("## 1. Setup & Library Imports")
    code("""import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# Scikit-Learn Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Scikit-Learn Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Feature Extraction & Metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

# Optional dataset loader for academic verification
from sklearn.datasets import load_breast_cancer

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.autolayout': True, 'font.size': 11})
print("✓ All libraries successfully loaded!")
""")

    # Cell 2: Load Single Dataset & Table Preview
    md("## 2. Single Dataset Ingestion & Tabular Inspection")
    code("""# Load the single unified dataset
df = pd.read_csv('data/customer_sales_analytics.csv')

print(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print("\\n--- Unified Dataset Preview (First 5 Rows) ---")
display(df.head(5))

print("\\n--- Dataset Summary Statistics ---")
display(df.describe().round(2))
""")

    # Cell 3: Preprocessing & Statistical Testing (Math & SciPy)
    md("## 3. Categorical Encoding (LabelEncoder) & Statistical Hypothesis Testing (SciPy & Math)")
    code("""# Encode Membership Tier
le = LabelEncoder()
df['Membership_Tier_Code'] = le.fit_transform(df['Membership_Tier'])

tier_mapping = pd.DataFrame({
    'Original_Tier': le.classes_,
    'Encoded_Code': le.transform(le.classes_)
})
print("--- LabelEncoder Mapping Table ---")
display(tier_mapping)

# Analytical Statistics using SciPy and Math
income = df['Annual_Income_kUSD'].values
spend = df['Total_Annual_Spend_USD'].values

pearson_r, p_val = stats.pearsonr(income, spend)
mean_inc = float(np.mean(income))
std_inc = math.sqrt(sum((x - mean_inc)**2 for x in income) / (len(income) - 1))

print(f"• SciPy Pearson Correlation (Income vs Spend): r = {pearson_r:.4f} (p-value: {p_val:.2e})")
print(f"• Math-Calculated Income Mean: ${mean_inc:.2f}k, StdDev: ${std_inc:.2f}k")
""")

    # Cell 4: Correlation Matrix & Heatmap
    md("### Correlation Matrix & Heatmap")
    code("""num_cols = [
    'Age', 'Membership_Tier_Code', 'Annual_Income_kUSD',
    'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD',
    'App_Weekly_Hours', 'Discount_Usage_Pct', 'Total_Annual_Spend_USD'
]
corr_matrix = df[num_cols].corr()

print("--- Feature Correlation Table ---")
display(corr_matrix.round(3))

plt.figure(figsize=(9.5, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Customer Feature Correlation Matrix Heatmap", fontsize=14, weight='bold')
plt.show()
""")

    # Cell 5: Linear Regression Pipeline
    md("## 4. Linear Regression: Predicting Annual Customer Spend")
    code("""feature_cols = [
    'Age', 'Membership_Tier_Code', 'Annual_Income_kUSD',
    'Purchase_Frequency_Yearly', 'Avg_Order_Value_USD',
    'App_Weekly_Hours', 'Discount_Usage_Pct'
]
X = df[feature_cols]
y = df['Total_Annual_Spend_USD']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# Train model
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

# Predict & Evaluate
y_pred = reg_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)  # Computed via math module
r2 = reg_model.score(X_test_scaled, y_test)

# Prediction comparison table
pred_df = pd.DataFrame({
    'Actual Spend ($)': y_test.values.round(2),
    'Predicted Spend ($)': y_pred.round(2),
    'Residual ($)': (y_test.values - y_pred).round(2),
    'Abs % Error': (np.abs(y_test.values - y_pred) / y_test.values * 100).round(2)
})
print("--- Linear Regression Test Set Predictions Table ---")
display(pred_df.head(7))

print(f"• Model MSE:  {mse:.2f}")
print(f"• Model RMSE: ${rmse:.2f} (calculated via math.sqrt)")
print(f"• Model R²:   {r2:.4f}")

# Visualizations: Fit Scatter & Residual Histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='royalblue', alpha=0.8, edgecolor='black', s=50)
min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label="Perfect Prediction (y = x)")
axes[0].set_title(f"Actual vs Predicted Annual Spend\\nR² = {r2:.3f}, RMSE = ${rmse:.2f}", fontsize=12, weight='bold')
axes[0].set_xlabel("Actual Annual Spend ($)")
axes[0].set_ylabel("Predicted Annual Spend ($)")
axes[0].legend()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=axes[1], color='teal', bins=18)
axes[1].axvline(0, color='red', linestyle='--', lw=2)
axes[1].set_title("Residual Error Distribution", fontsize=12, weight='bold')
axes[1].set_xlabel("Residual ($)")
plt.show()
""")

    # Cell 6: Classification Pipeline
    md("""## 5. Supervised Multi-Model Classification: Predicting VIP Customers
We benchmark 5 distinct classification algorithms on the single dataset to predict `High_Value_Customer`:
1. `LogisticRegression`
2. `DecisionTreeClassifier`
3. `KNeighborsClassifier`
4. `GaussianNB`
5. `SVC`
""")
    code("""y_clf = df['High_Value_Customer']
X_clf = df[feature_cols]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine (SVC)": SVC(kernel='rbf', random_state=42)
}

benchmark_records = []
cms = {}

for name, clf in models.items():
    clf.fit(X_train_c_scaled, y_train_c)
    preds = clf.predict(X_test_c_scaled)
    
    acc = accuracy_score(y_test_c, preds)
    prec = precision_score(y_test_c, preds)
    rec = recall_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)
    cm = confusion_matrix(y_test_c, preds)
    
    cms[name] = cm
    benchmark_records.append({
        "Model Name": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

benchmark_df = pd.DataFrame(benchmark_records)
benchmark_formatted = benchmark_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    benchmark_formatted[col] = (benchmark_formatted[col] * 100).round(2).astype(str) + "%"

print("--- 5-Classifier Performance Benchmark Table ---")
display(benchmark_formatted)
""")

    # Cell 7: Classification Visualizations
    md("### Classification Charts: Benchmark Bar Chart & Confusion Matrices")
    code("""# Plot 1: Performance Bar Chart
melted_df = benchmark_df.melt(id_vars="Model Name", var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 5.5))
sns.barplot(data=melted_df, x="Model Name", y="Score", hue="Metric", palette="mako")
plt.title("Comparative Performance Across 5 Classifiers (VIP Prediction)", fontsize=14, weight='bold')
plt.ylabel("Score (0.0 to 1.0)")
plt.ylim(0.70, 1.02)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.show()

# Plot 2: Confusion Matrices Grid
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (name, cm) in enumerate(cms.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Standard', 'VIP'],
                yticklabels=['Standard', 'VIP'])
    acc_val = benchmark_df.loc[benchmark_df['Model Name']==name, 'Accuracy'].values[0]
    ax.set_title(f"{name}\\nAccuracy: {acc_val*100:.1f}%", fontsize=11, weight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

axes[5].axis('off')
plt.show()
""")

    # Cell 8: Clustering Pipeline
    md("## 6. Unsupervised Learning: K-Means Customer Persona Segmentation")
    code("""X_cluster = df[['Annual_Income_kUSD', 'Total_Annual_Spend_USD']]
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Elbow Method
k_range = range(1, 9)
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster_scaled)
    inertias.append(km.inertia_)

# Fit Optimal k = 3
kmeans_opt = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans_opt.fit_predict(X_cluster_scaled)
centers = scaler_cluster.inverse_transform(kmeans_opt.cluster_centers_)

# Summary Table
persona_summary = df.groupby('Cluster_ID').agg(
    Customer_Count=('Customer_ID', 'count'),
    Avg_Income_kUSD=('Annual_Income_kUSD', 'mean'),
    Avg_Spend_USD=('Total_Annual_Spend_USD', 'mean'),
    Min_Spend_USD=('Total_Annual_Spend_USD', 'min'),
    Max_Spend_USD=('Total_Annual_Spend_USD', 'max'),
    Avg_Purchases=('Purchase_Frequency_Yearly', 'mean')
).round(2).reset_index()

print("--- K-Means Customer Personas Summary Table ---")
display(persona_summary)

# Visualizations: Elbow Curve & Clustered Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal k = 3')
axes[0].set_title("Elbow Method for Optimal k", fontsize=13, weight='bold')
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

sns.scatterplot(
    data=df, x='Annual_Income_kUSD', y='Total_Annual_Spend_USD',
    hue='Cluster_ID', palette='Set1', alpha=0.85, s=65, ax=axes[1]
)
axes[1].scatter(
    centers[:, 0], centers[:, 1],
    s=260, c='gold', marker='*', edgecolors='black', linewidths=1.5,
    label='Cluster Centroids', zorder=10
)
axes[1].set_title("Customer Market Segmentation (K-Means)", fontsize=13, weight='bold')
axes[1].set_xlabel("Annual Income ($k USD)")
axes[1].set_ylabel("Total Annual Spend ($ USD)")
axes[1].legend()
plt.show()
""")

    # Cell 9: NLP & Joblib Persistence
    md("## 7. NLP Text Analytics: Customer Sentiment Classification & Joblib Model Persistence")
    code("""# Train-test split on text reviews from the single dataset
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    df['Customer_Review'], df['Feedback_Sentiment'], test_size=0.25, random_state=42, stratify=df['Feedback_Sentiment']
)

# TF-IDF Feature Extraction
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300)
X_train_vec = tfidf.fit_transform(X_train_txt)
X_test_vec = tfidf.transform(X_test_txt)

# MultinomialNB Model
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train_vec, y_train_txt)

# Predict & Evaluate
y_pred_txt = mnb.predict(X_test_vec)
nlp_acc = accuracy_score(y_test_txt, y_pred_txt)
print(f"• MultinomialNB Sentiment Accuracy: {nlp_acc * 100:.2f}%\\n")
print("--- Classification Report ---")
print(classification_report(y_test_txt, y_pred_txt))

# Confusion Matrix Chart
plt.figure(figsize=(6, 4.5))
cm_nlp = confusion_matrix(y_test_txt, y_pred_txt, labels=['Negative', 'Positive'])
sns.heatmap(cm_nlp, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title(f"Sentiment Confusion Matrix\\nAccuracy = {nlp_acc*100:.1f}%", fontsize=12, weight='bold')
plt.xlabel("Predicted Sentiment")
plt.ylabel("True Sentiment")
plt.show()

# Model Persistence with Joblib
pipeline_dict = {'vectorizer': tfidf, 'model': mnb}
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline_dict, 'models/sentiment_pipeline.joblib')
print("✓ Pipeline saved to disk at 'models/sentiment_pipeline.joblib'")

# Reload Pipeline and Perform Live Real-Time Inferences
loaded_pipeline = joblib.load('models/sentiment_pipeline.joblib')

incoming_feedback = [
    "Incredible shopping experience! Checkout was fast and delivery arrived within 24 hours.",
    "Terrible experience. Package arrived late, broken merchandise and support was unhelpful."
]
test_vectors = loaded_pipeline['vectorizer'].transform(incoming_feedback)
live_predictions = loaded_pipeline['model'].predict(test_vectors)

live_df = pd.DataFrame({
    'Incoming Customer Review': incoming_feedback,
    'Automated Sentiment Classification': live_predictions
})
print("\\n--- Real-Time Live Inference with Saved Joblib Model ---")
display(live_df)
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("ml_mini_project.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("✓ Successfully regenerated ml_mini_project.ipynb on the single dataset!")

if __name__ == "__main__":
    create_notebook()
