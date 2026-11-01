"""
main.py
Master execution script for the Unified Machine Learning System.
Operates on a Single Enterprise Dataset ('data/customer_sales_analytics.csv')
to solve 4 key business problems:
1. Linear Regression: Predicting Annual Customer Spend (MSE, RMSE via math.sqrt, R², SciPy Pearson r)
2. Supervised Classification: Benchmarking 5 Classifiers to predict High-Value VIP Customers
3. Unsupervised Clustering: K-Means Customer Persona Segmentation & Elbow Method
4. NLP Text Analytics: Customer Sentiment Classification with TF-IDF, MultinomialNB & Joblib persistence
"""
import os
import sys

# Configure environment before importing plotting packages
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import time

# Ensure current directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_single_dataset import generate_unified_dataset
from src.utils import setup_plot_theme, print_section_header
from src.regression_module import run_regression_pipeline
from src.classification_module import run_classification_pipeline
from src.clustering_module import run_clustering_pipeline
from src.nlp_module import run_nlp_pipeline

def main():
    start_time = time.time()
    setup_plot_theme()

    print("\n" + "=" * 78)
    print("   ENTERPRISE CUSTOMER INTELLIGENCE & SALES SYSTEM (UNIFIED ML PIPELINE)")
    print("   Single Source of Truth: 'data/customer_sales_analytics.csv'")
    print("   Libraries: Math | NumPy | Pandas | Matplotlib | Seaborn | SciPy | Scikit-Learn")
    print("=" * 78)

    # Step 0: Ensure dataset exists
    data_dir = "data"
    charts_dir = "output_charts"
    models_dir = "models"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "customer_sales_analytics.csv")
    if not os.path.exists(dataset_path):
        print("\nGenerating single unified dataset...")
        generate_unified_dataset(dataset_path, n_samples=500)

    # Pipeline 1: Linear Regression (Predicting Customer Spend)
    reg_results = run_regression_pipeline(
        data_path=dataset_path,
        output_dir=charts_dir
    )

    # Pipeline 2: Supervised Multi-Model Classification (Predicting VIP Customers)
    clf_results = run_classification_pipeline(
        data_path=dataset_path,
        output_dir=charts_dir
    )

    # Pipeline 3: K-Means Customer Persona Segmentation
    cluster_results = run_clustering_pipeline(
        data_path=dataset_path,
        output_dir=charts_dir
    )

    # Pipeline 4: NLP Text Analytics & Real-Time Joblib Inference
    nlp_results = run_nlp_pipeline(
        data_path=dataset_path,
        output_dir=charts_dir,
        model_dir=models_dir
    )

    # Final Execution Summary
    elapsed = time.time() - start_time
    print_section_header("Project Execution Summary & Generated Assets")
    print(f"✓ All 4 Machine Learning pipelines executed successfully on the unified dataset in {elapsed:.2f} seconds.")
    print("\nGenerated High-Resolution Visualizations (saved in 'output_charts/'):")
    for file in sorted(os.listdir(charts_dir)):
        if file.endswith(".png"):
            file_path = os.path.join(charts_dir, file)
            print(f"  • {file_path} ({os.path.getsize(file_path) // 1024} KB)")

    print(f"\nPersisted Model Artifacts (saved in '{models_dir}/'):")
    for file in sorted(os.listdir(models_dir)):
        file_path = os.path.join(models_dir, file)
        print(f"  • {file_path} ({os.path.getsize(file_path) // 1024} KB)")

    print("\n" + "=" * 78)
    print("  Unified ML Mini Project Completed Successfully!")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
