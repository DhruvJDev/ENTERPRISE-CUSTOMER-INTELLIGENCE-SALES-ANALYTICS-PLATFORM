"""
clustering_module.py
Unsupervised K-Means Customer Segmentation Pipeline on the unified dataset.
Uses:
- numpy, pandas
- matplotlib.pyplot, seaborn
- sklearn.cluster.KMeans
- sklearn.preprocessing.StandardScaler
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils import display_table, print_section_header

def run_clustering_pipeline(data_path="data/customer_sales_analytics.csv", output_dir="output_charts"):
    os.makedirs(output_dir, exist_ok=True)
    print_section_header("3. Unsupervised Learning: K-Means Customer Persona Segmentation")

    # Step 1: Load data and select customer segmentation features
    df = pd.read_csv(data_path)
    features = ['Annual_Income_kUSD', 'Total_Annual_Spend_USD']
    X = df[features]
    display_table(X, title="Clustering Input Features (Income vs Annual Spend)", max_rows=5)

    # Step 2: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Elbow Method to identify optimal K (from k=1 to 8)
    k_range = range(1, 9)
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    elbow_df = pd.DataFrame({
        'Number_of_Clusters (k)': list(k_range),
        'Inertia (Sum of Squared Distances)': [round(val, 2) for val in inertias]
    })
    display_table(elbow_df, title="Elbow Method: Inertia vs Cluster Count (k)", max_rows=8)

    # Step 4: Fit Optimal Model (k=3 for Low, Medium, High Spender Personas)
    optimal_k = 3
    kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans_opt.fit_predict(X_scaled)
    
    # Calculate inverse transformed cluster centers in original business units
    cluster_centers_orig = scaler.inverse_transform(kmeans_opt.cluster_centers_)
    
    # Sort cluster IDs by spend center to assign clear business persona labels
    sorted_cluster_idx = np.argsort(cluster_centers_orig[:, 1])
    persona_names = {
        sorted_cluster_idx[0]: "Budget / Occasional Shoppers",
        sorted_cluster_idx[1]: "Mid-Market Core Regulars",
        sorted_cluster_idx[2]: "Affluent High-Spend VIPs"
    }
    df['Customer_Persona'] = df['Cluster_ID'].map(persona_names)

    # Tabular cluster persona summary
    persona_summary = df.groupby('Customer_Persona').agg(
        Customer_Count=('Customer_ID', 'count'),
        Avg_Income_kUSD=('Annual_Income_kUSD', 'mean'),
        Avg_Spend_USD=('Total_Annual_Spend_USD', 'mean'),
        Min_Spend_USD=('Total_Annual_Spend_USD', 'min'),
        Max_Spend_USD=('Total_Annual_Spend_USD', 'max'),
        Avg_Purchases=('Purchase_Frequency_Yearly', 'mean')
    ).round(2).reset_index()
    display_table(persona_summary, title="Customer Personas Segmentation Summary Table", max_rows=5)

    # Step 5: Visualizations (Dual-plot: Elbow Curve & Clustered Scatter Plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot A: Elbow Curve
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    axes[0].set_title("Elbow Method for Optimal k", fontsize=13, weight='bold')
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Plot B: 2D Clustered Scatter with Centroids
    palette = ['#2b5c8f', '#2a9d8f', '#e76f51']
    sns.scatterplot(
        data=df,
        x='Annual_Income_kUSD',
        y='Total_Annual_Spend_USD',
        hue='Customer_Persona',
        palette=palette,
        alpha=0.85,
        s=65,
        ax=axes[1]
    )

    # Overlay cluster centroids
    axes[1].scatter(
        cluster_centers_orig[:, 0],
        cluster_centers_orig[:, 1],
        s=260,
        c='gold',
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='Persona Centroids',
        zorder=10
    )

    axes[1].set_title("Customer Market Segmentation (K-Means)", fontsize=13, weight='bold')
    axes[1].set_xlabel("Annual Income ($k USD)")
    axes[1].set_ylabel("Total Annual Spend ($ USD)")
    axes[1].legend(title="Persona", loc="upper left")

    chart_path = os.path.join(output_dir, "05_kmeans_clustering.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved K-Means Clustering Chart to: {chart_path}")

    return {
        "kmeans": kmeans_opt,
        "scaler": scaler,
        "summary": persona_summary,
        "centers": cluster_centers_orig
    }

if __name__ == "__main__":
    run_clustering_pipeline()
