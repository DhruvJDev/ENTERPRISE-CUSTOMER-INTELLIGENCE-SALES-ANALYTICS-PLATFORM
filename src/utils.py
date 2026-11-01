"""
utils.py
Utility functions for tabular data presentation, statistical calculations (Math, SciPy),
and plot styling.
"""
import os
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib cache directory to avoid warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

def setup_plot_theme():
    """Sets a clean, modern aesthetic for charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.autolayout': True
    })

def print_section_header(title: str):
    """Prints a styled section header for CLI output."""
    separator = "=" * 78
    print(f"\n{separator}")
    print(f"  {title.upper()}")
    print(f"{separator}\n")

def display_table(df: pd.DataFrame, title: str = "", max_rows: int = 5):
    """
    Renders and prints a formatted tabular view of a pandas DataFrame.
    """
    if title:
        print(f"--- [Table View: {title}] ---")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    print(df.head(max_rows).to_string(index=True))
    print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns\n")

def compute_statistics(x: np.ndarray, y: np.ndarray):
    """
    Uses Math and SciPy to calculate analytical statistics:
    - Pearson correlation coefficient and p-value (SciPy)
    - Mean and Standard Deviation (Math & NumPy)
    - Standard Error of the Estimate (Math & SciPy)
    """
    pearson_corr, p_val = stats.pearsonr(x, y)
    
    # Statistical computations using math module
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    variance_x = sum((val - mean_x) ** 2 for val in x) / (len(x) - 1)
    std_x = math.sqrt(variance_x)
    
    # SciPy z-score test
    z_scores_y = stats.zscore(y)
    
    return {
        "pearson_r": pearson_corr,
        "p_value": p_val,
        "mean_x": mean_x,
        "std_x": std_x,
        "mean_y": mean_y,
        "outlier_count_y": int(np.sum(np.abs(z_scores_y) > 2.5))
    }

