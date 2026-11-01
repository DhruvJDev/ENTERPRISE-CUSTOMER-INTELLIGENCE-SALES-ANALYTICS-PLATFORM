# Enterprise Customer Intelligence & Sales Analytics Platform
### A Unified Machine Learning Project on a Single E-Commerce Dataset

This project converts disjoint machine learning exercises into a **single, realistic, cohesive, and useful enterprise business intelligence platform**. Every pipeline, tabular view, chart, and algorithm operates entirely on **one unified dataset**: [`data/customer_sales_analytics.csv`](data/customer_sales_analytics.csv).

---

## 🏢 Business Context & Core Tasks Solved

Our platform analyzes an E-Commerce enterprise with 500 customer records containing transaction history, behavioral app telemetry, demographics, and text feedback. Using this **single dataset**, we address 4 major business objectives:

1. **Revenue Forecasting (Linear Regression)**: Predict a customer's annual spend (`Total_Annual_Spend_USD`) based on their age, tier, income, purchase frequency, average order value, app usage, and discount habits.
2. **High-Value VIP Conversion (Supervised Multi-Model Classification)**: Benchmark 5 industry-standard classifiers (`LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `GaussianNB`, `SVC`) to accurately identify high-value VIP accounts for retention campaigns.
3. **Behavioral Customer Personas (Unsupervised K-Means Clustering)**: Segment customer accounts into natural spending personas using K-Means with the Elbow Method.
4. **Automated Feedback & Sentiment Triage (NLP & Model Persistence)**: Analyze customer text reviews (`Customer_Review`) with `TfidfVectorizer` and `MultinomialNB`, and persist the trained pipeline using `joblib` for real-time customer service routing.

---

## 🛠️ Complete Tech Stack & Library Integration

| Category | Library | Methods & Implementations |
| :--- | :--- | :--- |
| **Mathematics & Stats** | `math` | `math.sqrt()` (RMSE calculations), `math.log()` (feature synergy modeling) |
| **Vector Math & Arrays** | `numpy` | Matrix operations, z-score outlier detection, random seeds, indexing |
| **Tabular Data** | `pandas` | `DataFrame`, CSV ingestion, tabular previews, grouping, pivot aggregations |
| **Hypothesis Testing** | `scipy` | `scipy.stats.pearsonr()` (correlation $r$ and $p$-value), `scipy.stats.zscore()` |
| **Plot Layouts** | `matplotlib.pyplot` | Multi-panel subplots, customized axis scales, high-res figure generation |
| **Statistical Visuals** | `seaborn` | Styled correlation heatmaps, residual distribution plots, grouped bar plots |
| **Preprocessing** | `scikit-learn` | `LabelEncoder` (tier categorization), `StandardScaler` (feature scaling) |
| **Regression** | `scikit-learn` | `LinearRegression`, `train_test_split`, `mean_squared_error` |
| **Classification** | `scikit-learn` | `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `GaussianNB`, `SVC` |
| **Classification Metrics**| `scikit-learn` | `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `accuracy_score`, `classification_report` |
| **Clustering** | `scikit-learn` | `KMeans` (inertia computation across $k \in [1, 8]$, centroid extraction) |
| **NLP & Text Mining** | `scikit-learn` | `TfidfVectorizer`, `MultinomialNB` |
| **Model Persistence** | `joblib` | `joblib.dump()`, `joblib.load()` (production model serialization & live inference) |

---

## 📂 Project Structure

```
ML mini/
├── data/
│   ├── generate_single_dataset.py       # Single unified dataset generator
│   └── customer_sales_analytics.csv     # The Single Unified Dataset (500 rows × 13 columns)
├── models/
│   └── sentiment_pipeline.joblib        # Persisted NLP pipeline artifact (via joblib)
├── output_charts/                       # Generated high-resolution visualization charts
│   ├── 01_regression_analysis.png       # Scatter fit line & residual histogram
│   ├── 02_correlation_matrix.png        # Correlation heatmap across customer features
│   ├── 03_classification_metrics_comparison.png  # Multi-model 5-classifier benchmark bar chart
│   ├── 04_confusion_matrices_grid.png   # 2x3 grid of confusion matrix heatmaps
│   ├── 05_kmeans_clustering.png         # Elbow curve & 2D persona scatter with centroids
│   └── 06_nlp_text_classification.png   # Sentiment confusion matrix & word log-probability ratios
├── src/
│   ├── __init__.py
│   ├── utils.py                         # Tabular formatting, plot styling, math & scipy stats
│   ├── regression_module.py             # Revenue forecasting pipeline
│   ├── classification_module.py         # 5-classifier VIP benchmark pipeline
│   ├── clustering_module.py             # K-Means customer segmentation pipeline
│   └── nlp_module.py                    # NLP sentiment classification & joblib persistence
├── main.py                              # Master executable script
├── ml_mini_project.ipynb                # Interactive Jupyter Notebook
├── build_notebook.py                    # Notebook generator script
├── requirements.txt                     # Dependency list
└── README.md                            # Comprehensive project guide
```

---

## 📊 Single Dataset Schema (`customer_sales_analytics.csv`)

| Column Name | Data Type | Role in Project | Description |
| :--- | :--- | :--- | :--- |
| `Customer_ID` | String | Identifier | Unique account ID (`CUST-1001`, `CUST-1002`, ...) |
| `Age` | Integer | Feature | Customer age (19 to 68) |
| `Gender` | String | Feature | Demographics (`Female`, `Male`, `Other`) |
| `Membership_Tier` | String | Feature (Encoded) | Loyalty tier (`Bronze`, `Silver`, `Gold`, `Platinum`) via `LabelEncoder` |
| `Annual_Income_kUSD` | Float | Feature | Annual income in thousands of dollars |
| `Purchase_Frequency_Yearly`| Integer | Feature | Number of completed orders per year |
| `Avg_Order_Value_USD` | Float | Feature | Average dollar value per order |
| `App_Weekly_Hours` | Float | Feature | Engagement hours spent on mobile app per week |
| `Discount_Usage_Pct` | Float | Feature | Percentage of orders using coupon/discount codes |
| **`Total_Annual_Spend_USD`**| Float | **Target 1 (Regression)** | Total annual revenue generated by customer |
| **`High_Value_Customer`** | Binary | **Target 2 (Classification)**| $1 =$ VIP High-Value Customer, $0 =$ Standard Customer |
| **`Customer_Review`** | String | **Feature (NLP)** | Natural language text review written by customer |
| **`Feedback_Sentiment`** | String | **Target 3 (NLP)** | Sentiment classification label (`Positive` vs `Negative`) |

---

## 📈 Key Results & Tables Summary

### 1. Revenue Regression (`Total_Annual_Spend_USD`)
* **SciPy Pearson Correlation**: $r = 0.3172$ ($p = 3.80 \times 10^{-13}$) with Income; $r = 0.800$ with Purchase Frequency.
* **Model Fit**: $R^2 = \mathbf{0.9416}$ ($94.16\%$ variance explained).
* **Root Mean Squared Error (RMSE)**: $\mathbf{\$217.47}$ (calculated using `math.sqrt(mse)`).

### 2. Supervised Multi-Model Classification Benchmark (`High_Value_Customer`)
Features normalized via `StandardScaler`. Benchmarked on test set ($n=125$):

| Classifier Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVC)** | **96.00%** | **95.92%** | **94.00%** | **95.00%** |
| **Decision Tree** | 95.20% | 94.00% | **94.00%** | 93.88% |
| **Logistic Regression** | 94.40% | 93.88% | 92.00% | 93.62% |
| **K-Nearest Neighbors (KNN)** | 93.60% | 93.75% | 90.00% | 92.47% |
| **Gaussian Naive Bayes** | 92.00% | 91.67% | 88.00% | 90.72% |

### 3. K-Means Customer Personas (Unsupervised Segmentation)
Optimal cluster count identified at $k=3$ via the Elbow Method:

| Persona Segment | Customer Count | Avg Income ($k) | Avg Spend ($) | Avg Annual Orders |
| :--- | :---: | :---: | :---: | :---: |
| **Budget / Occasional Shoppers** | 163 | \$58.91k | \$809.04 | 10.3 orders |
| **Mid-Market Core Regulars** | 227 | \$67.45k | \$1,843.08 | 20.8 orders |
| **Affluent High-Spend VIPs** | 110 | \$81.08k | \$3,383.15 | 32.1 orders |

### 4. NLP Text Analytics & Live `joblib` Deployment
* **Model**: `TfidfVectorizer(ngram_range=(1,2))` + `MultinomialNB(alpha=0.1)`.
* **Test Accuracy**: $\mathbf{92.00\%}$ | Precision (Pos): $1.00$ | Recall (Neg): $1.00$.
* **Real-Time Live Deployment Test**:
  * Input 1: *"Incredible shopping experience! Checkout was fast and delivery arrived within 24 hours."* $\rightarrow$ **Positive**
  * Input 2: *"Terrible experience. Package arrived late, broken merchandise and support was unhelpful."* $\rightarrow$ **Negative**

---

## 🚀 How to Run the Project

```bash
# 1. Execute the master Python script
./.venv/bin/python main.py

# 2. Launch the interactive Jupyter Notebook
./.venv/bin/python -m jupyter notebook ml_mini_project.ipynb
```
*(Or open [ml_mini_project.ipynb](ml_mini_project.ipynb) directly in VS Code / Antigravity).*
