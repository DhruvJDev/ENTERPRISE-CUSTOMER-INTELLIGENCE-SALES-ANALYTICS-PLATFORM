"""
nlp_module.py
Natural Language Processing (NLP) Customer Sentiment Analysis & Model Persistence.
Operates on the 'Customer_Review' and 'Feedback_Sentiment' columns of the unified dataset.
Uses:
- pandas, numpy, joblib
- matplotlib.pyplot, seaborn
- sklearn.model_selection.train_test_split
- sklearn.feature_extraction.text.TfidfVectorizer
- sklearn.naive_bayes.MultinomialNB
- sklearn.metrics.accuracy_score, classification_report, confusion_matrix
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils import display_table, print_section_header

def run_nlp_pipeline(data_path="data/customer_sales_analytics.csv", output_dir="output_charts", model_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print_section_header("4. NLP Text Analytics: Customer Sentiment & Automated Feedback Routing")

    # Step 1: Load and tabular inspect text reviews
    df = pd.read_csv(data_path)
    display_table(
        df[['Customer_ID', 'Customer_Review', 'Feedback_Sentiment']],
        title="Customer Feedback Reviews & Ground Truth Sentiment (Sample)",
        max_rows=6
    )

    # Class distribution table
    sentiment_counts = df['Feedback_Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment Class', 'Review Count']
    sentiment_counts['Percentage (%)'] = (sentiment_counts['Review Count'] / len(df) * 100).round(2)
    display_table(sentiment_counts, title="Sentiment Class Balance in Unified Dataset", max_rows=3)

    # Step 2: Train-Test Split
    X = df['Customer_Review']
    y = df['Feedback_Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Step 3: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Display sample TF-IDF features table
    feature_names = vectorizer.get_feature_names_out()
    tfidf_sample = pd.DataFrame(
        X_train_tfidf.toarray()[:5, :8],
        columns=feature_names[:8]
    ).round(3)
    display_table(tfidf_sample, title="Sample TF-IDF Matrix (First 5 Reviews × 8 Tokens)", max_rows=5)

    # Step 4: Model Training with MultinomialNB
    nb_classifier = MultinomialNB(alpha=0.1)
    nb_classifier.fit(X_train_tfidf, y_train)

    # Predictions & Metrics
    y_pred = nb_classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive'])

    print(f"• MultinomialNB Sentiment Test Accuracy: {acc * 100:.2f}%\n")
    print("--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Sample Inference Table
    sample_eval_df = pd.DataFrame({
        'Review_Excerpt': [t[:65] + "..." if len(t) > 65 else t for t in X_test.values[:6]],
        'Actual_Sentiment': y_test.values[:6],
        'Predicted_Sentiment': y_pred[:6]
    })
    display_table(sample_eval_df, title="NLP Test Set Predictions vs Ground Truth", max_rows=6)

    # Step 5: Visualizations (Confusion Matrix & Top Predictive Features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot A: Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=axes[0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[0].set_title(f"Sentiment Confusion Matrix (MultinomialNB)\nAccuracy = {acc*100:.1f}%", fontsize=12, weight='bold')
    axes[0].set_xlabel("Predicted Sentiment")
    axes[0].set_ylabel("True Sentiment")

    # Plot B: Feature Log-Probability Differential
    neg_feature_probs = nb_classifier.feature_log_prob_[0]
    pos_feature_probs = nb_classifier.feature_log_prob_[1]
    diff_scores = pos_feature_probs - neg_feature_probs
    top_indices = np.argsort(diff_scores)
    
    selected_indices = np.concatenate([top_indices[:5], top_indices[-5:]])
    selected_words = [feature_names[i] for i in selected_indices]
    selected_scores = diff_scores[selected_indices]
    colors = ['crimson' if s < 0 else 'forestgreen' for s in selected_scores]

    axes[1].barh(selected_words, selected_scores, color=colors)
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_title("Key Tokens Driving Sentiment (Log Ratios)", fontsize=12, weight='bold')
    axes[1].set_xlabel("Log Probability Differential (Negative < 0 < Positive)")

    chart_path = os.path.join(output_dir, "06_nlp_text_classification.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved NLP Text Classification Chart to: {chart_path}")

    # Step 6: Model Serialization with Joblib
    model_payload = {
        'vectorizer': vectorizer,
        'classifier': nb_classifier,
        'labels': nb_classifier.classes_
    }
    saved_model_path = os.path.join(model_dir, "sentiment_pipeline.joblib")
    joblib.dump(model_payload, saved_model_path)
    print(f"✓ Saved Trained NLP Pipeline via Joblib to: {saved_model_path}")

    # Step 7: Live Inference with Reloaded Joblib Model
    loaded_payload = joblib.load(saved_model_path)
    new_reviews = [
        "Incredible shopping experience! Checkout was fast and delivery arrived within 24 hours.",
        "Terrible experience. Package arrived late, broken merchandise and support was unhelpful."
    ]
    test_vecs = loaded_payload['vectorizer'].transform(new_reviews)
    live_preds = loaded_payload['classifier'].predict(test_vecs)

    live_inference_df = pd.DataFrame({
        'New Incoming Review': new_reviews,
        'Automated Sentiment Classification': live_preds
    })
    display_table(live_inference_df, title="Real-Time Inference using Reloaded Joblib Model", max_rows=3)

    return {
        "model_path": saved_model_path,
        "accuracy": acc,
        "classifier": nb_classifier
    }

if __name__ == "__main__":
    run_nlp_pipeline()
