"""
generate_single_dataset.py
Generates a comprehensive, realistic single unified dataset:
'data/customer_sales_analytics.csv'
Contains continuous, categorical, behavioral, target, and text review data
all in ONE unified table for Regression, Multi-Model Classification,
Clustering, and NLP Sentiment Analysis.
"""
import os
import math
import numpy as np
import pandas as pd

def generate_unified_dataset(output_path="data/customer_sales_analytics.csv", n_samples=500):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    customer_ids = [f"CUST-{1000 + i}" for i in range(1, n_samples + 1)]
    ages = np.random.randint(19, 68, size=n_samples)
    genders = np.random.choice(['Female', 'Male', 'Other'], size=n_samples, p=[0.49, 0.47, 0.04])
    membership_tiers = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=n_samples, p=[0.35, 0.30, 0.22, 0.13])

    # Realistic continuous features
    annual_income = np.random.normal(68, 24, size=n_samples).clip(22, 160).round(1)
    app_weekly_hours = (np.random.exponential(scale=3.8, size=n_samples) + 0.5).clip(0.5, 22.0).round(1)
    discount_usage = np.random.beta(a=2, b=5, size=n_samples) * 50  # percentage 0 to 50%
    discount_usage = discount_usage.round(1)

    # Base purchase frequency and average order value influenced by income & tier
    tier_weights = {'Bronze': 1.0, 'Silver': 1.25, 'Gold': 1.6, 'Platinum': 2.1}
    
    purchase_frequency = []
    avg_order_value = []
    total_spend = []
    high_value_flag = []

    for i in range(n_samples):
        tw = tier_weights[membership_tiers[i]]
        inc = annual_income[i]
        
        # Frequency: 4 to 45 purchases/year
        freq = int(np.random.poisson(lam=12 * tw) + np.random.uniform(1, 5))
        freq = max(3, min(freq, 48))
        purchase_frequency.append(freq)

        # Average order value: $30 to $280
        base_aov = 40.0 + (inc * 0.85) * (tw * 0.6) + np.random.normal(0, 15)
        aov = round(max(28.0, base_aov), 2)
        avg_order_value.append(aov)

        # Mathematical total spend model using math.log and non-linear synergy
        spend_base = freq * aov
        # Synergy with mobile engagement (weekly hours)
        engagement_factor = 1.0 + (math.log(1.0 + app_weekly_hours[i]) * 0.08)
        discount_impact = 1.0 - (discount_usage[i] * 0.003)
        noise = np.random.normal(0, 75)

        spend = round((spend_base * engagement_factor * discount_impact) + noise, 2)
        spend = max(180.0, spend)
        total_spend.append(spend)

        # Binary Classification Target: High-Value VIP Customer (1 = Yes, 0 = No)
        # Defined by spend > $2200 or (spend > $1700 and Gold/Platinum)
        is_vip = 1 if (spend >= 2250.0 or (spend >= 1650.0 and membership_tiers[i] in ['Gold', 'Platinum'])) else 0
        high_value_flag.append(is_vip)

    # -------------------------------------------------------------
    # Natural Text Feedback & Reviews corresponding to customer experience
    # -------------------------------------------------------------
    pos_reviews = [
        "Incredible shopping experience! Checkout was fast and delivery arrived within 24 hours.",
        "The mobile app is smooth, responsive, and makes tracking packages effortless.",
        "Superb quality products and excellent customer support. Resolved my return in minutes.",
        "Very satisfied with my purchase. Great discount deals and rewards program.",
        "Top-notch reliability! Packaging was pristine and items exceeded my expectations.",
        "Loved the curated recommendations. Saved me so much time finding the right gifts.",
        "Five stars! The delivery driver was polite and customer care was wonderfully helpful.",
        "Seamless returns and instant refund processing. Best e-commerce site I've used.",
        "Outstanding value for money. Durable build quality and quick customer service.",
        "Extremely pleased with the fast dispatch and accurate product specifications.",
        "Fantastic user interface, quick load times, and easy payment via digital wallet.",
        "Generous loyalty perks and speedy resolution to my inquiries. Highly recommended!",
        "Brilliant shopping platform. Everything arrived safely and ahead of schedule.",
        "The customer service agent was knowledgeable, courteous, and very efficient.",
        "Great discounts during the seasonal sale! High quality products as always.",
        "Clean design, intuitive navigation, and reliable order tracking notifications."
    ]

    neg_reviews = [
        "Terrible experience. Package arrived two weeks late with torn cardboard and missing items.",
        "App keeps freezing and crashing whenever I try to add items to my shopping cart.",
        "Customer service was rude and completely unhelpful. Refused to honor my return policy.",
        "Very disappointed with the product quality. Broke after just two days of light use.",
        "Hidden fees added at final checkout. Misleading promotional pricing and false advertising.",
        "Horrible refund process. Waiting over a month for my money to be credited back.",
        "Defective item received. Nobody answers support tickets or customer helpline.",
        "Order was cancelled without explanation or notification. Extremely frustrating experience.",
        "Poor packaging and sluggish delivery. Item was scratched and dented upon arrival.",
        "The latest app update completely broke navigation and search filters. Needs immediate fix.",
        "Worst customer support ever. Automated bot goes in endless loops without solving anything.",
        "Overpriced and inferior build quality. Definitely not worth the high price tag.",
        "Constant delivery delays and tracking information was inaccurate throughout.",
        "Payment was charged twice but order failed to generate. Difficult to reach live agents.",
        "Faulty merchandise and frustrating return policy. Would rate zero stars if possible.",
        "Completely dissatisfied with this purchase. Poor durability and nonexistent warranty."
    ]

    # Assign sentiment and review text
    # Dissatisfied customers are correlated with lower tiers, delivery issues, or lower spend
    customer_reviews = []
    feedback_sentiments = []

    for i in range(n_samples):
        # Higher spend and higher tiers have higher probability of positive review
        pos_prob = 0.72 if high_value_flag[i] == 1 else 0.42
        if membership_tiers[i] in ['Platinum', 'Gold']:
            pos_prob += 0.15
        pos_prob = min(0.90, max(0.15, pos_prob))

        is_positive = np.random.rand() < pos_prob
        sentiment = "Positive" if is_positive else "Negative"
        feedback_sentiments.append(sentiment)

        # Pick review text
        pool = pos_reviews if is_positive else neg_reviews
        base_text = np.random.choice(pool)
        
        # Add slight natural variation
        addon = np.random.choice([
            "",
            " Definitely will recommend to friends.",
            " Will think twice before ordering again.",
            " Appreciate the quick update.",
            " Very frustrating to deal with."
        ])
        if sentiment == "Positive" and "recommend" in addon or "Appreciate" in addon or addon == "":
            review = f"{base_text}{addon}"
        elif sentiment == "Negative" and "twice" in addon or "frustrating" in addon or addon == "":
            review = f"{base_text}{addon}"
        else:
            review = base_text
        customer_reviews.append(review)

    # Build the single consolidated DataFrame
    unified_df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Membership_Tier': membership_tiers,
        'Annual_Income_kUSD': annual_income,
        'Purchase_Frequency_Yearly': purchase_frequency,
        'Avg_Order_Value_USD': avg_order_value,
        'App_Weekly_Hours': app_weekly_hours,
        'Discount_Usage_Pct': discount_usage,
        'Total_Annual_Spend_USD': total_spend,
        'High_Value_Customer': high_value_flag,
        'Customer_Review': customer_reviews,
        'Feedback_Sentiment': feedback_sentiments
    })

    unified_df.to_csv(output_path, index=False)
    print(f"✓ Successfully generated Unified Dataset with {len(unified_df)} rows and {unified_df.shape[1]} columns!")
    print(f"✓ Saved to: {output_path}")
    return unified_df

if __name__ == "__main__":
    generate_unified_dataset()

