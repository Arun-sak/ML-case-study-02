# ============================================================
# CASE STUDY 2: RESTAURANT REVIEW SENTIMENT ANALYSIS
# Algorithm: TF-IDF + Logistic Regression
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

file_path = r"C:\Users\dkrup\OneDrive\ドキュメント\case study 2\Restaurant_Reviews.csv"

# This dataset is tab-separated
df = pd.read_csv(file_path, sep="\t")

print("\n========== RESTAURANT SENTIMENT ANALYSIS ==========")
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst five records:")
print(df.head())

# ------------------------------------------------------------
# 2. Select features and target
# ------------------------------------------------------------

X = df["Review"].astype(str)
y = pd.to_numeric(df["Liked"], errors="coerce")

# Remove invalid target values and empty reviews
valid_rows = (
    y.notna()
    & (X.str.strip() != "")
)

X = X[valid_rows]
y = y[valid_rows].astype(int)

print("\nTotal reviews:", len(X))
print("Positive reviews:", (y == 1).sum())
print("Negative reviews:", (y == 0).sum())

# ------------------------------------------------------------
# 3. Split dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 4. Create model
# ------------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# ------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")

# ------------------------------------------------------------
# 6. Predict test data
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 7. Evaluate model
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ------------------------------------------------------------
# 8. Predict new restaurant reviews
# ------------------------------------------------------------

print("\n========== NEW REVIEW PREDICTION ==========")

while True:

    review = input(
        "\nEnter restaurant review "
        "(or type exit to stop): "
    ).strip()

    if review.lower() == "exit":
        print("Sentiment Analysis System closed.")
        break

    if review == "":
        print("Please enter a valid review.")
        continue

    prediction = model.predict([review])[0]

    probabilities = model.predict_proba([review])[0]

    if prediction == 1:
        sentiment = "Positive"
        confidence = probabilities[1] * 100
    else:
        sentiment = "Negative"
        confidence = probabilities[0] * 100

    print("Predicted Sentiment:", sentiment)
    print("Model Confidence:", round(confidence, 2), "%")
