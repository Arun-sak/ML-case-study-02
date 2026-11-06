# Case Study 2: Restaurant Sentiment Analysis

## Project Overview

This project classifies restaurant customer reviews as **Positive** or
**Negative** using Natural Language Processing (NLP), TF-IDF and
Logistic Regression.

## Problem Statement

Manually reading and classifying a large number of restaurant reviews
takes time. This project uses machine learning to identify the sentiment
expressed in each review.

## Objectives

-   Load restaurant reviews from a CSV file.
-   Prepare review text for machine learning.
-   Convert text into numerical features using TF-IDF.
-   Train a Logistic Regression classifier.
-   Predict positive or negative sentiment.
-   Evaluate the model using accuracy, precision, recall, F1-score and a
    confusion matrix.

## Dataset

**File:** `Restaurant_Reviews.csv`

  Column     Description
  ---------- ---------------------------------------------
  `Review`   Written customer review
  `Liked`    Sentiment label: 1 = positive, 0 = negative

The dataset contains 1,000 reviews: 500 positive and 500 negative.

**File format note:** The copy used for this project is tab-separated.
Load it with `pd.read_csv("Restaurant_Reviews.csv", sep="\\t")`. If your
copy is comma-separated, use the appropriate separator instead.

## Algorithms and Techniques

### 1. Text Preprocessing

Prepare the review text. Depending on the code, this may include
converting text to lowercase and removing punctuation or unnecessary
words.

### 2. TF-IDF Vectorization

TF-IDF converts review text into numerical features and gives more
importance to words that help distinguish reviews. Fit the vectorizer on
training data only, then use the fitted vectorizer to transform test
data.

### 3. Logistic Regression

Logistic Regression is a supervised classification algorithm. It learns
from labelled reviews and predicts whether a new review is positive or
negative.

## Workflow

1.  Import the required libraries.
2.  Load `Restaurant_Reviews.csv` with the correct separator.
3.  Check the columns and missing values.
4.  Separate `Review` as input text and `Liked` as the target.
5.  Split the data into training and testing sets using an 80:20
    stratified split.
6.  Convert the text into TF-IDF features.
7.  Train the Logistic Regression model.
8.  Predict sentiment for the test reviews.
9.  Evaluate the predictions using classification metrics and a
    confusion matrix.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Natural Language Processing (NLP)
-   TF-IDF Vectorizer
-   Logistic Regression
-   Matplotlib (if used for visualizations)

## Requirements

Install the required packages:

``` bash
pip install pandas numpy scikit-learn matplotlib
```

## Project Folder Structure

``` text
Restaurant-Sentiment-Analysis/
├── sentiment_analysis.py
├── Restaurant_Reviews.csv
└── README.md
```

Update the script's dataset path if your files are stored in a different
location.

## How to Run

1.  Place `Restaurant_Reviews.csv` in the project folder or update its
    path in the script.
2.  Install the required packages.
3.  Open the project in Spyder, VS Code or another Python IDE.
4.  Run `sentiment_analysis.py`.
5.  Review the predicted sentiment and evaluation results.

## Sample Input and Output

**Sample review:**

``` text
The food was delicious and the service was excellent.
```

**Illustrative output:**

``` text
Predicted sentiment: Positive
```

The actual prediction depends on the trained model.

## Model Evaluation

  -----------------------------------------------------------------------
  Metric                              Meaning
  ----------------------------------- -----------------------------------
  Accuracy                            Overall proportion of correct
                                      predictions

  Precision                           Among reviews predicted positive,
                                      the proportion actually positive

  Recall                              Among actual positive reviews, the
                                      proportion correctly identified

  F1-score                            Harmonic mean of precision and
                                      recall

  Confusion Matrix                    Counts correct and incorrect
                                      predictions for each class
  -----------------------------------------------------------------------

Use the values produced by your own program in the final report and
presentation. Results may vary with the split, random state,
preprocessing and model settings.

## Limitations

-   The model predicts only positive or negative sentiment.
-   Sarcasm, irony and mixed opinions may be difficult to classify.
-   Performance depends on dataset quality and size.
-   Spelling mistakes and unfamiliar words may affect predictions.

## Future Enhancements

-   Build a web interface for entering reviews.
-   Compare Logistic Regression with Naive Bayes, Decision Tree and
    Support Vector Machine.
-   Add more reviews from different restaurants.
-   Analyze separate aspects such as food, service, price and ambience.

## References

-   Scikit-learn: https://scikit-learn.org/stable/
-   TF-IDF Vectorizer:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
-   Logistic Regression:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
-   Model evaluation:
    https://scikit-learn.org/stable/modules/model_evaluation.html
