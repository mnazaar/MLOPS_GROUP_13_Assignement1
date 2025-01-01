import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess


def train_model_with_cv(df_local, min_df_hyper=0.001):
    df_local['cleaner_text'] = df_local['clean_text'].apply(preprocess)
    df_local = df_local.dropna(subset=['category'])
    df_local = df_local.dropna(subset=['cleaner_text'])

    documents = df_local['cleaner_text']
    labels = df_local['category']

    start_time = time.time()
    vectorizer = TfidfVectorizer(min_df=min_df_hyper)
    feature_data = vectorizer.fit_transform(documents)

    x_train, x_test, y_train, y_test = \
        train_test_split(feature_data, labels, test_size=0.20, random_state=42)

    lr_model = LogisticRegression()

    # Perform cross-validation
    cv_scores = cross_val_score(lr_model, x_train, y_train, cv=5, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean()}")

    # Train model on the full training data
    lr_model.fit(x_train, y_train)

    # Predict and evaluate
    lr_y_predictions = lr_model.predict(x_test)
    elapsed_time = time.time() - start_time
    model_scores = {"Logistic Regression": calculate_perf_stats(y_test, lr_y_predictions, elapsed_time)}
    return lr_model, vectorizer, model_scores, cv_scores


def calculate_perf_stats(y_test, y_prediction, time_taken):
    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction, average='weighted')
    recall = recall_score(y_test, y_prediction, average='weighted')
    f1 = f1_score(y_test, y_prediction, average='weighted')

    score = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "time_taken": f"{time_taken:.2f}"
    }
    return score

