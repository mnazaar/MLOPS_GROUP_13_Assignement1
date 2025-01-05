import os
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess

import mlflow

pkl_file_path = os.getenv("PKL_FILE_PATH")


def train_model_with_gs(df_local, param_grid):
    download_nlp()
    df_local['cleaner_tweet'] = df_local['tweet'].apply(preprocess)

    number_of_records = len(df_local)

    df_local = df_local.dropna(subset=['category'])
    df_local = df_local.dropna(subset=['cleaner_tweet'])

    documents = df_local['cleaner_tweet']
    labels = df_local['category']
    # Define pipeline with TfidfVectorizer and LogisticRegression

    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Placeholder for TF-IDF
        ('clf', LogisticRegression())  # Logistic Regression classifier
    ])

    # Perform GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    mlflow.set_tracking_uri('http://localhost:5000')
    if not mlflow.get_experiment_by_name('Twitter_Sentiment_Analysis'):
        mlflow.create_experiment('Twitter_Sentiment_Analysis')

    mlflow.set_experiment('Twitter_Sentiment_Analysis')
    with mlflow.start_run():
        mlflow.set_tag("Dataset Size", f"{number_of_records} Tweets")
        grid_search.fit(documents, labels)


        best_model = grid_search.best_estimator_
        lr_y_predictions = best_model.predict(x_test)
        elapsed_time = time.time() - start_time

        model_scores = calculate_perf_stats(y_test, lr_y_predictions, elapsed_time)
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_metric('accuracy', model_scores["accuracy"])
        mlflow.log_metric('precision', model_scores["precision"])
        mlflow.log_metric('recall', model_scores["recall"])
        mlflow.log_metric('f1_score', model_scores["f1_score"])
        mlflow.log_param("best parameters", grid_search.best_params_)

    print(f"Best parameters found: {grid_search.best_params_}")
    joblib.dump(best_model, pkl_file_path)  # Save the entire model as a pickle file

    return best_model, grid_search.best_params_, model_scores, grid_search.cv_results_


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
        "time_taken": f"{time_taken:.2f} seconds"
    }
    return score
