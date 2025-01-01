import os

import numpy as np
import pandas as pd
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model_with_grid_search import train_model_with_gs


def test_train_model():
    data_file_path = os.getenv("DATA_FILE_PATH")

    df_local = pd.read_csv(data_file_path)

    model, vectorizer, evaluation_scores, cv_scores = train_model_with_cv(df_local)

    print(cv_scores)
    print(evaluation_scores)

    assert cv_scores.mean() > 0.65, f"Expected mean CV accuracy to be greater than 0.65, but got {cv_scores.mean()}"
    assert np.all(cv_scores > 0.65), f"Some CV scores are below 0.5: {cv_scores}"
    assert 'Logistic Regression' in evaluation_scores, "Logistic Regression results not found in evaluation scores"
    assert evaluation_scores['Logistic Regression']['accuracy'] > 0.65, \
        f"Expected accuracy to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['accuracy']}"
    assert evaluation_scores['Logistic Regression']['precision'] > 0.65, \
        f"Expected precision to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['precision']}"
    assert evaluation_scores['Logistic Regression']['recall'] > 0.65, \
        f"Expected recall to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['recall']}"
    assert evaluation_scores['Logistic Regression']['f1_score'] > 0.65, \
        f"Expected F1 score to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['f1_score']}"

    assert float(evaluation_scores['Logistic Regression']['time_taken']) < 1.0, \
        "Model training took too long, expected time to be less than 1 second"


## The test for gridsearch with just one grid item per hyper parameter to ensure code is working fine
def test_train_model_with_gs():
    data_file_path = os.getenv("DATA_FILE_PATH")

    df_local = pd.read_csv(data_file_path)

    # Hyperparameter grid for both TfidfVectorizer and Logistic Regression
    param_grid = {
        'tfidf__min_df': [0.001],  # min_df range for TF-IDF
        'tfidf__max_features': [1000],  # max number of features for TF-IDF
        'clf__penalty': ['l1'],  # L2 regularization
        'clf__solver': ['liblinear'],  # Solvers for Logistic Regression
        'clf__max_iter': [1000]  # max_iter for Logistic Regression as a hyperparameter
    }

    best_model, best_params, evaluation_scores, cv_scores = train_model_with_gs(df_local, param_grid)

    assert cv_scores['mean_test_score'] > 0.65, f"Expected mean CV accuracy to be greater than 0.65, but got {cv_scores['mean_test_score']}"
    assert np.all(cv_scores['mean_test_score'] > 0.65), f"Some CV scores are below 0.5: {cv_scores}"
    assert 'Logistic Regression' in evaluation_scores, "Logistic Regression results not found in evaluation scores"
    assert evaluation_scores['Logistic Regression']['accuracy'] > 0.65, \
        f"Expected accuracy to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['accuracy']}"
    assert evaluation_scores['Logistic Regression']['precision'] > 0.65, \
        f"Expected precision to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['precision']}"
    assert evaluation_scores['Logistic Regression']['recall'] > 0.65, \
        f"Expected recall to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['recall']}"
    assert evaluation_scores['Logistic Regression']['f1_score'] > 0.65, \
        f"Expected F1 score to be greater than 0.7, but got {evaluation_scores['Logistic Regression']['f1_score']}"

    assert float(evaluation_scores['Logistic Regression']['time_taken']) < 20.0, \
        "Model training took too long, expected time to be less than 20 seconds"
