import os

import numpy as np
import pandas as pd
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model_with_grid_search import train_model_with_gs


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

    assert cv_scores[
               'mean_test_score'] > 0.50, f"Expected mean CV accuracy to be greater than 0.65, but got {cv_scores['mean_test_score']}"
    assert np.all(cv_scores['mean_test_score'] > 0.50), f"Some CV scores are below 0.5: {cv_scores}"
    assert evaluation_scores['accuracy'] > 0.50, \
        f"Expected accuracy to be greater than 0.50, but got {evaluation_scores['accuracy']}"
    assert evaluation_scores['precision'] > 0.50, \
        f"Expected precision to be greater than 0.50, but got {evaluation_scores['precision']}"
    assert evaluation_scores['recall'] > 0.50, \
        f"Expected recall to be greater than 0.50, but got {evaluation_scores['recall']}"
    assert evaluation_scores['f1_score'] > 0.50, \
        f"Expected F1 score to be greater than 0.50, but got {evaluation_scores['f1_score']}"
