import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv


def test_train_model():
    df_local = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")

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

