import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from Twitter_Sentiment_Indian_Election_2019.src.main.predict import predict
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv


def test_predict_positive_class():
    df_local = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")

    model, vectorizer, evaluation_scores, cv_scores = train_model_with_cv(df_local)
    result = predict(model, vectorizer, "Both Congress and BJP are doing their best for the people, though their ideas may differ")
    assert result == 1, f"Expected sentiment to be 1 (positive), but got {result}"


def test_predict_negative_class():
    df_local = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")

    model, vectorizer, evaluation_scores, cv_scores = train_model_with_cv(df_local)
    result = predict(model, vectorizer, "Most political leaders are corrupt and incompetent. There is no hope for the people")
    assert result == -1, f"Expected sentiment to be -1 (negative), but got {result}"


def test_predict_neutral_class():
    df_local = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")

    model, vectorizer, evaluation_scores, cv_scores = train_model_with_cv(df_local)
    result = predict(model, vectorizer, "I have no opinion on the election")
    assert result == 0, f"Expected sentiment to be 0 (neutral), but got {result}"
