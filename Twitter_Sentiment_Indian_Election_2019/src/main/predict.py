import os

import pandas as pd

from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv

data_file_path = os.getenv("DATA_FILE_PATH")

df_local = pd.read_csv(data_file_path)

model_trained, vectorizer_trained, evaluation_scores, cv_scores = train_model_with_cv(df_local)


def predict(model, vectorizer, text):
    download_nlp()
    # Step 1: Preprocess the text (same as training data preprocessing)
    cleaned_text = preprocess(text)

    # Step 2: Transform the text using the same vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])

    # Step 3: Predict sentiment using the trained model
    prediction = model.predict(text_vectorized)

    print(prediction[0])

    # Return the predicted category
    return prediction[0]
