import pandas as pd

from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv

df_local = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")

model_trained, vectorizer_trained, evaluation_scores, cv_scores = train_model_with_cv(df_local)


def predict(model, vectorizer, text):

    # Step 1: Preprocess the text (same as training data preprocessing)
    cleaned_text = preprocess(text)

    # Step 2: Transform the text using the same vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])

    # Step 3: Predict sentiment using the trained model
    prediction = model.predict(text_vectorized)

    print(prediction[0])

    # Return the predicted category
    return prediction[0]
