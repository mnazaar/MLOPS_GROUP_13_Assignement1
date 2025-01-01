import pandas as pd
from flask import Flask, request, jsonify

from Twitter_Sentiment_Indian_Election_2019.src.main.predict import predict
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv

app = Flask(__name__)

df_local = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")

model, vectorizer, evaluation_scores, cv_scores = train_model_with_cv(df_local)


@app.route('/mlops/predict_election_sentiment', methods=['GET'])
def predict_sentiment():
    try:
        # Get the sentence from the POST request
        data = request.get_json()
        sentence = data.get('sentence', None)

        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        # Predict sentiment using the model
        predicted_class = predict(model, vectorizer, sentence)

        # Return the predicted class as a JSON response
        return jsonify({"predicted_class": predicted_class}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
