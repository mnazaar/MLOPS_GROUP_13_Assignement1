import os
import json

import joblib
import mlflow
import pandas as pd
from flask import Flask, request, jsonify

from Twitter_Sentiment_Indian_Election_2019.src.main.predict import predict
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model import train_model_with_cv
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model_with_grid_search import train_model_with_gs

app = Flask(__name__)

data_file_path = os.getenv("DATA_FILE_PATH")

df_global = pd.read_csv(data_file_path)

model, vectorizer, evaluation_scores, cv_scores = train_model_with_cv(df_global)

# Hyperparameter grid for both TfidfVectorizer and Logistic Regression
param_grid = {
    'tfidf__min_df': [0.001, 0.01],  # min_df range for TF-IDF
    'tfidf__max_features': [1000, 1500, 10000],  # max number of features for TF-IDF
    'clf__penalty': ['l1', 'l2'],  # L2 regularization
    'clf__max_iter': [100, 500, 1000]  # max_iter for Logistic Regression as a hyperparameter
}


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
        predicted_class_string = translate_to_english(predicted_class)

        # Return the predicted class as a JSON response
        return jsonify({"predicted_class": predicted_class, "Equivalent english translation": predicted_class_string}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def translate_to_english(predicted_class):
    if predicted_class == 0:
        predicted_class_string = "Neutral Sentiment"
    elif predicted_class == 1:
        predicted_class_string = "Positive Sentiment"
    elif predicted_class == -1:
        predicted_class_string = "Negative Sentiment"
    return predicted_class_string


train_model_with_gs(df_global, param_grid)


@app.route('/mlops/retrain_on_demand', methods=['POST'])
def on_demand_retrain():
    global param_grid

    try:
        best_model, best_params, evaluation_scores, cv_scores = train_model_with_gs(df_global, param_grid)
    except Exception as e:
        return jsonify({"error": f"Model training failed: {str(e)}"}), 500

    # Prepare the response
    response = {
        "best_params": best_params,
        "evaluation_scores": evaluation_scores
    }

    return jsonify(response), 200


@app.route('/mlops/predict_sentiment_best_model', methods=['GET'])
def predict_sentiment_best_model():
    try:
        # Get the sentence from the POST request
        data = request.get_json()
        sentence = data.get('sentence', None)

        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        best_model = joblib.load('best_model_twitter_senti.pkl')  # Replace with your actual file path

        if best_model is None:
            return jsonify({"error": "Model not trained yet"}), 400

        predictions = best_model.predict([sentence])

        predicted_class_string = translate_to_english(predictions[0])


        # Return the predicted class as a JSON response
        return jsonify({"predicted_class": predictions[0], "Translated equivalent": predicted_class_string}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mlops/list_mlflow_experiments', methods=['GET'])
def list_mlflow_experiments():
    experiments = mlflow.search_experiments(filter_string=f'name="Twitter_Sentiment_Analysis"')
    experiment_results = []


    # Loop through each experiment
    for experiment in experiments:
        experiment_id = experiment.experiment_id
        experiment_name = experiment.name
        experiment_results = []
        # Get all runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])

        # Loop through each run
        for _, run in runs.iterrows():
            run_id = run['run_id']

            # Fetch parameters and metrics from the run
            params = mlflow.get_run(run_id).data.params
            metrics = mlflow.get_run(run_id).data.metrics

            # Extract relevant parameters (you can add more as needed)
            best_params = params.get('best parameters', 'N/A')


            # Extract accuracy and other metrics (precision, recall, F1 score)
            accuracy = metrics.get('accuracy', 'N/A')
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1_score = metrics.get('f1_score', 'N/A')

            # Append the details to the results list
            experiment_results.append({
                "Experiment ID": experiment_id,
                "Experiment Name": experiment_name,
                "Run ID": run_id,
                "Model Name": "Twitter Sentiment Analysis",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,
                "Best params found": best_params
            })

    return jsonify({"experiment_results":experiment_results}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
