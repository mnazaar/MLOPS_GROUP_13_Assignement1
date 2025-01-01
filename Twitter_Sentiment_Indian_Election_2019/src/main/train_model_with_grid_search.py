import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess


def train_model_with_gs(df_local, param_grid):
    download_nlp()
    df_local['cleaner_text'] = df_local['clean_text'].apply(preprocess)
    df_local = df_local.dropna(subset=['category'])
    df_local = df_local.dropna(subset=['cleaner_text'])

    documents = df_local['cleaner_text']
    labels = df_local['category']

    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

    # Define pipeline with TfidfVectorizer and LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Placeholder for TF-IDF
        ('clf', LogisticRegression())  # Logistic Regression classifier
    ])

    # Perform GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(documents, labels)

    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_model_twitter_senti.pkl')  # Save the entire model as a pickle file

    lr_y_predictions = best_model.predict(x_test)
    elapsed_time = time.time() - start_time
    model_scores = {"Logistic Regression": calculate_perf_stats(y_test, lr_y_predictions, elapsed_time)}


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
