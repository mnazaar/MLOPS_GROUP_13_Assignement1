import os

import joblib

data_file_path = os.getenv("DATA_FILE_PATH")
pkl_file_path = os.getenv("PKL_FILE_PATH")


def test_predict_positive_class():
    best_model = joblib.load(pkl_file_path)

    predictions = best_model.predict(["This election is going to make a positive impact to people"])

    assert predictions[0] == 1, f"Expected sentiment to be 1 (positive), but got {predictions[0]}"


def test_predict_negative_class():
    best_model = joblib.load(pkl_file_path)

    predictions = best_model.predict(["Political parties are all corrupt and incompetent."])

    assert predictions[0] == -1, f"Expected sentiment to be -1 (negative), but got {predictions[0]}"


def test_predict_neutral_class():
    best_model = joblib.load(pkl_file_path)

    predictions = best_model.predict(["I have no opinion on this election"])

    assert predictions[0] == 0, f"Expected sentiment to be 0 (neutral), but got {predictions[0]}"
