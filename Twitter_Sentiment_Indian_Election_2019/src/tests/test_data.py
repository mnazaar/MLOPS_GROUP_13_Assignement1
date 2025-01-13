import pandas as pd
import pytest
import os
from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess


@pytest.mark.parametrize("text, expected_output", [
    ("Check out https://example.com! It's amazing! #NLP @OpenAI", "check amazing"),
    (12345, ""),  # Assuming non-string input is handled as empty string
    ("", ""),
    ("the and is of in", ""),
    ("Hello!!! *** $$$ World???", "hello world"),
    ("There are 123 apples and 456 oranges.", "apple orange"),
    ("This Is A Mixed CASE Text.", "mixed case text"),
])
def test_preprocess(text, expected_output):
    download_nlp()
    assert preprocess(text) == expected_output


# Path to the CSV file set via environment variable
data_file_path = os.getenv("DATA_FILE_PATH")


#Test that after preprocessing ther eare no null rows in the dataset due to text cleaning.
#A few other data validations/consistency checks
def test_preprocess_on_csv():
    download_nlp()
    assert data_file_path, "DATA_FILE_PATH environment variable not set or invalid."

    # Read the CSV file
    df = pd.read_csv(data_file_path)

    # Ensure the CSV file has a column named 'text'
    assert 'tweet' in df.columns, "'tweet' column not found in the dataset."

    # Process each row in the 'text' column
    df['clean_tweet'] = df['tweet'].apply(preprocess)

    # Check if the processed_text column does not contain any null values
    assert df['clean_tweet'].isnull().sum() == 0, "Null values found in processed_text column."