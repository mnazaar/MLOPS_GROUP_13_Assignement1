import pandas as pd
import pytest

from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess


@pytest.mark.parametrize("text, expected_output", [
    ("Check out https://example.com! It's amazing! #NLP @OpenAI", "check amazing"),
    (12345, ""),
    ("", ""),
    ("the and is of in", ""),
    ("Hello!!! *** $$$ World???", "hello world"),
    ("There are 123 apples and 456 oranges.", "apple orange"),
    ("This Is A Mixed CASE Text.", "mixed case text"),
])
def test_preprocess(text, expected_output):
    download_nlp()
    assert preprocess(text) == expected_output


def test_preprocess_on_csv():
    download_nlp()
    df = pd.read_csv("../../data/Twitter_Data_1K_rows.csv")
    assert preprocess
