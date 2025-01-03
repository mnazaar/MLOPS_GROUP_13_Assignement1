import re

from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords



def preprocess(text):

    # Check if the input is a string
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string entries or NaN

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters, numbers, and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text
    tokens = word_tokenize(text)
    # Load stopwords

    stop_words = set(stopwords.words('english'))
    # Initialize the lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()


    # Remove stopwords and apply lemmatization
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join the words back into one string
    return ' '.join(filtered_words)