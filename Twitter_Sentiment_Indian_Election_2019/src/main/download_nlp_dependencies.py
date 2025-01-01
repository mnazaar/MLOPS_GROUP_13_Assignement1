import nltk
from nltk.corpus import stopwords
def download_nlp():
    # Download stopwords and punkt tokenizer if not already done
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
