"""Text preprocessing and feature extraction using TF-IDF."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer

# Common English stopwords (no NLTK needed)
STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "about", "but", "or", "and", "not",
    "no", "so", "if", "this", "that", "these", "those", "am", "than",
}


def clean_text(text: str) -> str:
    """Remove special characters and stopwords."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def create_vectorizer(max_features: int = 500) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams + bigrams
        stop_words="english",
    )
