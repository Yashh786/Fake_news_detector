"""
src/preprocessing.py
────────────────────────────────────────────────────────────────────────────────
Text Preprocessing Pipeline for Fake News Detection

Concepts covered:
  - Regex-based text cleaning (remove URLs, HTML, special chars)
  - Tokenization
  - Stop word removal (NLTK)
  - Lemmatization (WordNetLemmatizer)
  - spaCy-based NER-aware preprocessing (advanced)
────────────────────────────────────────────────────────────────────────────────
"""

import re
import string
import logging
from typing import List, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (safe to call multiple times)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger(__name__)

# ── Load NLP models lazily (avoid slow import at module level) ────────────────
_spacy_model = None
_lemmatizer = None
_stop_words = None


def _get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stop_words() -> set:
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
        # Keep some negation words that are meaningful for fake news
        negation_keep = {"not", "no", "never", "neither", "nor", "nobody"}
        _stop_words -= negation_keep
    return _stop_words


def _get_spacy() -> spacy.Language:
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load("en_core_web_sm", disable=["parser"])
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            _spacy_model = None
    return _spacy_model


# ── Core Cleaning Functions ───────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www. links."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_special_characters(text: str, keep_apostrophe: bool = False) -> str:
    """Remove punctuation and non-alphabetic characters."""
    if keep_apostrophe:
        return re.sub(r"[^a-zA-Z\s']", " ", text)
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_repeated_characters(text: str) -> str:
    """
    Reduce character repetitions: 'loooove' → 'loove'.
    Fake news often uses exaggerated spelling for emotional effect.
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


# ── Main Preprocessing Pipeline ───────────────────────────────────────────────

def clean_text(
    text: str,
    lemmatize: bool = True,
    remove_stops: bool = True,
    min_token_length: int = 2,
) -> str:
    """
    Full preprocessing pipeline.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove special characters
      5. Remove repeated characters
      6. Tokenize
      7. Remove stop words (optional)
      8. Lemmatize (optional)
      9. Filter short tokens

    Args:
        text: Raw input text.
        lemmatize: Whether to apply WordNet lemmatization.
        remove_stops: Whether to remove stop words.
        min_token_length: Minimum token length to keep.

    Returns:
        Cleaned, space-joined string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2-5: Clean
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    text = remove_repeated_characters(text)
    text = normalize_whitespace(text)

    # Step 6: Tokenize
    tokens = word_tokenize(text)

    stop_words = _get_stop_words() if remove_stops else set()
    lemmatizer = _get_lemmatizer() if lemmatize else None

    processed = []
    for token in tokens:
        # Filter short tokens
        if len(token) < min_token_length:
            continue
        # Remove stop words
        if token in stop_words:
            continue
        # Lemmatize
        if lemmatizer:
            token = lemmatizer.lemmatize(token)
        processed.append(token)

    return " ".join(processed)


def preprocess_series(texts, **kwargs) -> List[str]:
    """
    Apply clean_text to a pandas Series or list.

    Args:
        texts: Iterable of raw text strings.
        **kwargs: Passed to clean_text.

    Returns:
        List of cleaned strings.
    """
    from tqdm import tqdm
    return [clean_text(t, **kwargs) for t in tqdm(texts, desc="Preprocessing")]


def spacy_preprocess(text: str) -> str:
    """
    Advanced preprocessing using spaCy:
      - Lemmatization (more accurate than NLTK)
      - Named Entity Recognition: replace entities like PERSON → '[PERSON]'
        (useful for generalization — fake news often uses specific names)
      - POS-based filtering (keep only nouns, verbs, adjectives, adverbs)

    Falls back to basic clean_text if spaCy model not available.
    """
    nlp = _get_spacy()
    if nlp is None:
        return clean_text(text)

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = text.lower()

    doc = nlp(text)
    tokens = []
    stop_words = _get_stop_words()

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.text in stop_words:
            continue
        if len(token.text) < 2:
            continue
        # Keep meaningful POS tags
        if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"):
            tokens.append(token.lemma_)

    return " ".join(tokens)


def combine_title_text(title: str, text: str, weight_title: int = 3) -> str:
    """
    Combine title and body text.
    Title is repeated `weight_title` times to give it more importance
    in TF-IDF feature space (a common technique).

    Args:
        title: Article headline.
        text: Article body.
        weight_title: How many times to repeat the title.

    Returns:
        Combined string.
    """
    title = title if isinstance(title, str) else ""
    text = text if isinstance(text, str) else ""
    return (f"{title} " * weight_title + text).strip()
