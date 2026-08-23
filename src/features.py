"""
src/features.py
────────────────────────────────────────────────────────────────────────────────
Feature Engineering for Fake News Detection

Covers:
  - TF-IDF vectorization (unigram + bigram)
  - Bag-of-Words (CountVectorizer)
  - Handcrafted linguistic features (article length, avg word length,
    punctuation ratio, capital letter ratio, readability score)
  - Sentiment features using VADER
  - Word Embedding averaging (GloVe via gensim)
  - Feature combination utilities
────────────────────────────────────────────────────────────────────────────────
"""

import re
import os
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── TF-IDF ───────────────────────────────────────────────────────────────────

def build_tfidf(
    train_texts,
    test_texts,
    max_features: int = 50_000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    """
    Fit TF-IDF on training data and transform both splits.

    TF-IDF Formula:
        TF(t, d)  = count(t in d) / total_terms(d)
        IDF(t)    = log( N / df(t) ) + 1          [scikit-learn default]
        TF-IDF    = TF * IDF

    Why sublinear_tf=True?
        Replaces raw TF with 1 + log(TF) to dampen the effect of very
        frequent terms within one document (a word appearing 100x vs 10x
        matters less than 10x vs 1x).

    Why ngram_range=(1,2)?
        Bigrams like "fake news", "breaking story", "sources say" are
        highly predictive of misinformation — unigrams alone miss these.

    Args:
        train_texts: Iterable of cleaned training strings.
        test_texts:  Iterable of cleaned test strings.
        max_features: Vocabulary cap (top N by TF-IDF score).
        ngram_range:  (min_n, max_n) for n-gram extraction.
        sublinear_tf: Apply log normalization to TF.
        save_path:    Optional path to save fitted vectorizer.

    Returns:
        (X_train_tfidf, X_test_tfidf, fitted_vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,           # ignore terms appearing in < 2 docs (noise)
        max_df=0.95,        # ignore terms in > 95% of docs (too common)
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",  # only alphabetic tokens ≥ 2 chars
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    logger.info(f"Train matrix shape: {X_train.shape}, Test: {X_test.shape}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Vectorizer saved → {save_path}")

    return X_train, X_test, vectorizer


def load_tfidf_vectorizer(path: str) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Bag of Words ─────────────────────────────────────────────────────────────

def build_bow(
    train_texts,
    test_texts,
    max_features: int = 50_000,
) -> Tuple[csr_matrix, csr_matrix, CountVectorizer]:
    """Simple Bag-of-Words (term frequency counts, no IDF weighting)."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


# ── Handcrafted Linguistic Features ──────────────────────────────────────────

def extract_linguistic_features(texts: pd.Series) -> pd.DataFrame:
    """
    Extract handcrafted features that are NOT captured by TF-IDF.

    Features:
      - char_count          : Total characters
      - word_count          : Total words
      - avg_word_length     : Average word length (complex words → edu. content)
      - unique_word_ratio   : Unique words / total words (vocabulary richness)
      - capital_ratio       : Capitals / total chars (FAKE NEWS LOVES CAPS!)
      - punctuation_ratio   : Punctuation chars / total chars
      - exclamation_count   : Number of '!'
      - question_count      : Number of '?'
      - sentence_count      : Number of sentences (naive: count '.' '!' '?')
      - avg_sentence_length : Avg words per sentence

    Fake news signatures typically include:
      - High exclamation counts
      - High capital ratios
      - Shorter avg sentence lengths (simpler language)
      - Lower vocabulary richness
    """
    features = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            features.append({k: 0 for k in [
                "char_count", "word_count", "avg_word_length",
                "unique_word_ratio", "capital_ratio", "punctuation_ratio",
                "exclamation_count", "question_count", "sentence_count",
                "avg_sentence_length"
            ]})
            continue

        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]

        char_count = len(text)
        word_count = len(words)
        punct_chars = sum(1 for c in text if c in ".,;:!?\"'()-")
        cap_chars   = sum(1 for c in text if c.isupper())
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        unique_ratio = len(set(words)) / word_count if word_count > 0 else 0
        avg_sent_len = word_count / len(sentences) if sentences else 0

        features.append({
            "char_count":          char_count,
            "word_count":          word_count,
            "avg_word_length":     round(avg_word_len, 3),
            "unique_word_ratio":   round(unique_ratio, 3),
            "capital_ratio":       round(cap_chars / char_count, 4) if char_count else 0,
            "punctuation_ratio":   round(punct_chars / char_count, 4) if char_count else 0,
            "exclamation_count":   text.count("!"),
            "question_count":      text.count("?"),
            "sentence_count":      len(sentences),
            "avg_sentence_length": round(avg_sent_len, 3),
        })

    return pd.DataFrame(features)


# ── VADER Sentiment Features ──────────────────────────────────────────────────

def extract_sentiment_features(texts: pd.Series) -> pd.DataFrame:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment scores.

    Returns 4 scores per text:
      - neg      : Proportion of negative sentiment (0–1)
      - neu      : Proportion of neutral sentiment (0–1)
      - pos      : Proportion of positive sentiment (0–1)
      - compound : Overall sentiment (-1 very negative, +1 very positive)

    Why this matters for fake news:
      Fake articles tend toward extreme compound scores (< -0.5 or > 0.5)
      to trigger emotional reactions. Real reporting is more neutral.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    records = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            records.append({"vader_neg": 0.0, "vader_neu": 0.5,
                            "vader_pos": 0.0, "vader_compound": 0.0})
            continue
        scores = sia.polarity_scores(text)
        records.append({
            "vader_neg":      scores["neg"],
            "vader_neu":      scores["neu"],
            "vader_pos":      scores["pos"],
            "vader_compound": scores["compound"],
        })

    return pd.DataFrame(records)


# ── GloVe / Word Embedding Averaging ─────────────────────────────────────────

def load_glove_embeddings(glove_path: str) -> dict:
    """
    Load GloVe embeddings from a .txt file into a dict.

    Download: https://nlp.stanford.edu/projects/glove/
    Recommended: glove.6B.300d.txt (822 MB, 400k words, 300-dim vectors)

    Args:
        glove_path: Path to glove.*.txt file.

    Returns:
        Dict mapping word → np.array of shape (dim,)
    """
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    logger.info(f"Loaded {len(embeddings):,} GloVe vectors")
    return embeddings


def text_to_glove_vector(text: str, embeddings: dict, dim: int = 300) -> np.ndarray:
    """
    Average all word vectors in a text to get a document vector.

    Why averaging works:
      The centroid of word vectors captures the dominant topics of a
      document while being simple and fast. More sophisticated approaches
      (e.g., SIF weighting, doc2vec) exist but averaging is a strong baseline.
    """
    words = text.split() if isinstance(text, str) else []
    vecs  = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def build_glove_features(texts, embeddings: dict, dim: int = 300) -> np.ndarray:
    """Build document embedding matrix for a list of texts."""
    from tqdm import tqdm
    return np.vstack([
        text_to_glove_vector(t, embeddings, dim)
        for t in tqdm(texts, desc="Building GloVe features")
    ])


# ── Feature Combination ───────────────────────────────────────────────────────

def combine_features(
    tfidf_matrix: csr_matrix,
    linguistic_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[csr_matrix, MinMaxScaler]:
    """
    Horizontally stack TF-IDF sparse matrix with dense feature DataFrames.

    TF-IDF values are already in [0, 1] range.
    Dense features (word counts, sentiment) need scaling to the same range.

    Args:
        tfidf_matrix:   Sparse TF-IDF matrix (N, V).
        linguistic_df:  Dense linguistic features (N, F1).
        sentiment_df:   Dense VADER sentiment features (N, 4).
        scaler:         Pre-fitted MinMaxScaler (for test set transformation).
        fit_scaler:     If True, fit a new scaler on the dense features.

    Returns:
        (combined_sparse_matrix, fitted_scaler)
    """
    dense = pd.concat([linguistic_df, sentiment_df], axis=1).values

    if fit_scaler:
        scaler = MinMaxScaler()
        dense = scaler.fit_transform(dense)
    elif scaler is not None:
        dense = scaler.transform(dense)
    else:
        raise ValueError("Provide a fitted scaler or set fit_scaler=True")

    dense_sparse = csr_matrix(dense)
    combined = hstack([tfidf_matrix, dense_sparse])

    logger.info(f"Combined feature matrix shape: {combined.shape}")
    return combined, scaler
