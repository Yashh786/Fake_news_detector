"""
src/__init__.py — package marker
"""
from .preprocessing import clean_text, preprocess_series, combine_title_text
from .features import build_tfidf, extract_linguistic_features, extract_sentiment_features
from .model import get_all_models, train_model, evaluate_model, save_model, load_model
from .utils import setup_logging, set_seed, load_kaggle_dataset, split_dataset

__all__ = [
    "clean_text", "preprocess_series", "combine_title_text",
    "build_tfidf", "extract_linguistic_features", "extract_sentiment_features",
    "get_all_models", "train_model", "evaluate_model", "save_model", "load_model",
    "setup_logging", "set_seed", "load_kaggle_dataset", "split_dataset",
]
