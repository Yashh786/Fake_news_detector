"""
src/utils.py
────────────────────────────────────────────────────────────────────────────────
Utility functions: logging setup, data loading, plotting, seeding.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import random
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with timestamp format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_kaggle_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Kaggle Fake News dataset.

    Expected files in data_dir:
      - train.csv : id, title, author, text, label (0=REAL, 1=FAKE)
      - test.csv  : id, title, author, text (no label — competition format)

    If only train.csv, we split it ourselves (80/10/10 train/val/test).

    Download instructions:
      1. Sign up at https://www.kaggle.com
      2. Go to https://www.kaggle.com/competitions/fake-news/data
      3. Accept rules and download train.csv
      4. Place in data/raw/train.csv
    """
    train_path = Path(data_dir) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_path}.\n"
            "Download from: https://www.kaggle.com/competitions/fake-news/data\n"
            "Place train.csv in data/raw/"
        )

    df = pd.read_csv(train_path)
    logging.getLogger(__name__).info(f"Loaded dataset: {df.shape[0]:,} rows")
    return df


def split_dataset(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split: train / val / test.

    Why stratified?
      Ensures each split has the same class ratio as the original dataset.
      Without stratification, you might get an unlucky split where one class
      dominates the test set, giving misleading metrics.
    """
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_col]
    )
    relative_val = val_size / (1 - test_size)
    train, val   = train_test_split(
        train_val, test_size=relative_val, random_state=random_state,
        stratify=train_val[label_col]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label",
                             save_path: Optional[str] = None):
    """Bar chart of class balance."""
    from typing import Optional
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[label_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars   = ax.bar(["REAL (0)", "FAKE (1)"], counts.values, color=colors, edgecolor="black")
    ax.bar_label(bars, padding=3, fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Confusion Matrix",
                          save_path: Optional[str] = None):
    """Annotated confusion matrix heatmap."""
    from typing import Optional
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels or ["REAL", "FAKE"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name: str = "", save_path: Optional[str] = None):
    """ROC curve with AUC annotation."""
    from typing import Optional
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return roc_auc


def plot_model_comparison(results: list, metric: str = "f1_weighted",
                          save_path: Optional[str] = None):
    """
    Horizontal bar chart comparing multiple models on a metric.

    Args:
        results: List of dicts from evaluate_model().
        metric:  Key to compare (e.g. 'accuracy', 'f1_weighted', 'roc_auc').
    """
    from typing import Optional
    names  = [r["model"] for r in results]
    values = [r[metric] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars    = ax.barh(names, values, color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_xlim(min(values) - 0.05, 1.01)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_tfidf_features(vectorizer, model, n: int = 20,
                             save_path: Optional[str] = None):
    """
    Show top N TF-IDF features (words) with highest positive/negative weights.

    Works with Logistic Regression (coef_) and LinearSVC.
    """
    from typing import Optional
    try:
        coef       = model.coef_[0] if hasattr(model, "coef_") else model.estimator.coef_[0]
    except AttributeError:
        print("Model doesn't expose feature coefficients.")
        return

    feature_names = vectorizer.get_feature_names_out()
    top_pos_idx   = np.argsort(coef)[-n:][::-1]   # FAKE indicators
    top_neg_idx   = np.argsort(coef)[:n]           # REAL indicators

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, indices, label, color in [
        (axes[0], top_pos_idx, "Top FAKE Indicators", "#e74c3c"),
        (axes[1], top_neg_idx, "Top REAL Indicators", "#2ecc71"),
    ]:
        words  = [feature_names[i] for i in indices]
        scores = [abs(coef[i]) for i in indices]
        ax.barh(words[::-1], scores[::-1], color=color, edgecolor="black")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("|Coefficient Weight|")
        sns.despine()

    plt.suptitle("Most Predictive TF-IDF Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Typing fix ────────────────────────────────────────────────────────────────
from typing import Optional  # noqa: E402 (needed for function signatures above)
