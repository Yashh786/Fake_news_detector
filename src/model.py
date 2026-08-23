"""
src/model.py
────────────────────────────────────────────────────────────────────────────────
ML and Deep Learning Models for Fake News Detection

Covers:
  - Classical ML: Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost
  - Model training, evaluation, hyperparameter search
  - Save / load utilities
  - PyTorch BERT fine-tuning (GPU-optimized for RTX 4060)
────────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

logger = logging.getLogger(__name__)

# ── Classical ML Models ───────────────────────────────────────────────────────

def get_all_models() -> Dict:
    """
    Return a dict of named, pre-configured sklearn-compatible classifiers.

    Why these models?
    ─────────────────
    LogisticRegression:
      Maps TF-IDF feature weights to a probability via sigmoid. Fast,
      interpretable (you can inspect feature coefficients), strong baseline.
      max_iter=1000 needed because high-dimensional TF-IDF takes more iterations.

    ComplementNB:
      Complement Naive Bayes — variant of Multinomial NB that works better
      for imbalanced text classification. Unlike MultinomialNB, it models
      P(¬class | features) for each class and picks the best complement.

    LinearSVC (calibrated):
      Support Vector Machine with linear kernel — one of the best classical
      models for high-dimensional sparse text data. LinearSVC doesn't produce
      probabilities natively, so we wrap it with CalibratedClassifierCV.

    RandomForest:
      Bagging ensemble of decision trees. Less effective than SVM on sparse
      TF-IDF but useful for combined dense+sparse features. Inherently
      provides feature importance scores.

    XGBoost:
      Gradient boosting — sequential trees that correct previous errors.
      tree_method='hist' is fast. use_label_encoder removed in XGBoost 2.x.
      Best when combined with engineered dense features.
    """
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
        ),
        "ComplementNB": ComplementNB(
            alpha=0.1,  # Laplace smoothing; small value works well for TF-IDF
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
            cv=3,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",       # Use RTX 4060 for XGBoost too
            eval_metric="logloss",
            random_state=42,
        ),
    }


def train_model(model, X_train, y_train):
    """Fit a model. Returns the trained model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "") -> Dict:
    """
    Full evaluation suite.

    Returns dict with:
      - accuracy, precision, recall, f1 (weighted)
      - roc_auc (binary or macro)
      - classification_report (str)
      - confusion_matrix (np.ndarray)
    """
    y_pred = model.predict(X_test)

    # AUC-ROC needs probability scores
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except AttributeError:
        auc = None

    report = classification_report(y_test, y_pred, target_names=["REAL", "FAKE"])
    cm     = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    results = {
        "model":                  model_name,
        "accuracy":               accuracy_score(y_test, y_pred),
        "precision":              precision_score(y_test, y_pred, average="weighted"),
        "recall":                 recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted":            f1_score(y_test, y_pred, average="weighted"),
        "roc_auc":                auc,
        "classification_report":  report,
        "confusion_matrix":       cm,
        "_preds":                 y_pred,
    }

    logger.info(f"\n{'='*60}\n{model_name} Results\n{'='*60}")
    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"F1 (wtd):  {results['f1_weighted']:.4f}")
    logger.info(f"ROC-AUC:   {results['roc_auc']:.4f}" if auc else "ROC-AUC: N/A")
    logger.info(f"\n{report}")

    return results


def cross_validate_model(model, X, y, cv: int = 5) -> np.ndarray:
    """
    Stratified K-Fold cross-validation.

    Why stratified?
      StratiifedKFold ensures each fold has the same class distribution as
      the full dataset — critical for imbalanced data.
    """
    from sklearn.model_selection import cross_val_score
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    logger.info(f"CV F1 scores: {scores.round(4)} | Mean: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def hyperparameter_search(model_name: str, X_train, y_train) -> object:
    """
    Grid search for best hyperparameters.

    Only runs for Logistic Regression and LinearSVC for speed.
    Add other models as needed.
    """
    param_grids = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"],
        },
        "LinearSVC": {
            "estimator__C": [0.1, 1.0, 10.0],
        },
    }

    if model_name not in param_grids:
        raise ValueError(f"No param grid for {model_name}. Choose from: {list(param_grids)}")

    base_models = get_all_models()
    model = base_models[model_name]
    grid  = GridSearchCV(
        model,
        param_grids[model_name],
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    logger.info(f"Best params for {model_name}: {grid.best_params_}")
    logger.info(f"Best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def build_voting_ensemble(trained_models: list) -> VotingClassifier:
    """
    Soft voting ensemble from a list of (name, model) tuples.
    Averages predicted probabilities from all models.
    """
    ensemble = VotingClassifier(estimators=trained_models, voting="soft")
    return ensemble


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, path: str):
    """Save a sklearn model using joblib (efficient for large numpy arrays)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str):
    """Load a saved sklearn model."""
    return joblib.load(path)


# ── PyTorch BERT Fine-tuning ──────────────────────────────────────────────────

class BERTDataset:
    """
    PyTorch Dataset for BERT tokenization.

    Takes raw texts + labels and tokenizes them on the fly.
    This is memory-efficient — tokenization happens per-batch,
    not all at once.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def fine_tune_bert(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    save_dir: str = "models/bert",
):
    """
    Fine-tune a pretrained BERT (or DistilBERT) model for binary classification.

    Model choice:
      - 'bert-base-uncased'        : Full BERT (12 layers, 110M params). ~99% accuracy.
      - 'distilbert-base-uncased'  : 40% smaller, 60% faster, retains 97% of BERT accuracy.
                                     Recommended for RTX 4060 8GB VRAM.

    Training setup:
      - AdamW optimizer with linear warmup scheduler (standard for transformers)
      - Mixed precision (fp16) via torch.amp for 2x speedup on RTX 4060
      - Gradient clipping to prevent exploding gradients

    Args:
        train_texts: List of raw training strings.
        train_labels: List of 0/1 labels.
        val_texts: Validation strings.
        val_labels: Validation labels.
        model_name: HuggingFace model identifier.
        epochs: Number of fine-tuning epochs (3 is usually enough).
        batch_size: Per-GPU batch size (16 fits in 8GB VRAM with max_length=256).
        lr: Learning rate (2e-5 is the sweet spot for BERT fine-tuning).
        max_length: Max token length (256 covers ~95% of news articles).
        save_dir: Directory to save final model and tokenizer.

    Returns:
        (model, tokenizer) — the fine-tuned model on GPU.
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.cuda.amp import GradScaler, autocast
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import accuracy_score, f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"BERT training on: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Tokenizer & Model ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    train_ds = BERTDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds   = BERTDataset(val_texts,   val_labels,   tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()  # Mixed precision

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast():                    # fp16 for RTX 4060 speedup
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average="weighted")
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"  ✓ Best model saved (F1={val_f1:.4f}) → {save_dir}")

    return model, tokenizer


def predict_with_bert(
    texts: list,
    model_dir: str,
    batch_size: int = 32,
    max_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a list of texts using a saved BERT model.

    Returns:
        (predictions array, probability array)
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    dummy_labels = [0] * len(texts)
    dataset      = BERTDataset(texts, dummy_labels, tokenizer, max_length)
    loader       = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)
