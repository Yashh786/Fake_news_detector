# %% [markdown]
# # Notebook 03 — Feature Extraction
#
# **Goal**: Convert cleaned text into numerical feature matrices.
#
# Three approaches:
# 1. **TF-IDF** — sparse vectors based on word importance (primary approach)
# 2. **Bag of Words** — simple term count vectors (baseline comparison)
# 3. **Handcrafted + Sentiment features** — dense features to stack on top of TF-IDF

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.features import (
    build_tfidf, build_bow,
    extract_linguistic_features,
    extract_sentiment_features,
    combine_features
)
from src.utils import setup_logging, set_seed, split_dataset

setup_logging()
set_seed(42)

# %% [markdown]
# ## 1. Load Cleaned Data & Split

# %%
df = pd.read_csv("../data/processed/cleaned.csv")
df = df.dropna(subset=["text_clean", "label"]).reset_index(drop=True)
print(f"Loaded: {len(df):,} articles")

# Stratified 80/10/10 split
train_df, val_df, test_df = split_dataset(df, label_col="label", test_size=0.10, val_size=0.10)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Train label distribution:\n{train_df['label'].value_counts()}")

# %% [markdown]
# ## 2. TF-IDF Feature Extraction
#
# ### Theory Recap
# $$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}$$
# $$IDF(t) = \log\left(\frac{1 + N}{1 + df(t)}\right) + 1$$
# $$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$
#
# **Parameters we use:**
# - `max_features=50000`: Keep top 50k most informative terms
# - `ngram_range=(1,2)`: Unigrams + bigrams (2-word phrases)
# - `sublinear_tf=True`: Log-scale TF dampening
# - `min_df=2`: Ignore terms in < 2 documents
# - `max_df=0.95`: Ignore terms in > 95% of documents

# %%
X_train_tfidf, X_test_tfidf, tfidf_vec = build_tfidf(
    train_df["text_clean"],
    test_df["text_clean"],
    max_features=50_000,
    ngram_range=(1, 2),
    save_path="../models/tfidf_vectorizer.pkl",
)
_, X_val_tfidf, _ = build_tfidf(
    train_df["text_clean"],
    val_df["text_clean"],
    max_features=50_000,
    ngram_range=(1, 2),
)
# Refit with same vocab
X_val_tfidf = tfidf_vec.transform(val_df["text_clean"])

print(f"\nTF-IDF Matrix Shapes:")
print(f"  Train: {X_train_tfidf.shape}  (articles × vocabulary)")
print(f"  Val:   {X_val_tfidf.shape}")
print(f"  Test:  {X_test_tfidf.shape}")
print(f"\nSparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.2f}%")
print("(High sparsity is normal — each article only uses a tiny subset of the vocabulary)")

# %% [markdown]
# ## 3. Visualize TF-IDF: Top Terms

# %%
feature_names = tfidf_vec.get_feature_names_out()
mean_tfidf    = np.asarray(X_train_tfidf.mean(axis=0)).flatten()
top_idx       = mean_tfidf.argsort()[-20:][::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([feature_names[i] for i in top_idx][::-1],
        [mean_tfidf[i] for i in top_idx][::-1],
        color="#3498db", edgecolor="black")
ax.set_title("Top 20 Terms by Mean TF-IDF Score (Training Set)", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean TF-IDF Score")
plt.tight_layout()
plt.savefig("../reports/figures/06_top_tfidf_terms.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Bag-of-Words (Baseline Comparison)

# %%
X_train_bow, X_test_bow, bow_vec = build_bow(
    train_df["text_clean"],
    test_df["text_clean"],
    max_features=50_000,
)
X_val_bow = bow_vec.transform(val_df["text_clean"])
print(f"BoW Matrix: Train={X_train_bow.shape}, Val={X_val_bow.shape}, Test={X_test_bow.shape}")

# %% [markdown]
# ## 5. Handcrafted Linguistic Features
#
# These capture structural signals that TF-IDF misses:
# - Punctuation density (fake news often has more `!` marks)
# - Capital letter ratio (SHOUTING IS A FAKE NEWS TELL)
# - Vocabulary richness (unique_word_ratio)
# - Average sentence length

# %%
print("Extracting linguistic features...")
train_ling = extract_linguistic_features(train_df["combined"] if "combined" in train_df.columns else train_df["text"])
val_ling   = extract_linguistic_features(val_df["combined"] if "combined" in val_df.columns else val_df["text"])
test_ling  = extract_linguistic_features(test_df["combined"] if "combined" in test_df.columns else test_df["text"])

print(f"Linguistic feature columns: {list(train_ling.columns)}")
train_ling.head(3)

# %% [markdown]
# ### Visualize: Do linguistic features differ between REAL and FAKE?

# %%
ling_with_labels = train_ling.copy()
ling_with_labels["label"] = train_df["label"].values

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

features_to_plot = [
    "exclamation_count", "capital_ratio", "unique_word_ratio",
    "avg_word_length", "avg_sentence_length", "punctuation_ratio"
]

for ax, feat in zip(axes, features_to_plot):
    for label, color, name in [(0, "#2ecc71", "REAL"), (1, "#e74c3c", "FAKE")]:
        data = ling_with_labels[ling_with_labels["label"] == label][feat]
        ax.hist(data.clip(data.quantile(0.01), data.quantile(0.99)),
                bins=40, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

plt.suptitle("Linguistic Feature Distributions: REAL vs FAKE", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/figures/07_linguistic_features.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. VADER Sentiment Features

# %%
print("Extracting VADER sentiment features...")
# Use original (uncleaned) text for sentiment — sentiment words may be stripped by preprocessing
train_sent = extract_sentiment_features(
    train_df["text"] if "text" in train_df.columns else train_df["text_clean"])
val_sent   = extract_sentiment_features(
    val_df["text"] if "text" in val_df.columns else val_df["text_clean"])
test_sent  = extract_sentiment_features(
    test_df["text"] if "text" in test_df.columns else test_df["text_clean"])

print(f"Sample sentiment scores:\n{train_sent.head(3)}")

# Visualize compound score distribution
fig, ax = plt.subplots(figsize=(9, 4))
train_sent_labeled = train_sent.copy()
train_sent_labeled["label"] = train_df["label"].values

for label, color, name in [(0, "#2ecc71", "REAL"), (1, "#e74c3c", "FAKE")]:
    data = train_sent_labeled[train_sent_labeled["label"] == label]["vader_compound"]
    ax.hist(data, bins=50, alpha=0.6, color=color, label=name, density=True)

ax.set_xlabel("VADER Compound Sentiment Score")
ax.set_ylabel("Density")
ax.set_title("Sentiment Distribution: REAL vs FAKE News", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/08_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Combined Feature Matrix (TF-IDF + Linguistic + Sentiment)

# %%
from sklearn.preprocessing import MinMaxScaler

X_train_combined, scaler = combine_features(
    X_train_tfidf, train_ling, train_sent, fit_scaler=True
)
X_val_combined,  _      = combine_features(X_val_tfidf,   val_ling,   val_sent,  scaler=scaler)
X_test_combined, _      = combine_features(X_test_tfidf,  test_ling,  test_sent, scaler=scaler)

print(f"Combined feature matrix shape: {X_train_combined.shape}")
print(f"  (TF-IDF: 50,000 + Linguistic: 10 + Sentiment: 4 = 50,014 features)")

# %% [markdown]
# ## 8. Save Feature Matrices & Labels

# %%
import scipy.sparse as sp
import pickle

# Save sparse matrices
sp.save_npz("../data/processed/X_train_tfidf.npz", X_train_tfidf)
sp.save_npz("../data/processed/X_val_tfidf.npz",   X_val_tfidf)
sp.save_npz("../data/processed/X_test_tfidf.npz",  X_test_tfidf)

sp.save_npz("../data/processed/X_train_combined.npz", X_train_combined)
sp.save_npz("../data/processed/X_val_combined.npz",   X_val_combined)
sp.save_npz("../data/processed/X_test_combined.npz",  X_test_combined)

# Save labels
y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

np.save("../data/processed/y_train.npy", y_train)
np.save("../data/processed/y_val.npy",   y_val)
np.save("../data/processed/y_test.npy",  y_test)

# Save scaler
with open("../models/feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save split dataframes for text access in BERT notebook
train_df.to_csv("../data/processed/train_split.csv", index=False)
val_df.to_csv("../data/processed/val_split.csv",   index=False)
test_df.to_csv("../data/processed/test_split.csv",  index=False)

print("✓ All feature matrices and labels saved!")

# %% [markdown]
# ## Summary
#
# | Feature Set | Shape | Use Case |
# |-------------|-------|----------|
# | TF-IDF (1,2)-grams | (N, 50,000) | Primary for classical ML |
# | BoW | (N, 50,000) | Baseline comparison |
# | Combined (TF-IDF + Ling + Sentiment) | (N, 50,014) | Best for ensemble models |
#
# **Next → Notebook 04: ML Model Training**
