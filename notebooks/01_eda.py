# %% [markdown]
# # Notebook 01 — Exploratory Data Analysis (EDA)
#
# **Goal**: Understand the dataset before building any model.
# Key questions we answer here:
#   - How many samples? What's the class balance?
#   - What does the text look like (length, vocabulary)?
#   - What words distinguish fake vs. real news?
#
# Run each cell (Ctrl+Enter in VS Code with Jupyter extension, or shift+Enter in Jupyter Lab)

# %% [markdown]
# ## 1. Imports & Setup

# %%
import sys
sys.path.insert(0, "..")  # Allow imports from src/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Project modules
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import setup_logging, load_kaggle_dataset, plot_class_distribution

setup_logging()
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# %% [markdown]
# ## 2. Load Dataset
#
# **Before running**: Download `train.csv` from:
# https://www.kaggle.com/competitions/fake-news/data
# Place it in `data/raw/train.csv`
#
# Columns:
# | Column | Description |
# |--------|-------------|
# | id     | Unique article ID |
# | title  | Article headline |
# | author | Author name |
# | text   | Full article body |
# | label  | 0 = REAL, 1 = FAKE |

df = load_kaggle_dataset("../data/raw")
df["label"] = df["label"].replace({"real": 0, "fake": 1})
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
df.head(3)

# %% [markdown]
# ## 3. Basic Statistics

# %%
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Total articles  : {len(df):,}")
print(f"Missing values  :\n{df.isnull().sum()}")
print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nFake news %: {(df['label']==1).mean()*100:.1f}%")
print(f"Real news %: {(df['label']==0).mean()*100:.1f}%")

# %%
# Fill missing values
df["title"]  = df["title"].fillna("")
df["author"] = df["author"].fillna("Unknown")
df["text"]   = df["text"].fillna("")

# Combine title + text for analysis
df["full_text"] = df["title"] + " " + df["text"]
df["text_length"] = df["full_text"].str.split().str.len()

print("Missing values after fill:")
print(df[["title","author","text"]].isnull().sum())

# %% [markdown]
# ## 4. Class Distribution Plot

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Class counts
counts = df["label"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
bars   = axes[0].bar(["REAL (0)", "FAKE (1)"], counts.values, color=colors, edgecolor="black", linewidth=1.2)
axes[0].bar_label(bars, padding=3, fontsize=12, fontweight="bold")
axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Article Count")

# Pie chart
axes[1].pie(counts.values, labels=["REAL", "FAKE"], colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Balance", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("../reports/figures/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Text Length Analysis
#
# **Why this matters**: Fake news articles are often shorter (less research)
# or very long (flooding with detail to seem credible). The distribution
# reveals if length is a useful feature.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label, color, name in [
    (axes[0], 0, "#2ecc71", "REAL"),
    (axes[1], 1, "#e74c3c", "FAKE"),
]:
    lengths = df[df["label"] == label]["text_length"]
    ax.hist(lengths.clip(0, 2000), bins=50, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(lengths.median(), color="black", linestyle="--", linewidth=1.5,
               label=f"Median: {lengths.median():.0f}")
    ax.set_title(f"{name} News — Text Length Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.savefig("../reports/figures/02_text_length.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"REAL news avg length: {df[df['label']==0]['text_length'].mean():.0f} words")
print(f"FAKE news avg length: {df[df['label']==1]['text_length'].mean():.0f} words")

# %% [markdown]
# ## 6. Word Clouds
#
# Word clouds visualize word frequency — larger = more frequent.
# Comparing REAL vs FAKE word clouds shows what language patterns differ.

# %%
from src.preprocessing import clean_text

# Get text per class
real_text = " ".join(df[df["label"] == 0]["full_text"].dropna().tolist())
fake_text = " ".join(df[df["label"] == 1]["full_text"].dropna().tolist())

# Clean (remove stopwords)
print("Preprocessing for word clouds (this takes ~1 min)...")
real_clean = clean_text(real_text[:200_000])  # Sample to speed up
fake_clean = clean_text(fake_text[:200_000])

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, text, title, color in [
    (axes[0], real_clean, "REAL News — Most Frequent Words", "Greens"),
    (axes[1], fake_clean, "FAKE News — Most Frequent Words", "Reds"),
]:
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=color,
        max_words=100,
        collocations=False,
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("../reports/figures/03_wordclouds.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Top Discriminating Words
#
# What words appear MUCH more in fake vs real news?
# We compute the ratio: fake_freq / real_freq for each word.

# %%
def get_word_counts(texts, n_top=500):
    from collections import Counter
    all_words = " ".join(texts).split()
    return Counter(all_words).most_common(n_top)

real_words = dict(get_word_counts(df[df["label"]==0]["full_text"].apply(clean_text)))
fake_words = dict(get_word_counts(df[df["label"]==1]["full_text"].apply(clean_text)))

# Words that appear heavily in fake but not real
fake_biased = {
    w: fake_words.get(w, 0) / (real_words.get(w, 1))
    for w in fake_words
    if w in real_words and len(w) > 3
}
top_fake_words = sorted(fake_biased.items(), key=lambda x: x[1], reverse=True)[:15]

fig, ax = plt.subplots(figsize=(9, 5))
words, ratios = zip(*top_fake_words)
ax.barh(list(words)[::-1], list(ratios)[::-1], color="#e74c3c", edgecolor="black")
ax.set_xlabel("Fake/Real Frequency Ratio", fontsize=11)
ax.set_title("Words Most Associated with FAKE News", fontsize=13, fontweight="bold")
ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Equal frequency")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/04_discriminating_words.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Author Analysis

# %%
print("\nTop 10 authors in FAKE news:")
print(df[df["label"]==1]["author"].value_counts().head(10))

print("\nTop 10 authors in REAL news:")
print(df[df["label"]==0]["author"].value_counts().head(10))

# Proportion of fake articles per author (authors with > 5 articles)
author_stats = df.groupby("author").agg(
    total=("label", "count"),
    fake_count=("label", "sum"),
).query("total > 5")
author_stats["fake_rate"] = author_stats["fake_count"] / author_stats["total"]

print("\nAuthors with highest fake rate (min 5 articles):")
print(author_stats.sort_values("fake_rate", ascending=False).head(10))

# %% [markdown]
# ## 9. Save Processed Data

# %%
# Save the cleaned dataframe for use in subsequent notebooks
df.to_csv("../data/processed/data_with_features.csv", index=False)
print("✓ Saved to data/processed/data_with_features.csv")

# %% [markdown]
# ## 10. EDA Summary
#
# | Finding | Implication |
# |---------|------------|
# | ~50/50 class balance | No need for class weighting or oversampling |
# | Fake news often shorter OR very long | text_length is a useful feature |
# | Fake news uses emotional, polarizing words | Sentiment analysis will help |
# | Some authors are consistently fake | Author encoding could help |
#
# Next step → **Notebook 02: Text Preprocessing**
