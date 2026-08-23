# %% [markdown]
# # Notebook 02 — Text Preprocessing
#
# **Goal**: Apply the full NLP preprocessing pipeline and visualize its effect.
# Build the cleaned dataset that feeds into all subsequent notebooks.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import clean_text, preprocess_series, combine_title_text, spacy_preprocess
from src.utils import setup_logging, set_seed

setup_logging()
set_seed(42)

# %% [markdown]
# ## 1. Load Data

# %%
df = pd.read_csv("../data/processed/data_with_features.csv")
print(f"Loaded {len(df):,} articles")
df.head(2)

# %% [markdown]
# ## 2. Preprocessing Pipeline — Step by Step Walkthrough
#
# Let's trace what happens to a sample fake news article at each step.

# %%
sample = df[df["label"]==1]["text"].dropna().iloc[0][:500]
print("ORIGINAL TEXT:")
print(sample)
print()

import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

steps = {}

# Step 1: Lowercase
s1 = sample.lower()
steps["1. Lowercase"] = s1

# Step 2: Remove URLs
s2 = re.sub(r"https?://\S+|www\.\S+", " ", s1)
steps["2. Remove URLs"] = s2

# Step 3: Remove HTML
s3 = re.sub(r"<[^>]+>", " ", s2)
steps["3. Remove HTML"] = s3

# Step 4: Remove special chars
s4 = re.sub(r"[^a-zA-Z\s]", " ", s3)
steps["4. Remove Punctuation/Digits"] = s4

# Step 5: Tokenize
s5 = word_tokenize(s4)
steps["5. Tokenization"] = str(s5[:15]) + " ..."

# Step 6: Remove stopwords
stop_words = set(stopwords.words("english"))
s6 = [w for w in s5 if w not in stop_words]
steps["6. Remove Stopwords"] = str(s6[:15]) + " ..."

# Step 7: Lemmatize
lem = WordNetLemmatizer()
s7 = [lem.lemmatize(w) for w in s6]
steps["7. Lemmatization"] = " ".join(s7[:15]) + " ..."

for step, result in steps.items():
    print(f"{'─'*60}")
    print(f"  {step}")
    print(f"  {result[:200]}")
    print()

# %% [markdown]
# ## 3. Apply to Full Dataset

# %%
# Combine title + text (title gets 3x weight for TF-IDF importance)
print("Combining title + text...")
df["combined"] = df.apply(
    lambda row: combine_title_text(
        str(row.get("title", "")),
        str(row.get("text", ""))
    ), axis=1
)

# Apply preprocessing pipeline
print("\nApplying preprocessing pipeline...")
df["text_clean"] = preprocess_series(df["combined"].tolist())
print(f"✓ Done. Sample cleaned text:\n  {df['text_clean'].iloc[0][:200]}")

# %% [markdown]
# ## 4. Before vs After: Length Comparison

# %%
df["clean_length"] = df["text_clean"].str.split().str.len()
df["orig_length"]  = df["combined"].str.split().str.len()
df["compression_ratio"] = df["clean_length"] / df["orig_length"].clip(lower=1)

print(f"Average compression ratio: {df['compression_ratio'].mean():.2f}")
print(f"(Original text reduced to {df['compression_ratio'].mean()*100:.0f}% of original size)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["orig_length"].clip(0, 1500), bins=50, alpha=0.6, label="Original", color="#3498db")
ax.hist(df["clean_length"].clip(0, 1500), bins=50, alpha=0.6, label="After Preprocessing", color="#e67e22")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.set_title("Text Length: Before vs After Preprocessing", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/05_preprocessing_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Empty Text Check
#
# After aggressive preprocessing, some very short articles may become empty strings.

# %%
empty_mask = df["text_clean"].str.strip() == ""
print(f"Empty after preprocessing: {empty_mask.sum()} articles ({empty_mask.mean()*100:.2f}%)")

# Remove empty rows
df = df[~empty_mask].reset_index(drop=True)
print(f"Dataset after removing empty: {len(df):,} articles")

# %% [markdown]
# ## 6. spaCy Advanced Preprocessing (Optional)
#
# spaCy uses POS tagging and NER — more accurate lemmatization and entity handling.
# Takes ~5-10x longer but can improve model accuracy.

# %%
# Only run this if you want the advanced version
APPLY_SPACY = False  # Set to True to enable

if APPLY_SPACY:
    print("Applying spaCy preprocessing (this may take 10-20 min for full dataset)...")
    # For demonstration, apply to first 100 samples
    sample_results = [spacy_preprocess(t) for t in df["text_clean"].head(100)]
    print(f"spaCy sample: {sample_results[0][:200]}")

# %% [markdown]
# ## 7. Save Processed Dataset

# %%
save_cols = ["id", "title", "author", "text", "combined", "text_clean",
             "text_length", "clean_length", "label"]
save_cols = [c for c in save_cols if c in df.columns]

df[save_cols].to_csv("../data/processed/cleaned.csv", index=False)
print(f"✓ Saved cleaned dataset → data/processed/cleaned.csv")
print(f"  Shape: {df[save_cols].shape}")

# %% [markdown]
# ## Next → Notebook 03: Feature Extraction (TF-IDF + Embeddings)
