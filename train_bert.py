"""
train_bert.py — Fine-tune DistilBERT for fake news detection
─────────────────────────────────────────────────────────────
Run from the project root:
    python train_bert.py

Configuration (edit the CONSTANTS block below):
  - N_TRAIN     : number of training rows to sample
  - N_VAL       : number of validation rows to sample
  - EPOCHS      : fine-tuning epochs (2-3 is usually enough)
  - BATCH_SIZE  : per-device batch size (8 works well on CPU)
  - MAX_LENGTH  : max token length (128 is fast on CPU; 256 for quality)
  - LR          : AdamW learning rate (2e-5 is standard for BERT)
  - SAVE_DIR    : output directory for model weights and tokenizer
"""

import os, sys, time, pathlib
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Configuration ─────────────────────────────────────────────────────────────
N_TRAIN    = 20_000  # rows sampled from train split
N_VAL      = 2_000   # rows sampled from val split
EPOCHS     = 3
BATCH_SIZE = 16      # GPU: 16 fits easily in 8.6GB VRAM with max_length=256
MAX_LENGTH = 256     # better context quality on GPU
LR         = 2e-5
TEXT_COL   = "combined"   # raw title+article text — BERT needs this, NOT text_clean
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR   = pathlib.Path("data/processed")
SAVE_DIR   = pathlib.Path("models/bert_finetuned")
SEED       = 42

# ── Setup ─────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # fp16 only on GPU

print("\n" + "="*60)
print("  DistilBERT Fine-tuning - Fake News Detector")
print("="*60)
print(f"  Device      : {device}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  Train rows  : {N_TRAIN:,}")
print(f"  Val rows    : {N_VAL:,}")
print(f"  Epochs      : {EPOCHS}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Max length  : {MAX_LENGTH}")
print(f"  Save dir    : {SAVE_DIR}")
print("="*60 + "\n")

# ── Load & sample data ────────────────────────────────────────────────────────
print("Loading data splits...")
train_df = pd.read_csv(DATA_DIR / "train_split.csv").dropna(subset=[TEXT_COL, "label"])
val_df   = pd.read_csv(DATA_DIR / "val_split.csv").dropna(subset=[TEXT_COL, "label"])

# Stratified sample to keep class balance (compatible with all pandas versions)
def stratified_sample(df, n_per_class, seed):
    parts = []
    for lbl in df["label"].unique():
        sub = df[df["label"] == lbl]
        parts.append(sub.sample(min(len(sub), n_per_class), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)

train_df = stratified_sample(train_df, N_TRAIN // 2, SEED)
val_df   = stratified_sample(val_df,   N_VAL   // 2, SEED)

train_texts  = train_df[TEXT_COL].astype(str).tolist()
train_labels = train_df["label"].astype(int).tolist()
val_texts    = val_df[TEXT_COL].astype(str).tolist()
val_labels   = val_df["label"].astype(int).tolist()

print(f"Train: {len(train_texts):,}  (REAL={train_labels.count(0):,}, FAKE={train_labels.count(1):,})")
print(f"Val  : {len(val_texts):,}  (REAL={val_labels.count(0):,}, FAKE={val_labels.count(1):,})\n")


# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Tokenizer & Model ─────────────────────────────────────────────────────────
print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}\n")

# ── DataLoaders ───────────────────────────────────────────────────────────────
# num_workers=0 is required on Windows to avoid multiprocessing pickling issues
train_loader = DataLoader(
    NewsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
)
val_loader = DataLoader(
    NewsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH),
    batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
)

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
total_steps = len(train_loader) * EPOCHS
optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(0.1 * total_steps)),
    num_training_steps=total_steps,
)

if USE_AMP:
    scaler = torch.amp.GradScaler("cuda")

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_f1  = 0.0
best_val_acc = 0.0

secs_per_batch_est = 3.0  # conservative CPU estimate per batch
print(f"Starting training — {total_steps:,} total steps")
print(f"Estimated time per epoch on CPU: ~{len(train_loader) * secs_per_batch_est / 60:.0f} minutes\n")

for epoch in range(EPOCHS):
    t0         = time.time()
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_b       = batch["label"].to(device)

        optimizer.zero_grad()

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_b)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_b)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        # Progress every 10 batches
        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            elapsed  = time.time() - t0
            avg_loss = total_loss / (i + 1)
            pct      = (i + 1) / len(train_loader) * 100
            eta_s    = elapsed / (i + 1) * (len(train_loader) - i - 1)
            print(
                f"  Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {i+1}/{len(train_loader)} ({pct:5.1f}%) | "
                f"Loss: {avg_loss:.4f} | "
                f"ETA: {int(eta_s//60)}m {int(eta_s%60):02d}s",
                flush=True,
            )

    # Validation
    model.eval()
    all_preds, all_labels_val = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b       = batch["label"]
            out            = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = out.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_val.extend(labels_b.numpy())

    val_acc = accuracy_score(all_labels_val, all_preds)
    val_f1  = f1_score(all_labels_val, all_preds, average="weighted")
    elapsed = time.time() - t0
    is_best = val_f1 > best_val_f1

    print(f"\n" + "-"*60)
    print(f"  Epoch {epoch+1} done in {int(elapsed//60)}m {int(elapsed%60):02d}s")
    print(f"  Train Loss : {total_loss/len(train_loader):.4f}")
    print(f"  Val Acc    : {val_acc:.4f}  |  Val F1: {val_f1:.4f}  {'<-- BEST' if is_best else ''}")

    if is_best:
        best_val_f1  = val_f1
        best_val_acc = val_acc
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  Saved best model -> {SAVE_DIR}")

    print("-"*60 + "\n")

# ── Final report ──────────────────────────────────────────────────────────────
print("="*60)
print("  Training complete")
print(f"  Best Val Acc : {best_val_acc:.4f}")
print(f"  Best Val F1  : {best_val_f1:.4f}")
print(f"  Model saved  : {SAVE_DIR}")
print("="*60 + "\n")

print("Loading best checkpoint for final classification report...")
best_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
best_model.eval()

all_preds_f, all_labels_f = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_b       = batch["label"]
        out            = best_model(input_ids=input_ids, attention_mask=attention_mask)
        preds          = out.logits.argmax(dim=-1)
        all_preds_f.extend(preds.cpu().numpy())
        all_labels_f.extend(labels_b.numpy())

print(classification_report(all_labels_f, all_preds_f, target_names=["REAL", "FAKE"]))
print(f"Model is ready at: {SAVE_DIR.resolve()}")
