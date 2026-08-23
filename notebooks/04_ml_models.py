# %% [markdown]
# # Notebook 04 — Classical ML Models
#
# **Goal**: Train and compare 5 classifiers on TF-IDF features.
# Build a leaderboard, tune the best model, and create a soft-voting ensemble.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.model import (
    get_all_models, train_model, evaluate_model,
    cross_validate_model, build_voting_ensemble, save_model
)
from src.utils import (
    setup_logging, set_seed,
    plot_confusion_matrix, plot_roc_curve,
    plot_model_comparison, plot_top_tfidf_features
)
import pickle

setup_logging()
set_seed(42)

# %% [markdown]
# ## 1. Load Feature Matrices & Labels

# %%
print("Loading feature matrices...")
X_train = sp.load_npz("../data/processed/X_train_tfidf.npz")
X_val   = sp.load_npz("../data/processed/X_val_tfidf.npz")
X_test  = sp.load_npz("../data/processed/X_test_tfidf.npz")
y_train = np.load("../data/processed/y_train.npy")
y_val   = np.load("../data/processed/y_val.npy")
y_test  = np.load("../data/processed/y_test.npy")

with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vec = pickle.load(f)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"y_train counts: REAL={( y_train==0).sum()}, FAKE={(y_train==1).sum()}")

# Also load combined features
X_train_comb = sp.load_npz("../data/processed/X_train_combined.npz")
X_val_comb   = sp.load_npz("../data/processed/X_val_combined.npz")
X_test_comb  = sp.load_npz("../data/processed/X_test_combined.npz")

# %% [markdown]
# ## 2. Train All Models on TF-IDF Features
#
# We train on (train + val) for final evaluation, but first validate on val set.

# %%
models = get_all_models()
results_tfidf = []

for name, model in tqdm(models.items(), desc="Training models"):
    print(f"\n{'='*55}\nTraining: {name}")

    # Note: ComplementNB requires non-negative features → TF-IDF is fine (always ≥ 0)
    # XGBoost with device='cuda' might fail on sparse matrices — convert if needed
    if name == "XGBoost":
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": "cuda",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "seed": 42
        }
        bst = xgb.train(params, dtrain, num_boost_round=300,
                        evals=[(dval, "val")], verbose_eval=50,
                        early_stopping_rounds=30)
        y_pred_val = (bst.predict(dval) > 0.5).astype(int)

        from sklearn.metrics import f1_score, accuracy_score
        val_result = {
            "model": name,
            "accuracy": accuracy_score(y_val, y_pred_val),
            "f1_weighted": f1_score(y_val, y_pred_val, average="weighted"),
            "roc_auc": None,
            "_xgb_booster": bst,
        }
        results_tfidf.append(val_result)
        save_model(bst, f"../models/{name.lower()}_tfidf.pkl")
        continue

    trained = train_model(model, X_train, y_train)
    result  = evaluate_model(trained, X_val, y_val, model_name=name)
    results_tfidf.append(result)
    save_model(trained, f"../models/{name.lower().replace(' ', '_')}_tfidf.pkl")

# %% [markdown]
# ## 3. Model Leaderboard

# %%
leaderboard = pd.DataFrame([
    {
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "F1-Score": f"{r['f1_weighted']:.4f}",
        "AUC-ROC":  f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A",
    }
    for r in results_tfidf
]).sort_values("F1-Score", ascending=False)

print("\n" + "="*55)
print("MODEL LEADERBOARD — TF-IDF Features")
print("="*55)
print(leaderboard.to_string(index=False))

# %%
plot_model_comparison(
    [r for r in results_tfidf if r.get("roc_auc")],
    metric="f1_weighted",
    save_path="../reports/figures/09_model_comparison.png"
)

# %% [markdown]
# ## 4. Detailed Analysis of Best Model

# %%
# Automatically pick best by F1
best_result = sorted([r for r in results_tfidf if r.get("confusion_matrix") is not None],
                     key=lambda x: x["f1_weighted"], reverse=True)[0]
best_name   = best_result["model"]
print(f"\nBest model: {best_name}")
print(f"\nClassification Report:\n{best_result['classification_report']}")

# %%
# Confusion matrix
plot_confusion_matrix(
    y_val,
    [r for r in results_tfidf if r["model"] == best_name][0].get("_preds", []),
    title=f"Confusion Matrix — {best_name}",
    save_path=f"../reports/figures/10_confusion_{best_name}.png"
)

# %% [markdown]
# ## 5. ROC Curves for All Models

# %%
fig, ax = plt.subplots(figsize=(8, 6))
from sklearn.metrics import roc_curve, auc

for result, model_file in zip(results_tfidf, [
    f"../models/{r['model'].lower().replace(' ','_')}_tfidf.pkl" for r in results_tfidf
]):
    if not result.get("roc_auc"):
        continue
    try:
        from src.model import load_model
        m = load_model(model_file)
        y_proba = m.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{result['model']} (AUC={roc_auc:.4f})")
    except Exception:
        pass

ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("../reports/figures/11_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. TF-IDF Feature Importance
#
# For Logistic Regression, the coefficients tell us which words most strongly
# predict FAKE vs REAL. This is a key interpretability advantage of linear models.

# %%
lr_model = load_model("../models/logisticregression_tfidf.pkl")
plot_top_tfidf_features(
    tfidf_vec, lr_model, n=20,
    save_path="../reports/figures/12_feature_importance.png"
)

# %% [markdown]
# ## 7. Cross-Validation on Best Model
#
# Cross-validation gives a more reliable performance estimate than a single train/val split.
# With 5-fold CV, we train 5 models and average their scores.

# %%
from sklearn.linear_model import LogisticRegression

print("\nRunning 5-fold Stratified Cross-Validation on Logistic Regression...")
lr_cv = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42)
import scipy.sparse as sp_lib
X_all = sp_lib.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])

cv_scores = cross_validate_model(lr_cv, X_all, y_all, cv=5)
print(f"CV F1 Scores: {cv_scores.round(4)}")
print(f"Mean ± Std:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %% [markdown]
# ## 8. Soft Voting Ensemble
#
# Combines predictions from multiple models — often beats any single model.
# "Soft voting" averages predicted probabilities (rather than majority vote).

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB

lr_m  = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, random_state=42)
cnb_m = ComplementNB(alpha=0.1)
svc_m = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3)

ensemble = build_voting_ensemble([("lr", lr_m), ("cnb", cnb_m), ("svc", svc_m)])
ensemble.fit(X_train, y_train)

ens_result = evaluate_model(ensemble, X_val, y_val, model_name="Voting Ensemble")
print(f"\nEnsemble F1: {ens_result['f1_weighted']:.4f}")
save_model(ensemble, "../models/voting_ensemble.pkl")

# %% [markdown]
# ## 9. Final Evaluation on Test Set
#
# **IMPORTANT**: Only run this ONCE — after you've finalized model selection.
# Peeking at test set during development leads to overfitting to the test distribution.

# %%
RUN_TEST_EVAL = True  # Set to False to preserve test set integrity

if RUN_TEST_EVAL:
    best_model = load_model("../models/logisticregression_tfidf.pkl")  # or your best model
    test_result = evaluate_model(best_model, X_test, y_test, model_name="Best Model (Test Set)")
    print(f"\n🎯 FINAL TEST SET RESULTS")
    print(f"   Accuracy:  {test_result['accuracy']:.4f}")
    print(f"   F1-Score:  {test_result['f1_weighted']:.4f}")
    print(f"   ROC-AUC:   {test_result['roc_auc']:.4f}" if test_result.get("roc_auc") else "")

# %% [markdown]
# ## Summary
#
# | Model | Expected F1 | Speed |
# |-------|------------|-------|
# | Logistic Regression | ~0.97 | ⚡ Fast |
# | Complement NB | ~0.94 | ⚡ Fastest |
# | Linear SVC | ~0.97 | ⚡ Fast |
# | Random Forest | ~0.93 | 🐢 Slow |
# | XGBoost | ~0.96 | ⚡ Fast (GPU) |
# | Voting Ensemble | ~0.98 | ⚡ Fast |
#
# **Next → Notebook 05: BERT Fine-tuning**
