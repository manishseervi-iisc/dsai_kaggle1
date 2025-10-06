# ==============================
# Imports and Data Loading
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

# Load data (edit paths as needed)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

# ==============================
# Basic Setup
# ==============================
# Replace with your actual feature list
FEATURES = [col for col in train.columns if col not in ["id", "song_popularity", "fold"]]

# Identify categorical features if any
cat_features = [col for col in FEATURES if train[col].dtype == "object"]

# Number of folds for cross-validation
N_FOLDS = 5

# Out-of-fold and test predictions
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# ==============================
# CatBoost Parameters
# ==============================
cat_params = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": 42,
    "early_stopping_rounds": 50,
    "verbose": 100,
    "use_best_model": True
}

# ==============================
# Stratified K-Fold Training
# ==============================
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, train["song_popularity"])):
    print(f"\nTraining fold {fold+1}/{N_FOLDS}")

    X_tr, y_tr = train.loc[tr_idx, FEATURES], train.loc[tr_idx, "song_popularity"]
    X_va, y_va = train.loc[va_idx, FEATURES], train.loc[va_idx, "song_popularity"]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_va, y_va, cat_features=cat_features)

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool)

    # OOF predictions
    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]

    # Test predictions (average across folds)
    test_preds += model.predict_proba(test[FEATURES])[:, 1] / N_FOLDS

    # Fold AUC
    fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")

# ==============================
# Overall Evaluation
# ==============================
overall_auc = roc_auc_score(train["song_popularity"], oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.6f}")

# ==============================
# Submission File
# ==============================
submission = sample.copy()
submission["song_popularity"] = test_preds
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")
