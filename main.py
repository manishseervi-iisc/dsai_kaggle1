

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

train_df = pd.read_csv("train.csv")  # training data
test_df = pd.read_csv("test.csv")    # test data for predictions
sample_sub = pd.read_csv("sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Feature selection - excluding ID and target
feature_cols = []
for col in train_df.columns:
    if col not in ["id", "song_popularity", "fold"]:  
        feature_cols.append(col)

print(f"Using {len(feature_cols)} features")

# Check for categorical features
categorical_features = []
for feature in feature_cols:
    if train_df[feature].dtype == "object":
        categorical_features.append(feature)
        
if len(categorical_features) > 0:
    print(f"Found {len(categorical_features)} categorical features: {categorical_features}")
else:
    print("No categorical features detected")


num_folds = 5  
random_state = 42  

# Initialize prediction arrays
out_of_fold_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

# CatBoost hyperparameters
catboost_params = {
    "iterations": 2000,        # might be overkill but better safe than sorry
    "learning_rate": 0.05,     
    "depth": 6,                
    "eval_metric": "AUC",      # our main metric
    "random_seed": random_state,
    "early_stopping_rounds": 50,  
    "verbose": 100,           
    "use_best_model": True     
}

print("Starting cross-validation...")


stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

fold_scores = []  

for fold_num, (train_indices, valid_indices) in enumerate(stratified_kfold.split(train_df, train_df["song_popularity"])):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_num + 1} out of {num_folds}")
    print(f"{'='*50}")
    
   
    X_train = train_df.loc[train_indices, feature_cols]
    y_train = train_df.loc[train_indices, "song_popularity"]
    X_valid = train_df.loc[valid_indices, feature_cols]
    y_valid = train_df.loc[valid_indices, "song_popularity"]
    
    print(f"Train size: {len(X_train)}, Valid size: {len(X_valid)}")
    
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)
    
    # Initialize and train the model
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(train_pool, eval_set=valid_pool)
    
    # Get out-of-fold predictions
    valid_predictions = catboost_model.predict_proba(X_valid)[:, 1]
    out_of_fold_predictions[valid_indices] = valid_predictions
    
    # Accumulate test predictions (we'll average them later)
    current_test_preds = catboost_model.predict_proba(test_df[feature_cols])[:, 1]
    test_predictions += current_test_preds / num_folds  # averaging as we go
    
    # Calculate and store fold AUC
    current_fold_auc = roc_auc_score(y_valid, valid_predictions)
    fold_scores.append(current_fold_auc)
    print(f"Fold {fold_num + 1} Validation AUC: {current_fold_auc:.6f}")

print(f"\n{'='*60}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*60}")

# Show individual fold scores
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1}: {score:.6f}")

# Calculate overall performance
overall_oof_auc = roc_auc_score(train_df["song_popularity"], out_of_fold_predictions)
mean_cv_score = np.mean(fold_scores)
std_cv_score = np.std(fold_scores)

print(f"\nMean CV AUC: {mean_cv_score:.6f} (+/- {std_cv_score:.6f})")
print(f"Overall OOF AUC: {overall_oof_auc:.6f}")

# Create submission file
print("\nPreparing submission...")
final_submission = sample_sub.copy()
final_submission["song_popularity"] = test_predictions

# Save the submission
submission_filename = "submission.csv"
final_submission.to_csv(submission_filename, index=False)
print(f"Submission saved as: {submission_filename}")

# Quick check of submission format
print(f"\nSubmission shape: {final_submission.shape}")
print("First few predictions:")
print(final_submission.head())
