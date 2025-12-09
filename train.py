# Basic
import warnings
import pickle
from collections import Counter

# Data Handling
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.model_selection import train_test_split
from utils import FeatureEngineer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Model
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)

# Calibration
from sklearn.calibration import CalibratedClassifierCV

# Ignore warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Fraud.csv")

# print(df.sample(5))
# print(df.shape)

## Data Cleaning

# Drop any missing values and duplicates
df.dropna(inplace = True)
df.drop_duplicates(inplace = True)

# Check data types
# print(df.dtypes)

# Optimize memory usage for large numeric columns
for col in df.columns:
    if df[col].dtype == "int64":
        df[col] = df[col].astype("int32")  # smaller integer type
    elif df[col].dtype == "float64":
        df[col] = df[col].astype("float32")  # smaller integer type

# Optimize memory usage for binary columns
binary_cols = ["isFraud", "isFlaggedFraud"]

for col in binary_cols:
    df[col] = df[col].astype("int8")

# Check data types again
# print(df.dtypes)

## Data Splitting

# Separate the data into feature and target variables
X = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]
y = df["isFraud"]

# Split the data to train, validation, and test sets

# First generate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size = 0.5,  # 5% test
    stratify = y,
    random_state = 42)

# Then generate train + validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size = 0.10,  # 10% Validation
    stratify = y_temp,
    random_state = 42)

# print(X_train.shape, X_test.shape, X_val.shape, y_val.shape, y_train.shape, y_test.shape)

# Generate a separate set for calibration and threshold tuning
X_train_final, X_cal, y_train_final, y_cal = train_test_split(
    X_temp,
    y_temp,
    test_size = 0.10,         # 10% for calibration + threshold tuning
    stratify = y_temp,
    random_state = 42
)

# print(X_train_final.shape, X_cal.shape, y_train_final.shape, y_cal.shape)

## Data Preprocessing

# Column types
cat_cols = ["type"]
num_cols = ["amount", "balanceDiffOrig", "balanceDiffDest", "isLargeTransaction",
            "zero_balance_flag", "sender_amount_ratio", "receiver_amount_ratio"]

# Preprocessing pipeline
full_preprocessing  = ColumnTransformer(transformers = [
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop = "first", sparse_output = False), cat_cols)
])

## Modeling

# Dynamically compute scale_pos_weight
counter = Counter(y_train_final)
spw = counter[0] / max(1, counter[1])  # neg / pos

print(f"Dynamic scale_pos_weight: {spw:.2f}")

# Final model
final_xgb = Pipeline([
    ("features", FeatureEngineer()),    # Feature Engineering
    ("preprocessing", full_preprocessing),
    ("xgbc", XGBClassifier(n_estimators = 600, tree_method = "hist", subsample = 0.8, reg_lambda = 5, reg_alpha = 3, max_depth = 7, learning_rate = 0.07, gamma = 0.1,
                           colsample_bytree = 0.8, use_label_encoder = False, eval_metric = "aucpr", scale_pos_weight = spw))
])

# Train the final model
final_xgb.fit(X_train_final, y_train_final)

## Model Calibration

# Calibrate the Xgb Classifier

calibrated_xgb = CalibratedClassifierCV(estimator = final_xgb, method = "isotonic", cv = 7)
calibrated_xgb.fit(X_cal, y_cal)

## Threshold Tuning

# Generate fraud probabilities on the calibration and test sets
y_probs_cal = calibrated_xgb.predict_proba(X_cal)[:, 1]
y_probs_test = calibrated_xgb.predict_proba(X_test)[:, 1]
y_cal_arr = np.array(y_cal)

# Compute precision–recall curve
precision, recall, thresholds = precision_recall_curve(y_cal_arr, y_probs_cal)
precision, recall = precision[:-1], recall[:-1]  # remove last point to match threshold array

# Find the threshold that meets target recall and precision
target_recall = 0.99
target_precision = 0.95

valid_idx = np.where((recall >= target_recall) & (precision >= target_precision))[0]

best_idx = valid_idx[0]

best_threshold = thresholds[best_idx]
print("Selected threshold and metrics on the calibration set:")
print(f"Threshold: {best_threshold:.3f}, Precision: {precision[best_idx]:.3f}, Recall: {recall[best_idx]:.3f}")

# Apply threshold to the test set
y_pred_thr = (y_probs_test >= best_threshold).astype(int)

# Some metrics for Calibrated Xgb Classifier after threshold tuning
precision = precision_score(y_test, y_pred_thr)
recall = recall_score(y_test, y_pred_thr)
f1 = f1_score(y_test, y_pred_thr)
roc_auc = roc_auc_score(y_test, y_pred_thr)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1_Score: {f1:.3f}, ROC_AOC: {roc_auc:.3f}")

## Save the Model

# Save the Calibrated Xgb Classifier with the tuned threshold
with open("calibrated_xgb_clf_with_tuned_threshold.pkl", "wb") as f:
    pickle.dump({
        "pipeline": calibrated_xgb,
        "threshold": best_threshold
    }, f)