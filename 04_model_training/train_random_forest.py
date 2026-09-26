"""
train_random_forest.py — Stage 04, Step 1: Random Forest Baseline
==================================================================
Trains a RandomForestClassifier on the 60-class closed-set keystroke
identification task using the scaled session-level feature vectors from
Stage 03.

This is the baseline sanity check: if RF cannot beat random-chance
accuracy (~1.7% for 60 classes), something upstream is broken.

Inputs (from Stage 03):
    - 03_feature_engineering_preprocessing/train_scaled.csv   (720 x 127)
    - 03_feature_engineering_preprocessing/test_scaled.csv    (180 x 127)
    - 03_feature_engineering_preprocessing/label_encoder.joblib
    - 03_feature_engineering_preprocessing/label_mapping.json

Outputs (into 04_model_training/):
    - rf_model.joblib                     (serialized trained model)
    - rf_feature_importances.csv          (sorted feature importances)
    - Random_Forest_results.md            (markdown evaluation summary)
    - Random_Forest_confusion_matrix.png  (60x60 heatmap)
    - Random_Forest_classification_report.txt
    - Random_Forest_metrics.json

Usage:
    python 04_model_training/train_random_forest.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path so evaluation_utils can be imported
# when running from the repo root OR from inside 04_model_training/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation_utils import evaluate_predictions


def train_random_forest():
    # -----------------------------------------------------------------
    # 1. PATHS & CONFIGURATION
    # -----------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage03_dir = os.path.join(base_dir, "03_feature_engineering_preprocessing")
    output_dir = os.path.join(base_dir, "04_model_training")
    os.makedirs(output_dir, exist_ok=True)

    RANDOM_STATE = 42
    N_ESTIMATORS = 300
    MODEL_NAME = "Random_Forest"

    print(f"Base Directory: {base_dir}")
    print(f"Stage 03 Input: {stage03_dir}")
    print(f"Stage 04 Output: {output_dir}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Number of Trees: {N_ESTIMATORS}")

    # -----------------------------------------------------------------
    # 2. LOAD DATA & ARTIFACTS FROM STAGE 03
    # -----------------------------------------------------------------
    print("\nLoading scaled train/test data from Stage 03...")
    df_train = pd.read_csv(os.path.join(stage03_dir, "train_scaled.csv"))
    df_test = pd.read_csv(os.path.join(stage03_dir, "test_scaled.csv"))

    label_encoder = joblib.load(os.path.join(stage03_dir, "label_encoder.joblib"))

    # Separate features (X) from identifiers/labels
    id_cols = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'label']
    feature_cols = [c for c in df_train.columns if c not in id_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values.astype(int)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_train unique classes: {len(np.unique(y_train))}")
    print(f"y_test  unique classes: {len(np.unique(y_test))}")

    # -----------------------------------------------------------------
    # 3. TRAIN RANDOM FOREST CLASSIFIER
    # -----------------------------------------------------------------
    print(f"\nTraining RandomForestClassifier (n_estimators={N_ESTIMATORS})...")
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',  # Handle any minor class imbalance
        max_depth=None,           # Let trees grow fully
    )
    rf_model.fit(X_train, y_train)
    print("Training complete.")

    # -----------------------------------------------------------------
    # 4. PREDICT & EVALUATE
    # -----------------------------------------------------------------
    print("\nRunning predictions on test set...")
    y_pred = rf_model.predict(X_test)

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        model_name=MODEL_NAME,
        output_dir=output_dir,
    )

    # -----------------------------------------------------------------
    # 5. FEATURE IMPORTANCE EXTRACTION
    # -----------------------------------------------------------------
    print("\nExtracting feature importances...")
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Save full sorted feature importances
    importance_path = os.path.join(output_dir, "rf_feature_importances.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importances: {importance_path}")

    # Print top 20 most important features
    print(f"\nTop 20 Most Important Features (out of {len(feature_cols)}):")
    print("-" * 50)
    for i, row in importance_df.head(20).iterrows():
        print(f"  {i+1:2d}. {row['feature']:35s}  {row['importance']:.6f}")
    print("-" * 50)

    # Count near-zero importance features (< 0.001)
    near_zero = (importance_df['importance'] < 0.001).sum()
    print(f"\nFeatures with near-zero importance (<0.001): {near_zero}/{len(feature_cols)}")

    # -----------------------------------------------------------------
    # 6. SAVE TRAINED MODEL
    # -----------------------------------------------------------------
    model_path = os.path.join(output_dir, "rf_model.joblib")
    joblib.dump(rf_model, model_path)
    print(f"\nSaved trained model: {model_path}")

    print(f"\n{'='*60}")
    print(f"  Stage 04, Step 1 (Random Forest) — COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_random_forest()
