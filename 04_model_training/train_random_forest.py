"""
train_random_forest.py — Stage 04, Step 1: Random Forest for 1,000-User Cohort
=============================================================================
Trains a RandomForestClassifier on the 1,000-class closed-set keystroke
identification task using the scaled session-level feature vectors from
cohort_1000/03_feature_engineering_preprocessing/.

Inputs:
    - ../03_feature_engineering_preprocessing/train_scaled.csv   (12,000 x 127)
    - ../03_feature_engineering_preprocessing/test_scaled.csv    (3,000 x 127)
    - ../03_feature_engineering_preprocessing/label_encoder.joblib

Outputs:
    - rf_model.joblib                     (serialized trained model)
    - rf_feature_importances.csv          (sorted feature importances)
    - Random_Forest_results.md            (markdown evaluation summary)
    - Random_Forest_confusion_matrix.png  (1,000x1,000 heatmap)
    - Random_Forest_classification_report.txt
    - Random_Forest_metrics.json

Usage:
    python train_random_forest.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation_utils import evaluate_predictions


def train_random_forest():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage03_dir = os.path.join(base_dir, "03_feature_engineering_preprocessing")
    output_dir = os.path.join(base_dir, "04_model_training")
    os.makedirs(output_dir, exist_ok=True)

    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    MAX_DEPTH = 25
    MIN_SAMPLES_LEAF = 2
    N_JOBS = 4
    MODEL_NAME = "Random_Forest"

    print("=" * 60)
    print(f"STAGE 04: Training Random Forest ({MODEL_NAME}) on 1,000 Users")
    print("=" * 60)
    print(f"Base Directory:   {base_dir}")
    print(f"Stage 03 Inputs:  {stage03_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Estimators:       {N_ESTIMATORS}")
    print(f"Max Depth:        {MAX_DEPTH}")
    print(f"Min Samples Leaf: {MIN_SAMPLES_LEAF}")
    print(f"Parallel Workers: {N_JOBS}")

    # 1. Load Data
    print("\nLoading scaled train/test partitions...")
    df_train = pd.read_csv(os.path.join(stage03_dir, "train_scaled.csv"))
    df_test = pd.read_csv(os.path.join(stage03_dir, "test_scaled.csv"))
    label_encoder = joblib.load(os.path.join(stage03_dir, "label_encoder.joblib"))

    id_cols = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'label']
    feature_cols = [c for c in df_train.columns if c not in id_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values.astype(int)

    num_classes = len(label_encoder.classes_)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Target Classes: {num_classes}")

    # 2. Train Model
    print(f"\nTraining RandomForestClassifier with {N_ESTIMATORS} trees...")
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        class_weight='balanced',
    )
    rf_model.fit(X_train, y_train)
    print("Training complete.")

    # 3. Predict & Evaluate
    print("\nEvaluating predictions on held-out test partition...")
    y_pred = rf_model.predict(X_test)

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        model_name=MODEL_NAME,
        output_dir=output_dir,
        train_samples=len(X_train)
    )

    # 4. Feature Importances
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    fi_path = os.path.join(output_dir, "rf_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved feature importances: {fi_path}")

    print("\nTop 10 Most Discriminative Biometric Features:")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:<28} : {row['importance']:.6f}")

    # 5. Serialize Model
    model_path = os.path.join(output_dir, "rf_model.joblib")
    joblib.dump(rf_model, model_path, compress=3)
    print(f"Saved trained Random Forest model: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_random_forest()
