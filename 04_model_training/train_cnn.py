"""
train_cnn.py — Stage 04, Step 2: 1D Convolutional Neural Network (v2)
======================================================================
Builds and trains a lightweight 1D-CNN model for 60-class keystroke
identification. Designed for small-dataset conditions (720 train, 180 test).

Key design choices for this dataset size:
  - Slim Conv1D filters (32→64) to prevent overfitting
  - Heavy Dropout (0.4–0.5) and L2 regularization
  - Stratified train/val split to ensure every class appears in validation
  - Learning rate warmup + ReduceLROnPlateau
  - Gaussian noise augmentation layer for implicit data augmentation

Inputs (from Stage 03):
    - 03_feature_engineering_preprocessing/train_scaled.csv   (720 x 127)
    - 03_feature_engineering_preprocessing/test_scaled.csv    (180 x 127)
    - 03_feature_engineering_preprocessing/label_encoder.joblib

Outputs (into 04_model_training/):
    - cnn_model.keras                     (serialized trained model)
    - cnn_training_history.csv            (per-epoch loss/accuracy log)
    - 1D_CNN_results.md                   (markdown evaluation summary)
    - 1D_CNN_confusion_matrix.png         (60x60 heatmap)
    - 1D_CNN_classification_report.txt
    - 1D_CNN_metrics.json

Usage:
    python 04_model_training/train_cnn.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib

# -----------------------------------------------------------------
# REPRODUCIBILITY: Set all random seeds BEFORE importing TensorFlow
# -----------------------------------------------------------------
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation_utils import evaluate_predictions


def train_cnn():
    # -----------------------------------------------------------------
    # 1. PATHS & CONFIGURATION
    # -----------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage03_dir = os.path.join(base_dir, "03_feature_engineering_preprocessing")
    output_dir = os.path.join(base_dir, "04_model_training")
    os.makedirs(output_dir, exist_ok=True)

    MODEL_NAME = "1D_CNN"
    EPOCHS = 150          # More epochs — early stopping will handle it
    BATCH_SIZE = 32
    PATIENCE = 20         # Generous patience for small datasets
    NUM_CLASSES = 60
    VAL_FRACTION = 0.15   # Stratified validation split
    L2_REG = 1e-3

    print(f"Base Directory: {base_dir}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print(f"L2 Regularization: {L2_REG}")

    # -----------------------------------------------------------------
    # 2. LOAD DATA & ARTIFACTS FROM STAGE 03
    # -----------------------------------------------------------------
    print("\nLoading scaled train/test data from Stage 03...")
    df_train = pd.read_csv(os.path.join(stage03_dir, "train_scaled.csv"))
    df_test = pd.read_csv(os.path.join(stage03_dir, "test_scaled.csv"))
    label_encoder = joblib.load(os.path.join(stage03_dir, "label_encoder.joblib"))

    id_cols = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'label']
    feature_cols = [c for c in df_train.columns if c not in id_cols]
    NUM_FEATURES = len(feature_cols)

    X_train_full = df_train[feature_cols].values
    y_train_full = df_train['label'].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values.astype(int)

    print(f"X_train_full shape: {X_train_full.shape}")
    print(f"X_test shape:       {X_test.shape}")
    print(f"Number of features: {NUM_FEATURES}")

    # -----------------------------------------------------------------
    # 3. STRATIFIED TRAIN / VALIDATION SPLIT
    # -----------------------------------------------------------------
    # Critical: random split can leave classes unrepresented in validation
    # with only 12 samples per class. Stratified split guarantees each
    # class appears in both partitions.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )
    print(f"\nStratified split:")
    print(f"  Training:   {X_train.shape[0]} samples ({len(np.unique(y_train))} classes)")
    print(f"  Validation: {X_val.shape[0]} samples ({len(np.unique(y_val))} classes)")

    # -----------------------------------------------------------------
    # 4. RESHAPE FOR CONV1D INPUT
    # -----------------------------------------------------------------
    # Conv1D expects shape: (samples, timesteps, channels)
    X_train_cnn = X_train.reshape(X_train.shape[0], NUM_FEATURES, 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], NUM_FEATURES, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], NUM_FEATURES, 1)

    # One-hot encode targets
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_CLASSES)

    print(f"X_train_cnn shape: {X_train_cnn.shape}")
    print(f"X_val_cnn shape:   {X_val_cnn.shape}")

    # -----------------------------------------------------------------
    # 5. BUILD LIGHTWEIGHT 1D-CNN ARCHITECTURE
    # -----------------------------------------------------------------
    # Design rationale for small datasets (720 total, ~612 train):
    # - Slim filters (32→64) instead of (64→128) to reduce parameter count
    # - GaussianNoise layer for implicit data augmentation during training
    # - L2 regularization on all Dense layers
    # - Heavy Dropout to prevent memorization
    # Total params target: <100K (vs 1.08M in v1)
    print("\nBuilding lightweight 1D-CNN architecture...")

    model = keras.Sequential([
        # Input noise augmentation (only active during training)
        layers.GaussianNoise(0.05, input_shape=(NUM_FEATURES, 1)),

        # Block 1: Local feature patterns
        layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Block 2: Deeper extraction
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Global Average Pooling (much fewer params than Flatten)
        layers.GlobalAveragePooling1D(),

        # Classification Head
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG)),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,} (target: <100K)")

    # -----------------------------------------------------------------
    # 6. TRAIN WITH EARLY STOPPING + LR SCHEDULING
    # -----------------------------------------------------------------
    print(f"\nTraining 1D-CNN (max {EPOCHS} epochs, patience {PATIENCE})...")

    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',      # Monitor accuracy, not loss
        patience=PATIENCE,
        restore_best_weights=True,
        mode='max',
        verbose=1,
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        X_train_cnn, y_train_onehot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_cnn, y_val_onehot),  # Explicit stratified val set
        callbacks=[early_stop, lr_scheduler],
        verbose=1,
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(output_dir, "cnn_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # -----------------------------------------------------------------
    # 7. PREDICT & EVALUATE ON HELD-OUT TEST SET
    # -----------------------------------------------------------------
    print("\nRunning predictions on test set...")
    y_pred_probs = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        model_name=MODEL_NAME,
        output_dir=output_dir,
    )

    # -----------------------------------------------------------------
    # 8. SAVE TRAINED MODEL
    # -----------------------------------------------------------------
    model_path = os.path.join(output_dir, "cnn_model.keras")
    model.save(model_path)
    print(f"Saved trained model: {model_path}")

    print(f"\n{'='*60}")
    print(f"  Stage 04, Step 2 (1D-CNN v2) — COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_cnn()
