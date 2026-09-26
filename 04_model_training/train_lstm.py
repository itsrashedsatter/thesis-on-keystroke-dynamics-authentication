"""
train_lstm.py — Stage 04, Step 3: Bidirectional LSTM (v2)
==========================================================
Builds and trains a bidirectional LSTM for 60-class keystroke
identification. Designed for small-dataset conditions.

The 124-feature session vector is reshaped to (samples, 31, 4) by
grouping features into 31 blocks of 4 — each block represents a
meaningful group of related biometric measurements.

Key v2 improvements (same philosophy as CNN v2):
  - Stratified train/val split ensuring all 60 classes in validation
  - Reduced LSTM units (64→32 per direction) to prevent overfitting
  - L2 regularization on Dense layers
  - GaussianNoise augmentation
  - ReduceLROnPlateau + generous early stopping patience

Inputs (from Stage 03):
    - 03_feature_engineering_preprocessing/train_scaled.csv   (720 x 127)
    - 03_feature_engineering_preprocessing/test_scaled.csv    (180 x 127)
    - 03_feature_engineering_preprocessing/label_encoder.joblib

Outputs (into 04_model_training/):
    - lstm_model.keras                    (serialized trained model)
    - lstm_training_history.csv           (per-epoch loss/accuracy log)
    - LSTM_results.md                     (markdown evaluation summary)
    - LSTM_confusion_matrix.png           (60x60 heatmap)
    - LSTM_classification_report.txt
    - LSTM_metrics.json

Usage:
    python 04_model_training/train_lstm.py
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


def train_lstm():
    # -----------------------------------------------------------------
    # 1. PATHS & CONFIGURATION
    # -----------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage03_dir = os.path.join(base_dir, "03_feature_engineering_preprocessing")
    output_dir = os.path.join(base_dir, "04_model_training")
    os.makedirs(output_dir, exist_ok=True)

    MODEL_NAME = "LSTM"
    EPOCHS = 150
    BATCH_SIZE = 32
    PATIENCE = 20
    NUM_CLASSES = 60
    VAL_FRACTION = 0.15
    L2_REG = 1e-3

    # LSTM reshaping: group 124 features into blocks
    TIMESTEPS = 31
    FEATURE_DIM = 4  # 31 x 4 = 124 features

    print(f"Base Directory: {base_dir}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print(f"LSTM input shape: ({TIMESTEPS}, {FEATURE_DIM})")

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
    # 4. RESHAPE FOR LSTM INPUT
    # -----------------------------------------------------------------
    required = TIMESTEPS * FEATURE_DIM  # 124
    if NUM_FEATURES > required:
        print(f"Note: Truncating from {NUM_FEATURES} to {required} features.")
        X_train = X_train[:, :required]
        X_val = X_val[:, :required]
        X_test = X_test[:, :required]
    elif NUM_FEATURES < required:
        pad_width = required - NUM_FEATURES
        print(f"Note: Zero-padding {pad_width} features.")
        X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
        X_val = np.pad(X_val, ((0, 0), (0, pad_width)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')

    X_train_lstm = X_train.reshape(X_train.shape[0], TIMESTEPS, FEATURE_DIM)
    X_val_lstm = X_val.reshape(X_val.shape[0], TIMESTEPS, FEATURE_DIM)
    X_test_lstm = X_test.reshape(X_test.shape[0], TIMESTEPS, FEATURE_DIM)

    y_train_onehot = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_CLASSES)

    print(f"X_train_lstm shape: {X_train_lstm.shape}")
    print(f"X_val_lstm shape:   {X_val_lstm.shape}")

    # -----------------------------------------------------------------
    # 5. BUILD BIDIRECTIONAL LSTM ARCHITECTURE (SLIM)
    # -----------------------------------------------------------------
    print("\nBuilding Bidirectional LSTM architecture (slim variant)...")

    model = keras.Sequential([
        # Input noise augmentation (only active during training)
        layers.GaussianNoise(0.05, input_shape=(TIMESTEPS, FEATURE_DIM)),

        # Bidirectional LSTM Layer 1
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                        recurrent_dropout=0.1),
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Bidirectional LSTM Layer 2
        layers.Bidirectional(
            layers.LSTM(32, return_sequences=False,
                        recurrent_dropout=0.1),
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

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

    # -----------------------------------------------------------------
    # 6. TRAIN WITH EARLY STOPPING + LR SCHEDULING
    # -----------------------------------------------------------------
    print(f"\nTraining Bidirectional LSTM (max {EPOCHS} epochs, patience {PATIENCE})...")

    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
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
        X_train_lstm, y_train_onehot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_lstm, y_val_onehot),
        callbacks=[early_stop, lr_scheduler],
        verbose=1,
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(output_dir, "lstm_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # -----------------------------------------------------------------
    # 7. PREDICT & EVALUATE ON HELD-OUT TEST SET
    # -----------------------------------------------------------------
    print("\nRunning predictions on test set...")
    y_pred_probs = model.predict(X_test_lstm)
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
    model_path = os.path.join(output_dir, "lstm_model.keras")
    model.save(model_path)
    print(f"Saved trained model: {model_path}")

    print(f"\n{'='*60}")
    print(f"  Stage 04, Step 3 (LSTM v2) — COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_lstm()
