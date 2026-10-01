"""
train_cnn.py — Stage 04, Step 2: 1D Convolutional Neural Network for 1,000 Users
================================================================================
Builds and trains a 1D-CNN model for 1,000-class keystroke identification.
Uses GaussianNoise augmentation, Conv1D temporal filters, GlobalAveragePooling1D,
and L2 + Dropout regularization.

Inputs:
    - ../03_feature_engineering_preprocessing/train_scaled.csv   (12,000 x 127)
    - ../03_feature_engineering_preprocessing/test_scaled.csv    (3,000 x 127)
    - ../03_feature_engineering_preprocessing/label_encoder.joblib

Outputs:
    - cnn_model.keras                     (serialized trained model)
    - cnn_training_history.csv            (per-epoch loss/accuracy log)
    - 1D_CNN_results.md                   (markdown evaluation summary)
    - 1D_CNN_confusion_matrix.png         (1,000x1,000 heatmap)
    - 1D_CNN_classification_report.txt
    - 1D_CNN_metrics.json

Usage:
    python train_cnn.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib

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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage03_dir = os.path.join(base_dir, "03_feature_engineering_preprocessing")
    output_dir = os.path.join(base_dir, "04_model_training")
    os.makedirs(output_dir, exist_ok=True)

    MODEL_NAME = "1D_CNN"
    EPOCHS = 150
    BATCH_SIZE = 64
    PATIENCE = 20
    VAL_FRACTION = 0.15
    L2_REG = 1e-3

    print("=" * 60)
    print(f"STAGE 04: Training 1D-CNN ({MODEL_NAME}) on 1,000 Users")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available:      {tf.config.list_physical_devices('GPU')}")
    print(f"Max Epochs:         {EPOCHS}")
    print(f"Batch Size:         {BATCH_SIZE}")

    # 1. Load Data
    print("\nLoading scaled train/test partitions...")
    df_train = pd.read_csv(os.path.join(stage03_dir, "train_scaled.csv"))
    df_test = pd.read_csv(os.path.join(stage03_dir, "test_scaled.csv"))
    label_encoder = joblib.load(os.path.join(stage03_dir, "label_encoder.joblib"))

    id_cols = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'label']
    feature_cols = [c for c in df_train.columns if c not in id_cols]
    NUM_FEATURES = len(feature_cols)
    NUM_CLASSES = len(label_encoder.classes_)

    X_train_full = df_train[feature_cols].values
    y_train_full = df_train['label'].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values.astype(int)

    print(f"X_train shape: {X_train_full.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Features:      {NUM_FEATURES}")
    print(f"Classes:       {NUM_CLASSES}")

    # 2. Stratified Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )
    print(f"Stratified Split -> Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    # 3. Reshape for Conv1D (samples, timesteps, channels)
    X_train_cnn = X_train.reshape(X_train.shape[0], NUM_FEATURES, 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], NUM_FEATURES, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], NUM_FEATURES, 1)

    y_train_onehot = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_CLASSES)

    # 4. Model Architecture
    model = keras.Sequential([
        layers.GaussianNoise(0.05, input_shape=(NUM_FEATURES, 1)),
        layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling1D(),

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    # 5. Callbacks & Training
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

    print(f"\nTraining 1D-CNN on {NUM_CLASSES} classes...")
    history = model.fit(
        X_train_cnn, y_train_onehot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_cnn, y_val_onehot),
        callbacks=[early_stop, lr_scheduler],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(output_dir, "cnn_training_history.csv")
    history_df.to_csv(history_path, index=False)

    best_val_acc = max(history.history['val_accuracy'])
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # 6. Evaluation
    print("\nRunning predictions on test set...")
    y_pred_probs = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        model_name=MODEL_NAME,
        output_dir=output_dir,
        train_samples=len(X_train_full)
    )

    # 7. Save Model
    model_path = os.path.join(output_dir, "cnn_model.keras")
    model.save(model_path)
    print(f"Saved trained 1D-CNN model: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_cnn()
