# Stage 04: Model Training & Benchmarking (1,000-User Cohort)

This directory contains the training and evaluation pipelines for the **1,000-user closed-set keystroke identification task** ($C = 1,000$ classes).

---

## 🔬 Dataset Inputs (Zero-Leakage from Stage 03)
- `../03_feature_engineering_preprocessing/train_scaled.csv` (12,000 samples: 12 sessions $\times$ 1,000 users)
- `../03_feature_engineering_preprocessing/test_scaled.csv` (3,000 samples: 3 sessions $\times$ 1,000 users)
- `../03_feature_engineering_preprocessing/label_encoder.joblib` (Fitted LabelEncoder)
- `../03_feature_engineering_preprocessing/scaler.joblib` (Fitted StandardScaler)

---

## 🚀 Execution Instructions

### 1. Train Random Forest Baseline
```bash
python train_random_forest.py
```
- Fits 300 bagged trees across all available CPU cores.
- Evaluates on the held-out 3,000 test sessions.
- Generates `rf_model.joblib`, `rf_feature_importances.csv`, `Random_Forest_metrics.json`, `Random_Forest_confusion_matrix.png`, and `Random_Forest_results.md`.

### 2. Train 1D Convolutional Neural Network
```bash
python train_cnn.py
```
- Trains a regularized 1D-CNN with Gaussian noise augmentation, multi-scale temporal convolutions, and GlobalAveragePooling1D.
- Generates `cnn_model.keras`, `cnn_training_history.csv`, and evaluation artifacts.

### 3. Train Bidirectional LSTM
```bash
python train_lstm.py
```
- Reshapes the 124 biometric features into a sequence of $(31, 4)$ (31 temporal steps of 4 related timing metrics).
- Trains forward and backward LSTM layers with recurrent dropout.
- Generates `lstm_model.keras`, `lstm_training_history.csv`, and evaluation artifacts.

---

## 📁 Clean Directory Contents

| File | Purpose |
|---|---|
| `evaluation_utils.py` | Shared evaluation module (accuracy, macro/weighted F1, confusion matrix, markdown report). |
| `train_random_forest.py` | Standalone Random Forest training script. |
| `train_cnn.py` | Standalone 1D-CNN training script. |
| `train_lstm.py` | Standalone Bidirectional LSTM training script. |
| `requirements_stage04.txt` | Python dependencies. |
