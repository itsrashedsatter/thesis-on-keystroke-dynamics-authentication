# Behavior-Based Continuous Verification using Keystroke Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: Biometrics](https://img.shields.io/badge/Research-Behavioral%20Biometrics-success.svg)]()
[![Zero Trust: Continuous Auth](https://img.shields.io/badge/Zero%20Trust-Continuous%20Verification-orange.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itsrashedsatter/thesis-on-keystroke-dynamics-authentication/blob/main/keystroke_dynamics_pipeline.ipynb)

> **University Thesis Project**: Continuous behavioral biometric authentication and closed-set user identification using temporal keystroke dynamics and digraph latency modeling under a Zero Trust Architecture (ZTA).

---

## 📌 Project Overview
Traditional point-of-entry authentication mechanisms (passwords, PINs, OTPs) verify a user's identity only once at session inception, leaving systems vulnerable to session hijacking and physical unauthorized access. This research implements **passive, continuous identity verification** by analyzing fine-grained keystroke timing biometrics (Hold Times, Inter-Key Flight Times, and Latency Cadence).

```
   [ Raw Keystrokes ] 
          │  (Press / Release Timestamps)
          ▼
   [ 01: Data Ingestion & Merging ] ───► Cohort of 1,000 Verified Typists (15 sessions each)
          │
          ▼
   [ 02: Cleaning & EDA ] ─────────────► Jitter Removal, Rollover Flagging, Distribution Analysis
          │
          ▼
   [ 03: Feature Engineering ] ────────► 124 Biometric Features (Global + Top 25 Digraph Timings)
          │
          ▼
   [ 04: Continuous Verification ] ────► Stratified Train/Test (80/20) for ML/DL Benchmarks
```

---

## 🔬 Dataset & Experimental Cohort
- **Base Data**: Aalto University 136 Million Keystrokes Dataset (`noelsmathew/keystrokes`).
- **Target Cohort**: **1,000 participants** with $\ge 15$ completed typing sessions each (total **730,950 valid keystrokes** after cleaning).
- **Session Design**: 15 distinct sentence sessions per user $\rightarrow$ **15,000 total feature vectors**.
- **Data Partition**: Stratified session-level 80/20 split $\rightarrow$ **12,000 training** and **3,000 test sessions** (zero temporal leakage).
- **Random-Chance Baseline**: $1/1{,}000 = \mathbf{0.10\%}$.

---

## 📊 Biometric Feature Architecture (124 Features)

| Category | Count | Description |
|---|:---:|---|
| **Global Timing Statistics** | 12 | Mean, standard deviation, median, and IQR of Hold Time ($HT$), Flight Time ($UD$), and Latency ($DD$). |
| **Cadence & Speed** | 7 | Total keystrokes, session duration, keys/sec, Words Per Minute (WPM), rollover ratio ($UD < 0$), cognitive pause rates. |
| **Error & Correction** | 5 | Backspace frequency, backspace rate ($BKSP / Total$), shift modifier rate, edit length delta. |
| **Top 25 Digraph Biometrics** | 100 | Mean $UD$, mean $DD$, mean $HT$, and frequency for the top 25 character transitions (`t_h`, `h_e`, `e_SPACE`, `BKSP_BKSP`, etc.). |

---

## 🚀 Benchmark Results (1,000 Enrolled Typists, 3,000 Test Sessions)

| Model Architecture | Accuracy | Macro F1 | Correct / Total | Equal Error Rate (EER) | ROC-AUC | Factor over Random |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (RF)** | **97.00%** | **0.9687** | **2,910 / 3,000** | **2.87%** | **0.9998** | **970×** |
| **Bidirectional LSTM** | 86.97% | 0.8649 | 2,609 / 3,000 | 7.43% | 0.9972 | 870× |
| **1D CNN** | 49.07% | 0.4606 | 1,472 / 3,000 | 24.12% | 0.9575 | 490× |
| *Random Chance Baseline* | 0.10% | 0.0010 | $\approx 3$ / 3,000 | 50.00% | 0.5000 | 1× |

---

## 📁 Repository Structure

```
├── keystroke_dynamics_pipeline.ipynb  # End-to-end Google Colab pipeline (runnable in 1 click)
├── inspect_and_merge.py               # Stage 01: Raw participant scanning & cohort selection
├── clean_and_eda.py                   # Stage 02: Telemetry cleaning, outlier filtering & EDA
├── feature_engineering_and_preprocessing.py # Stage 03: 124-feature extraction & 80/20 scaling
├── 04_model_training/                 # Stage 04: Standalone model training scripts
│   ├── train_random_forest.py         # Random Forest training & evaluation
│   ├── train_lstm.py                  # Bidirectional LSTM training & evaluation
│   ├── train_cnn.py                   # 1D CNN training & evaluation
│   ├── evaluation_utils.py            # FAR, FRR, EER, and multiclass metric routines
│   └── requirements_stage04.txt       # Dependencies
├── poster.tex                         # Academic conference poster (Beamer)
└── README.md
```

---

## ⚡ Running the Pipeline

### Option 1: One-Click Execution on Google Colab (Recommended)
Click the badge below to open and run the complete self-contained pipeline on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itsrashedsatter/thesis-on-keystroke-dynamics-authentication/blob/main/keystroke_dynamics_pipeline.ipynb)

### Option 2: Local Execution
```bash
# 1. Install dependencies
pip install -r 04_model_training/requirements_stage04.txt

# 2. Train models
python 04_model_training/train_random_forest.py
python 04_model_training/train_lstm.py
python 04_model_training/train_cnn.py
```

---

## 📜 Citation & License
This research is developed as a University Thesis at BRAC University. Licensed under the [MIT License](LICENSE).
