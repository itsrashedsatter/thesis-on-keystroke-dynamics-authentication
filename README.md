# Behavior-Based Continuous Verification using Keystroke Dynamics

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: Biometrics](https://img.shields.io/badge/Research-Behavioral%20Biometrics-success.svg)]()
[![Zero Trust: Continuous Auth](https://img.shields.io/badge/Zero%20Trust-Continuous%20Verification-orange.svg)]()

> **University Thesis Project**: Continuous behavioral biometric authentication and closed-set user identification using temporal keystroke dynamics and digram latency modeling under a Zero Trust Architecture (ZTA).

---

## 📌 Project Overview
Traditional point-of-entry authentication mechanisms (passwords, PINs, OTPs) verify a user's identity only once at session inception, leaving systems vulnerable to session hijacking and physical unauthorized access. This research implements **passive, continuous identity verification** by analyzing fine-grained keystroke timing biometrics (Hold Times, Inter-Key Flight Times, and Latency Cadence).

```
   [ Raw Keystrokes ] 
          │  (Press / Release Timestamps)
          ▼
   [ 01: Data Ingestion & Merging ] ───► Cohort of 60 Verified Typists (15 sessions each)
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
- **Target Cohort**: 60 participants with $\ge 15$ completed typing sessions each (total **42,929 raw keystrokes**).
- **Session Design**: 15 distinct sentence sessions per user $\rightarrow$ **900 total feature vectors**.

---

## 📊 Biometric Feature Architecture (124 Features)

| Category | Count | Description |
|---|:---:|---|
| **Global Timing Statistics** | 12 | Mean, standard deviation, median, and IQR of Hold Time ($HT$), Flight Time ($UD$), and Latency ($DD$). |
| **Cadence & Speed** | 6 | Total keystrokes, session duration, keys/sec, Words Per Minute (WPM), rollover ratio ($UD < 0$), cognitive pause rate. |
| **Error & Correction** | 5 | Backspace frequency, backspace rate ($BKSP / Total$), shift modifier rate, edit length delta. |
| **Top 25 Digraph Biometrics** | 100 | Mean $UD$, mean $DD$, mean $HT$, and frequency for the top 25 character transitions (`t_h`, `h_e`, `e_SPACE`, `i_n`, etc.). |

---

## 📁 Repository Structure

```
├── 01_data_merge/
│   ├── inspection_report.md          # Full dataset layout & field verification
│   ├── selected_participants.csv      # Metadata for the 60 experimental typists
│   ├── merged_raw_keystrokes.csv      # Merged raw telemetry data (42,939 rows)
│   └── merged_raw_keystrokes.parquet  # High-performance Parquet format
├── 02_cleaning_eda/
│   ├── cleaning_log.md               # Before/after row counts & anomaly justifications
│   ├── eda_summary.md                # Key empirical findings for methodology section
│   ├── cleaned_keystrokes.csv        # Cleaned dataset (99.98% retention)
│   └── plots/                        # High-resolution visualization artifacts
│       ├── overall_timing_distributions.png
│       ├── participant_overlay_distributions.png
│       ├── timing_correlation_heatmap.png
│       └── participant_session_balance.png
├── 03_feature_engineering_preprocessing/
│   ├── feature_list.md               # Complete documentation of all 124 features
│   ├── feature_table.csv             # Full session feature matrix (900 x 126)
│   ├── train.csv / test.csv          # Stratified 80/20 train/test splits (720 / 180 rows)
│   ├── train_scaled.csv / test_scaled.csv # StandardScaled feature partitions (Zero leakage)
│   ├── scaler.joblib                 # Serialized fitted StandardScaler object
│   ├── label_encoder.joblib          # Serialized LabelEncoder for 60 target classes
│   └── label_mapping.json            # Participant ID to class index dictionary
├── inspect_and_merge.py              # Automated data ingestion pipeline
├── clean_and_eda.py                  # Automated cleaning, timing extraction & EDA
├── feature_engineering_and_preprocessing.py # Feature engineering & preprocessing pipeline
└── README.md                         # Project documentation
```

---

## 🚀 Setup & Execution

```bash
# 1. Clone repository
git clone https://github.com/itsrashedsatter/thesis-on-keystroke-dynamics-authentication.git
cd thesis-on-keystroke-dynamics-authentication

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib pyarrow

# 3. Run pipelines
python inspect_and_merge.py
python clean_and_eda.py
python feature_engineering_and_preprocessing.py
```

---

## 👥 Authors & Academic Context
- **Researcher**: [Rashed Satter](https://github.com/itsrashedsatter)
- **Topic**: Behavior-Based Continuous Verification using Keystroke Dynamics
- **Domain**: Cybersecurity, Behavioral Biometrics, Zero Trust Architecture (ZTA)
