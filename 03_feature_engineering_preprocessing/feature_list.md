# Feature Engineering & Preprocessing Specification

## 1. Overview
- **Input Dataset**: `02_cleaning_eda/cleaned_keystrokes.csv` (42,929 raw keystrokes)
- **Engineered Feature Table**: `03_feature_engineering_preprocessing/feature_table.csv`
- **Granularity**: 1 feature vector per typing session (15 sessions $\times$ 60 participants = **900 total rows**)
- **Total Features**: **`124` numeric biometric features** + `PARTICIPANT_ID`, `TEST_SECTION_ID`, `label`
- **Train/Test Partition**: Stratified session-level split (80% Train: 720 rows, 20% Test: 180 rows, exactly 12 train and 3 test sessions per participant). Zero cross-split session leakage.

---

## 2. Feature Definitions & Groupings

### A. Global Keystroke Timing Statistics (12 Features)
Computed across all keystrokes in the session to capture overall neuromotor pace and consistency:
- `overall_HT_mean`, `overall_HT_std`, `overall_HT_median`, `overall_HT_iqr`: Central tendency and dispersion of key Hold Time (ms).
- `overall_UD_mean`, `overall_UD_std`, `overall_UD_median`, `overall_UD_iqr`: Central tendency and dispersion of Up-to-Down Flight Time (ms).
- `overall_DD_mean`, `overall_DD_std`, `overall_DD_median`, `overall_DD_iqr`: Central tendency and dispersion of Down-to-Down Latency (ms).

### B. Cadence, Duration & Speed Dynamics (6 Features)
Measures typing throughput, motor speed, and rhythm:
- `total_keystrokes`: Total number of key events in the session.
- `session_duration_sec`: Total duration of the active typing session in seconds.
- `keystrokes_per_sec`: Raw typing rate (keys/sec).
- `typing_speed_wpm`: Words Per Minute standard rate ($(\text{keys}/5) / (\text{duration}/60)$).
- `rollover_rate`: Proportion of keystroke transitions with negative flight time ($UD < 0$), reflecting multi-key overlap.
- `long_pause_count`, `long_pause_rate`: Frequency and proportion of cognitive pauses ($DD > 5,000$ ms).

### C. Error & Correction Dynamics (5 Features)
Captures individual error recovery habits and punctuation behaviors:
- `backspace_count`: Total backspace key presses (`KEYCODE == 8`).
- `backspace_rate`: Proportion of backspaces relative to total keystrokes ($BKSP / \text{Total Keys}$).
- `shift_count`: Total shift modifier key presses (`KEYCODE == 16`).
- `shift_rate`: Proportion of shift modifications.
- `char_length_diff`: Absolute character length discrepancy between prompt `SENTENCE` and typed `USER_INPUT`.

### D. Top 25 Digraph Biometrics (100 Features: 4 per Digraph)
The 25 most frequent character transitions across the cohort were selected:
`unknown_unknown, e_SPACE, BKSP_BKSP, SPACE_t, unknown_SPACE, t_h, SPACE_unknown, s_SPACE, t_SPACE, h_e, i_n, SPACE_a, d_SPACE, SPACE_w, SPACE_SHIFT, n_SPACE, r_e, e_r, o_u, o_n, SPACE_i, a_n, o_SPACE, y_SPACE, r_SPACE`

For each digraph `d`, four biometric metrics are engineered:
1. `dg_{d}_count`: Frequency of occurrence in the session.
2. `dg_{d}_mean_UD`: Mean Up-to-Down flight time for transition `d`.
3. `dg_{d}_mean_DD`: Mean Down-to-Down latency for transition `d`.
4. `dg_{d}_mean_HT`: Mean Hold Time of the trailing key in transition `d`.

---

## 3. Missing Value Imputation Strategy for Sparse Digraphs
In short sentences, certain digraphs may not appear. A **Two-Tier Hierarchical Imputation Strategy** was deployed:
1. **Tier 1 (Personal Baseline)**: Impute missing digraph metrics with the participant's historical mean for that specific digraph across their remaining sessions.
2. **Tier 2 (Cohort Baseline)**: If the participant never executed that digraph across all 15 sessions, impute with the global cohort median.
3. Count features (`dg_{d}_count`) are explicitly set to `0` when absent.

---

## 4. Preprocessing & Normalization Rationale
- **Target Encoding**: `PARTICIPANT_ID` is mapped to integer labels $y \in [0, 59]$ using `LabelEncoder`.
- **Scaling (`StandardScaler`)**: Features are standardized ($z = (x - \mu) / \sigma$) using `StandardScaler` fitted **strictly on the Training partition (720 samples)** and applied to the Test partition (180 samples) to guarantee zero data leakage.
- **Suitability**: StandardScaler is optimal for gradient-based Deep Learning models (1D-CNN, LSTM) and distance-based classifiers (SVM, kNN), while preserving feature interpretability for tree-based baselines (Random Forest, XGBoost).

---

## 5. Preprocessed Artifacts
- **Full Feature Table**: [`03_feature_engineering_preprocessing/feature_table.csv`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/feature_table.csv)
- **Train Set (Unscaled)**: [`03_feature_engineering_preprocessing/train.csv`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/train.csv)
- **Test Set (Unscaled)**: [`03_feature_engineering_preprocessing/test.csv`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/test.csv)
- **Train Set (StandardScaled)**: [`03_feature_engineering_preprocessing/train_scaled.csv`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/train_scaled.csv)
- **Test Set (StandardScaled)**: [`03_feature_engineering_preprocessing/test_scaled.csv`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/test_scaled.csv)
- **Saved Scaler**: [`03_feature_engineering_preprocessing/scaler.joblib`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/scaler.joblib)
- **Saved Label Encoder**: [`03_feature_engineering_preprocessing/label_encoder.joblib`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/label_encoder.joblib)
- **Label Mapping**: [`03_feature_engineering_preprocessing/label_mapping.json`](file:///f:/thesis%20original/03_feature_engineering_preprocessing/label_mapping.json)
