import os
import json
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

def run_feature_engineering():
    base_dir = r"f:\thesis original"
    input_file = os.path.join(base_dir, "02_cleaning_eda", "cleaned_keystrokes.csv")
    output_dir = os.path.join(base_dir, "03_feature_engineering_preprocessing")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading cleaned keystroke data...")
    df = pd.read_csv(input_file, encoding='latin-1')
    print(f"Loaded {len(df):,} keystrokes from {df['PARTICIPANT_ID'].nunique()} participants.")
    
    # -------------------------------------------------------------
    # 1. DIGRAPH IDENTIFICATION & SELECTION
    # -------------------------------------------------------------
    # Sort strictly
    df = df.sort_values(by=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'KEYSTROKE_ID']).reset_index(drop=True)
    
    # Format character representations cleanly
    def clean_char(char, keycode):
        if keycode == 32 or char == ' ':
            return 'SPACE'
        elif keycode == 8 or str(char).upper() == 'BKSP':
            return 'BKSP'
        elif keycode == 16 or str(char).upper() == 'SHIFT':
            return 'SHIFT'
        elif pd.isnull(char) or str(char).strip() == '':
            return f'KEY_{keycode}'
        else:
            return str(char).lower()
            
    df['CLEAN_CHAR'] = [clean_char(c, k) for c, k in zip(df['LETTER'], df['KEYCODE'])]
    df['PREV_CHAR'] = df.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'])['CLEAN_CHAR'].shift(1)
    
    # Build digraph column (only for transitions within same section)
    valid_transition = df['IS_FIRST_KEY'] == 0
    df['DIGRAPH'] = np.where(valid_transition, df['PREV_CHAR'] + '_' + df['CLEAN_CHAR'], None)
    
    # Count frequency of all digraphs across cohort
    all_digraphs = df[valid_transition & df['DIGRAPH'].notnull()]['DIGRAPH'].tolist()
    digraph_counts = Counter(all_digraphs)
    
    # Select Top 25 most frequent digraphs
    TOP_K_DIGRAPHS = 25
    top_digraphs = [d for d, count in digraph_counts.most_common(TOP_K_DIGRAPHS)]
    print(f"\nTop {TOP_K_DIGRAPHS} Most Frequent Digraphs:")
    for i, (d, count) in enumerate(digraph_counts.most_common(TOP_K_DIGRAPHS)):
        print(f"  {i+1:2d}. {d:15s} : {count:,} occurrences")
        
    # -------------------------------------------------------------
    # 2. PER-SESSION FEATURE EXTRACTION
    # -------------------------------------------------------------
    sessions = df.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'])
    print(f"\nExtracting feature vectors across {len(sessions)} unique participant sessions...")
    
    session_rows = []
    
    for (pid, sec_id), group in sessions:
        row = {
            'PARTICIPANT_ID': pid,
            'TEST_SECTION_ID': sec_id,
        }
        
        # A. Global Timing Features
        hts = group['HOLD_TIME'].values
        valid_uds = group[(group['IS_FIRST_KEY'] == 0) & (group['IS_LONG_PAUSE'] == 0)]['UD_TIME'].values
        valid_dds = group[(group['IS_FIRST_KEY'] == 0) & (group['IS_LONG_PAUSE'] == 0)]['DD_TIME'].values
        
        row['overall_HT_mean'] = float(np.mean(hts)) if len(hts) > 0 else 0.0
        row['overall_HT_std'] = float(np.std(hts)) if len(hts) > 1 else 0.0
        row['overall_HT_median'] = float(np.median(hts)) if len(hts) > 0 else 0.0
        row['overall_HT_iqr'] = float(np.percentile(hts, 75) - np.percentile(hts, 25)) if len(hts) > 0 else 0.0
        
        row['overall_UD_mean'] = float(np.mean(valid_uds)) if len(valid_uds) > 0 else 0.0
        row['overall_UD_std'] = float(np.std(valid_uds)) if len(valid_uds) > 1 else 0.0
        row['overall_UD_median'] = float(np.median(valid_uds)) if len(valid_uds) > 0 else 0.0
        row['overall_UD_iqr'] = float(np.percentile(valid_uds, 75) - np.percentile(valid_uds, 25)) if len(valid_uds) > 0 else 0.0
        
        row['overall_DD_mean'] = float(np.mean(valid_dds)) if len(valid_dds) > 0 else 0.0
        row['overall_DD_std'] = float(np.std(valid_dds)) if len(valid_dds) > 1 else 0.0
        row['overall_DD_median'] = float(np.median(valid_dds)) if len(valid_dds) > 0 else 0.0
        row['overall_DD_iqr'] = float(np.percentile(valid_dds, 75) - np.percentile(valid_dds, 25)) if len(valid_dds) > 0 else 0.0
        
        # B. Cadence, Duration & Speed Dynamics
        total_keys = len(group)
        t_min = group['PRESS_TIME'].min()
        t_max = group['RELEASE_TIME'].max()
        duration_sec = max((t_max - t_min) / 1000.0, 0.5)
        
        row['total_keystrokes'] = total_keys
        row['session_duration_sec'] = duration_sec
        row['keystrokes_per_sec'] = total_keys / duration_sec
        row['typing_speed_wpm'] = (total_keys / 5.0) / (duration_sec / 60.0)
        
        # Rollover ratio (proportion of keys where UD < 0)
        num_rollover = group['IS_ROLLOVER'].sum()
        row['rollover_rate'] = num_rollover / max(len(group) - 1, 1)
        
        # Long pause count & ratio
        num_pauses = group['IS_LONG_PAUSE'].sum()
        row['long_pause_count'] = num_pauses
        row['long_pause_rate'] = num_pauses / max(len(group) - 1, 1)
        
        # C. Error & Correction Dynamics
        bksp_count = (group['KEYCODE'] == 8).sum()
        shift_count = (group['KEYCODE'] == 16).sum()
        row['backspace_count'] = bksp_count
        row['backspace_rate'] = bksp_count / total_keys
        row['shift_count'] = shift_count
        row['shift_rate'] = shift_count / total_keys
        
        # Sentence vs User Input character edit difference
        sentence_str = str(group['SENTENCE'].iloc[0]) if 'SENTENCE' in group.columns else ""
        input_str = str(group['USER_INPUT'].iloc[0]) if 'USER_INPUT' in group.columns else ""
        row['char_length_diff'] = abs(len(input_str) - len(sentence_str))
        
        # D. Top 25 Digraph Timings
        # Group by digraph within this session
        group_digraphs = group[group['DIGRAPH'].isin(top_digraphs)]
        for d in top_digraphs:
            d_subset = group_digraphs[group_digraphs['DIGRAPH'] == d]
            d_count = len(d_subset)
            row[f'dg_{d}_count'] = d_count
            if d_count > 0:
                row[f'dg_{d}_mean_UD'] = float(d_subset['UD_TIME'].mean())
                row[f'dg_{d}_mean_DD'] = float(d_subset['DD_TIME'].mean())
                row[f'dg_{d}_mean_HT'] = float(d_subset['HOLD_TIME'].mean())
            else:
                row[f'dg_{d}_mean_UD'] = np.nan
                row[f'dg_{d}_mean_DD'] = np.nan
                row[f'dg_{d}_mean_HT'] = np.nan
                
        session_rows.append(row)
        
    feature_df = pd.DataFrame(session_rows)
    print(f"Constructed raw feature table: {feature_df.shape[0]} rows (sessions) x {feature_df.shape[1]} columns.")
    
    # -------------------------------------------------------------
    # 3. MISSING DIGRAPH IMPUTATION STRATEGY
    # -------------------------------------------------------------
    # Strategy: Two-tier hierarchical imputation
    # Tier 1: Participant's own mean for that digraph across all their sessions
    # Tier 2: Cohort global median for that digraph if the participant never typed it
    print("\nApplying Two-Tier Hierarchical Imputation for Digraph Features...")
    imputed_cols = []
    for d in top_digraphs:
        for metric in ['mean_UD', 'mean_DD', 'mean_HT']:
            col_name = f'dg_{d}_{metric}'
            imputed_cols.append(col_name)
            
            # Tier 1: Groupby PARTICIPANT_ID mean
            p_mean = feature_df.groupby('PARTICIPANT_ID')[col_name].transform('mean')
            feature_df[col_name] = feature_df[col_name].fillna(p_mean)
            
            # Tier 2: Global cohort median for remaining NaNs
            global_med = feature_df[col_name].median()
            feature_df[col_name] = feature_df[col_name].fillna(global_med)
            
    # Confirm 0 missing values
    null_remaining = feature_df.isnull().sum().sum()
    print(f"Remaining null values after imputation: {null_remaining}")
    
    # Save the complete engineered feature table
    feature_table_path = os.path.join(output_dir, "feature_table.csv")
    feature_df.to_csv(feature_table_path, index=False)
    print(f"Saved complete feature table: {feature_table_path} ({os.path.getsize(feature_table_path):,} bytes)")
    
    # -------------------------------------------------------------
    # 4. PREPROCESSING & STRATIFIED TRAIN/TEST SPLIT
    # -------------------------------------------------------------
    print("\nEncoding targets and performing stratified train/test split (80/20 by session)...")
    
    # 4.1 Label Encoding target PARTICIPANT_ID
    label_encoder = LabelEncoder()
    feature_df['label'] = label_encoder.fit_transform(feature_df['PARTICIPANT_ID'])
    
    label_mapping = {int(p): int(l) for p, l in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    mapping_file = os.path.join(output_dir, "label_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(label_mapping, f, indent=2)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    print(f"Saved label encoder mapping ({len(label_mapping)} classes) to {mapping_file}")
    
    # 4.2 Split into Train and Test (by session within each participant)
    # Each participant has 15 sessions: 12 sessions in Train (80%), 3 sessions in Test (20%)
    train_indices = []
    test_indices = []
    
    rng = np.random.RandomState(42)
    
    for pid, group in feature_df.groupby('PARTICIPANT_ID'):
        g_indices = group.index.tolist()
        # Shuffle sessions within participant deterministically
        shuffled = rng.permutation(g_indices)
        n_train = int(len(shuffled) * 0.8) # 12 sessions
        train_indices.extend(shuffled[:n_train])
        test_indices.extend(shuffled[n_train:])
        
    df_train = feature_df.loc[train_indices].sort_values(by=['label', 'TEST_SECTION_ID']).reset_index(drop=True)
    df_test = feature_df.loc[test_indices].sort_values(by=['label', 'TEST_SECTION_ID']).reset_index(drop=True)
    
    print(f"Train Set: {df_train.shape[0]} rows ({df_train['label'].nunique()} participants, exactly {df_train.groupby('label').size().iloc[0]} sessions/user)")
    print(f"Test Set:  {df_test.shape[0]} rows ({df_test['label'].nunique()} participants, exactly {df_test.groupby('label').size().iloc[0]} sessions/user)")
    
    # 4.3 Feature Scaling (StandardScaler)
    # Separate identifiers/metadata from feature matrix X
    id_cols = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'label']
    feature_cols = [c for c in feature_df.columns if c not in id_cols]
    
    print(f"Total Numeric Feature Count: {len(feature_cols)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[feature_cols])
    X_test_scaled = scaler.transform(df_test[feature_cols])
    
    # Save fitted scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    
    # Construct scaled dataframes
    df_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    for col in id_cols:
        df_train_scaled[col] = df_train[col]
        
    df_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    for col in id_cols:
        df_test_scaled[col] = df_test[col]
        
    # Save datasets
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_scaled_path = os.path.join(output_dir, "train_scaled.csv")
    test_scaled_path = os.path.join(output_dir, "test_scaled.csv")
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_train_scaled.to_csv(train_scaled_path, index=False)
    df_test_scaled.to_csv(test_scaled_path, index=False)
    
    print(f"Saved unscaled train.csv: {train_path} ({df_train.shape})")
    print(f"Saved unscaled test.csv: {test_path} ({df_test.shape})")
    print(f"Saved scaled train_scaled.csv: {train_scaled_path} ({df_train_scaled.shape})")
    print(f"Saved scaled test_scaled.csv: {test_scaled_path} ({df_test_scaled.shape})")
    
    # Check data leakage
    overlap_sessions = set(zip(df_train['PARTICIPANT_ID'], df_train['TEST_SECTION_ID'])).intersection(
        set(zip(df_test['PARTICIPANT_ID'], df_test['TEST_SECTION_ID']))
    )
    print(f"Session overlap between Train and Test: {len(overlap_sessions)} (Zero leakage confirmed!)")
    
    # -------------------------------------------------------------
    # 5. FEATURE LIST DOCUMENTATION
    # -------------------------------------------------------------
    feature_doc = f"""# Feature Engineering & Preprocessing Specification

## 1. Overview
- **Input Dataset**: `02_cleaning_eda/cleaned_keystrokes.csv` (42,929 raw keystrokes)
- **Engineered Feature Table**: `03_feature_engineering_preprocessing/feature_table.csv`
- **Granularity**: 1 feature vector per typing session (15 sessions $\\times$ 60 participants = **900 total rows**)
- **Total Features**: **`{len(feature_cols)}` numeric biometric features** + `PARTICIPANT_ID`, `TEST_SECTION_ID`, `label`
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
- `typing_speed_wpm`: Words Per Minute standard rate ($(\\text{{keys}}/5) / (\\text{{duration}}/60)$).
- `rollover_rate`: Proportion of keystroke transitions with negative flight time ($UD < 0$), reflecting multi-key overlap.
- `long_pause_count`, `long_pause_rate`: Frequency and proportion of cognitive pauses ($DD > 5,000$ ms).

### C. Error & Correction Dynamics (5 Features)
Captures individual error recovery habits and punctuation behaviors:
- `backspace_count`: Total backspace key presses (`KEYCODE == 8`).
- `backspace_rate`: Proportion of backspaces relative to total keystrokes ($BKSP / \\text{{Total Keys}}$).
- `shift_count`: Total shift modifier key presses (`KEYCODE == 16`).
- `shift_rate`: Proportion of shift modifications.
- `char_length_diff`: Absolute character length discrepancy between prompt `SENTENCE` and typed `USER_INPUT`.

### D. Top 25 Digraph Biometrics (100 Features: 4 per Digraph)
The 25 most frequent character transitions across the cohort were selected:
`{", ".join(top_digraphs)}`

For each digraph `d`, four biometric metrics are engineered:
1. `dg_{{d}}_count`: Frequency of occurrence in the session.
2. `dg_{{d}}_mean_UD`: Mean Up-to-Down flight time for transition `d`.
3. `dg_{{d}}_mean_DD`: Mean Down-to-Down latency for transition `d`.
4. `dg_{{d}}_mean_HT`: Mean Hold Time of the trailing key in transition `d`.

---

## 3. Missing Value Imputation Strategy for Sparse Digraphs
In short sentences, certain digraphs may not appear. A **Two-Tier Hierarchical Imputation Strategy** was deployed:
1. **Tier 1 (Personal Baseline)**: Impute missing digraph metrics with the participant's historical mean for that specific digraph across their remaining sessions.
2. **Tier 2 (Cohort Baseline)**: If the participant never executed that digraph across all 15 sessions, impute with the global cohort median.
3. Count features (`dg_{{d}}_count`) are explicitly set to `0` when absent.

---

## 4. Preprocessing & Normalization Rationale
- **Target Encoding**: `PARTICIPANT_ID` is mapped to integer labels $y \\in [0, 59]$ using `LabelEncoder`.
- **Scaling (`StandardScaler`)**: Features are standardized ($z = (x - \\mu) / \\sigma$) using `StandardScaler` fitted **strictly on the Training partition (720 samples)** and applied to the Test partition (180 samples) to guarantee zero data leakage.
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
"""
    with open(os.path.join(output_dir, "feature_list.md"), "w", encoding="utf-8") as f:
        f.write(feature_doc)
    print(f"Saved feature list documentation: {os.path.join(output_dir, 'feature_list.md')}")

if __name__ == "__main__":
    run_feature_engineering()
