# Dataset Inspection Report

## 1. Dataset Overview & File Structure
- **Dataset Source**: Aalto University 136 Million Keystrokes Dataset (Kaggle Mirror: `noelsmathew/keystrokes`)
- **Dataset Root Directory**: `f:\thesis original\Thesis P2 Keyboard dynamics dataset\Keystrokes\files`
- **Total Participant Keystroke Files**: `168,593` individual `_keystrokes.txt` files
- **Participant Metadata File**: `metadata_participants.txt` (21,255,894 bytes, 168,594 participant records)

## 2. Metadata Columns (`metadata_participants.txt`)
The metadata file contains demographic, setup, and typing benchmark statistics:
- **Columns (16)**:
  `PARTICIPANT_ID, AGE, GENDER, HAS_TAKEN_TYPING_COURSE, COUNTRY, LAYOUT, NATIVE_LANGUAGE, FINGERS, TIME_SPENT_TYPING, KEYBOARD_TYPE, ERROR_RATE, AVG_WPM_15, AVG_IKI, ECPC, KSPC, ROR`

| Column | Description |
|---|---|
| `PARTICIPANT_ID` | Unique identifier for each participant |
| `AGE` | Participant age |
| `GENDER` | Gender category (male, female, none, etc.) |
| `HAS_TAKEN_TYPING_COURSE` | Binary flag (0 or 1) |
| `COUNTRY` | Two-letter country code (e.g. US, MY, IN, PH) |
| `LAYOUT` | Keyboard layout (e.g. qwerty) |
| `NATIVE_LANGUAGE` | Primary native language code |
| `FINGERS` | Typing finger style category (e.g. '7-8', '1-2', '9-10') |
| `TIME_SPENT_TYPING` | Self-reported daily typing hours |
| `KEYBOARD_TYPE` | Hardware form factor ('laptop', 'full', etc.) |
| `ERROR_RATE` | Aggregate error rate percentage |
| `AVG_WPM_15` | Average Words Per Minute across 15 sentences |
| `AVG_IKI` | Average Inter-Key Interval (ms) |
| `ECPC` | Error Corrections Per Character |
| `KSPC` | Keystrokes Per Character |
| `ROR` | Ratio of Replacement |

## 3. Sample Keystroke File Inspection
Three sample files across the dataset were inspected:

### Sample 1: `100001_keystrokes.txt`
- **File Size**: `97,304` bytes
- **Row Count (Keystrokes)**: `658` rows
- **Unique Test Sections (Sessions)**: `15` sections
- **Column Names & Data Types**:
  - `PARTICIPANT_ID`: `int64`
  - `TEST_SECTION_ID`: `int64`
  - `SENTENCE`: `str`
  - `USER_INPUT`: `str`
  - `KEYSTROKE_ID`: `int64`
  - `PRESS_TIME`: `int64`
  - `RELEASE_TIME`: `int64`
  - `LETTER`: `str`
  - `KEYCODE`: `int64`

### Sample 2: `330188_keystrokes.txt`
- **File Size**: `152,769` bytes
- **Row Count (Keystrokes)**: `921` rows
- **Unique Test Sections (Sessions)**: `15` sections
- **Column Names & Data Types**:
  - `PARTICIPANT_ID`: `int64`
  - `TEST_SECTION_ID`: `int64`
  - `SENTENCE`: `str`
  - `USER_INPUT`: `str`
  - `KEYSTROKE_ID`: `int64`
  - `PRESS_TIME`: `int64`
  - `RELEASE_TIME`: `int64`
  - `LETTER`: `str`
  - `KEYCODE`: `int64`

### Sample 3: `99998_keystrokes.txt`
- **File Size**: `96,339` bytes
- **Row Count (Keystrokes)**: `660` rows
- **Unique Test Sections (Sessions)**: `15` sections
- **Column Names & Data Types**:
  - `PARTICIPANT_ID`: `int64`
  - `TEST_SECTION_ID`: `int64`
  - `SENTENCE`: `str`
  - `USER_INPUT`: `str`
  - `KEYSTROKE_ID`: `int64`
  - `PRESS_TIME`: `int64`
  - `RELEASE_TIME`: `int64`
  - `LETTER`: `str`
  - `KEYCODE`: `int64`

## 4. Keystroke Volume & Field Verification
- **Keystroke Count per Participant**:
  - Sample Mean: `720.7` keystrokes
  - Sample Median: `716.5` keystrokes
  - Range: `469` to `1,143` keystrokes
  - Each standard completed session consists of 15 test sections (sentences), yielding ~600–1,200 raw keystroke events.

- **Field Presence Verification**:
| Expected Field | Actual Column Found | Verified |
|---|---|:---:|
| Participant Identifier | `PARTICIPANT_ID` | Yes |
| Session / Sentence Identifier | `TEST_SECTION_ID` | Yes |
| Target Sentence | `SENTENCE` | Yes |
| Actual User Input | `USER_INPUT` | Yes |
| Keystroke Event ID | `KEYSTROKE_ID` | Yes |
| Key Press Timestamp | `PRESS_TIME` (Unix timestamp in ms) | Yes |
| Key Release Timestamp | `RELEASE_TIME` (Unix timestamp in ms) | Yes |
| Key Character / Symbol | `LETTER` | Yes |
| Key Code | `KEYCODE` | Yes |

## 5. Subset Selection & Merging Configuration
- **Criteria**: Participants with at least 15 completed typing sessions (`TEST_SECTION_ID`).
- **Target Subset Size ($K$)**: `60` participants.
- **Random Seed**: `42` (Strictly reproducible).
- **Selected Participant IDs (60)**:
  `1011, 10019, 10023, 10112, 100001, 100007, 100031, 100033, 100068, 100081, 100102, 100108, 100168, 100188, 100210, 100215, 100224, 100232, 100249, 100262, 100281, 100306, 100327, 100381, 100397, 100446, 100461, 100495, 100512, 100519, 100542, 100545, 100561, 100576, 100580, 100773, 100852, 100898, 100899, 100904, 100994, 101009, 101034, 101039, 101045, 101073, 101079, 101104, 101124, 101129, 101130, 101144, 101226, 101256, 101292, 101323, 101328, 101329, 101336, 101338`
- **Merged Dataset Summary**:
  - Total Keystroke Rows: `42,939`
  - Total Columns: `24` (Keystroke telemetry + Participant Metadata)
  - Output Files:
    - `01_data_merge/merged_raw_keystrokes.csv` (11,842,970 bytes)
    - `01_data_merge/merged_raw_keystrokes.parquet` (978,997 bytes)
    - `01_data_merge/selected_participants.csv` (8,278 bytes)
