# Data Cleaning Log

### Initial State
- **Total Raw Rows**: `42,939`
- **Total Unique Participants**: `60`

### Missing Values Analysis & Handling
- **`LETTER`**: `4,416` missing (10.28%).
  - *Resolution*: Missing `LETTER` values corresponding to `KEYCODE == 32` were imputed as space character `' '`. Any residual null characters were labeled `'UNKNOWN'` to preserve keystroke event continuity.

### Duplicate Removal
- **Duplicate Rows Found**: `0` duplicate records. Keystroke telemetry unique per participant session.

### Physical Validity & Anomaly Handling
- **Negative / Zero Hold Times Dropped ($HT \le 0$)**: `3` rows.
  - *Reasoning*: A key release timestamp occurring at or before its press timestamp represents hardware driver clock jitter or timer desynchronization.
- **Extreme Hold Time Artifacts Dropped ($HT > 3,000$ ms)**: `7` rows.
  - *Reasoning*: Dwell times exceeding 3 seconds reflect stuck keys or disconnected web sockets, which distort true motor dynamic distributions.
- **Long Pauses Flagged ($DD > 5,000$ ms)**: `17` occurrences flagged via `IS_LONG_PAUSE = 1`.
  - *Reasoning*: Pauses greater than 5 seconds represent cognitive pauses or reading delays. They are flagged separately rather than deleted to maintain full transcript fidelity while allowing timing models to filter out inter-sentence breaks.
- **Key Rollover Events ($UD < 0$)**: `11,166` occurrences flagged via `IS_ROLLOVER = 1`.
  - *Reasoning*: Negative Up-to-Down intervals naturally occur when a skilled typist presses a subsequent key before fully releasing the preceding one (multi-key rollover). These are preserved as genuine behavioral signatures.

### Summary of Row Counts
- **Initial Raw Rows**: `42,939`
- **Total Rows Dropped**: `10` (0.02%)
- **Final Cleaned Rows**: `42,929`
- **Retention Rate**: `99.98%`
- **Cleaned Dataset File**: `02_cleaning_eda/cleaned_keystrokes.csv`
