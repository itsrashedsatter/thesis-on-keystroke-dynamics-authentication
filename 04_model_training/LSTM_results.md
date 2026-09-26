# LSTM — Evaluation Results

## 1. Task Configuration
- **Task**: Closed-set user identification (60-class classification)
- **Test Samples**: `180` sessions (3 per participant x 60 participants)
- **Training Samples**: `720` sessions (12 per participant x 60 participants)
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `0.8928` | `0.8928` |
| **Recall** | `0.8667` | `0.8667` |
| **F1-Score** | `0.8599` | `0.8599` |

- **Overall Accuracy**: `0.8667` (156/180 correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`LSTM_confusion_matrix.png`](LSTM_confusion_matrix.png)
- **Full Per-Class Report**: [`LSTM_classification_report.txt`](LSTM_classification_report.txt)
- **Machine-Readable Metrics**: [`LSTM_metrics.json`](LSTM_metrics.json)
