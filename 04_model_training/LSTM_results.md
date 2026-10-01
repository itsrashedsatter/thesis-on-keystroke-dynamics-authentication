# LSTM — Evaluation Results (1000 Classes)

## 1. Task Configuration
- **Task**: Closed-set user identification (1000-class classification)
- **Random Chance Baseline**: `0.1000%` (1 / 1000)
- **Test Samples**: `3,000` sessions
- **Training Samples**: `12,000` sessions
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `0.8948` | `0.8948` |
| **Recall** | `0.8697` | `0.8697` |
| **F1-Score** | `0.8649` | `0.8649` |

- **Overall Accuracy**: `86.97%` (2,609/3,000 correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`LSTM_confusion_matrix.png`](LSTM_confusion_matrix.png)
- **Full Per-Class Report**: [`LSTM_classification_report.txt`](LSTM_classification_report.txt)
- **Machine-Readable Metrics**: [`LSTM_metrics.json`](LSTM_metrics.json)
