# 1D_CNN — Evaluation Results

## 1. Task Configuration
- **Task**: Closed-set user identification (60-class classification)
- **Test Samples**: `180` sessions (3 per participant x 60 participants)
- **Training Samples**: `720` sessions (12 per participant x 60 participants)
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `0.6457` | `0.6457` |
| **Recall** | `0.6556` | `0.6556` |
| **F1-Score** | `0.6120` | `0.6120` |

- **Overall Accuracy**: `0.6556` (118/180 correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`1D_CNN_confusion_matrix.png`](1D_CNN_confusion_matrix.png)
- **Full Per-Class Report**: [`1D_CNN_classification_report.txt`](1D_CNN_classification_report.txt)
- **Machine-Readable Metrics**: [`1D_CNN_metrics.json`](1D_CNN_metrics.json)
