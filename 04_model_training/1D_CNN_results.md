# 1D_CNN — Evaluation Results (1000 Classes)

## 1. Task Configuration
- **Task**: Closed-set user identification (1000-class classification)
- **Random Chance Baseline**: `0.1000%` (1 / 1000)
- **Test Samples**: `3,000` sessions
- **Training Samples**: `12,000` sessions
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `0.5041` | `0.5041` |
| **Recall** | `0.4907` | `0.4907` |
| **F1-Score** | `0.4606` | `0.4606` |

- **Overall Accuracy**: `49.07%` (1,472/3,000 correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`1D_CNN_confusion_matrix.png`](1D_CNN_confusion_matrix.png)
- **Full Per-Class Report**: [`1D_CNN_classification_report.txt`](1D_CNN_classification_report.txt)
- **Machine-Readable Metrics**: [`1D_CNN_metrics.json`](1D_CNN_metrics.json)
