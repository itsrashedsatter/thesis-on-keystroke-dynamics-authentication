# Random_Forest — Evaluation Results (1000 Classes)

## 1. Task Configuration
- **Task**: Closed-set user identification (1000-class classification)
- **Random Chance Baseline**: `0.1000%` (1 / 1000)
- **Test Samples**: `3,000` sessions
- **Training Samples**: `12,000` sessions
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `0.9753` | `0.9753` |
| **Recall** | `0.9700` | `0.9700` |
| **F1-Score** | `0.9687` | `0.9687` |

- **Overall Accuracy**: `97.00%` (2,910/3,000 correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`Random_Forest_confusion_matrix.png`](Random_Forest_confusion_matrix.png)
- **Full Per-Class Report**: [`Random_Forest_classification_report.txt`](Random_Forest_classification_report.txt)
- **Machine-Readable Metrics**: [`Random_Forest_metrics.json`](Random_Forest_metrics.json)
