# Random_Forest — Evaluation Results

## 1. Task Configuration
- **Task**: Closed-set user identification (60-class classification)
- **Test Samples**: `180` sessions (3 per participant x 60 participants)
- **Training Samples**: `720` sessions (12 per participant x 60 participants)
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `0.9958` | `0.9958` |
| **Recall** | `0.9944` | `0.9944` |
| **F1-Score** | `0.9943` | `0.9943` |

- **Overall Accuracy**: `0.9944` (179/180 correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`Random_Forest_confusion_matrix.png`](Random_Forest_confusion_matrix.png)
- **Full Per-Class Report**: [`Random_Forest_classification_report.txt`](Random_Forest_classification_report.txt)
- **Machine-Readable Metrics**: [`Random_Forest_metrics.json`](Random_Forest_metrics.json)
