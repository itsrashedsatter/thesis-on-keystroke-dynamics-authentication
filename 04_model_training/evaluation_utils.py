"""
evaluation_utils.py — Shared Evaluation Module for 1,000-User Model Training
=============================================================================
Provides a reusable function `evaluate_predictions()` that computes:
  - Accuracy, macro and weighted precision/recall/F1
  - Generates a confusion matrix heatmap (saved as PNG)
  - Saves a full scikit-learn classification report
  - Saves machine-readable JSON metrics and markdown summary

Dynamically scales to 1,000+ classes without hardcoding.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_predictions(y_true, y_pred, label_encoder, model_name, output_dir, train_samples=12000):
    """
    Evaluate model predictions and save all results to disk.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth integer labels.
    y_pred : array-like of shape (n_samples,)
        Predicted integer labels from the model.
    label_encoder : sklearn.preprocessing.LabelEncoder
        The fitted LabelEncoder used in Stage 03.
    model_name : str
        Identifier for the model (e.g. "Random_Forest", "1D_CNN", "LSTM").
    output_dir : str
        Directory path where output artifacts will be saved.
    train_samples : int
        Number of training samples used.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # -----------------------------------------------------------------
    # 1. Compute Aggregate Metrics
    # -----------------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    num_classes = len(label_encoder.classes_)
    total_samples = len(y_true)
    correct = int(np.sum(y_true == y_pred))

    metrics = {
        "model_name": model_name,
        "total_test_samples": total_samples,
        "total_train_samples": train_samples,
        "num_classes": num_classes,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 6),
        "test_accuracy": round(accuracy, 6),
        "precision_macro": round(precision_macro, 6),
        "recall_macro": round(recall_macro, 6),
        "f1_macro": round(f1_macro, 6),
        "macro_f1": round(f1_macro, 6),
        "precision_weighted": round(precision_weighted, 6),
        "recall_weighted": round(recall_weighted, 6),
        "f1_weighted": round(f1_weighted, 6),
        "weighted_f1": round(f1_weighted, 6),
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Metrics ({num_classes} Classes)")
    print(f"{'='*60}")
    print(f"  Total Test Samples:     {total_samples:,}")
    print(f"  Correct Predictions:    {correct:,} / {total_samples:,}")
    print(f"  Overall Accuracy:       {accuracy*100:.2f}%")
    print(f"  Random Chance Baseline: {100.0/num_classes:.4f}%")
    print(f"  Macro Precision:        {precision_macro:.4f}")
    print(f"  Macro Recall:           {recall_macro:.4f}")
    print(f"  Macro F1-Score:         {f1_macro:.4f}")
    print(f"  Weighted F1-Score:      {f1_weighted:.4f}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------
    # 2. Per-Class Classification Report
    # -----------------------------------------------------------------
    target_names = [str(c) for c in label_encoder.classes_]
    report_str = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name} — Full Classification Report ({num_classes} Classes)\n")
        f.write(f"{'='*60}\n\n")
        f.write(report_str)
    print(f"Saved classification report: {report_path}")

    # -----------------------------------------------------------------
    # 3. Confusion Matrix Heatmap
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Hide individual tick labels if more than 80 classes to prevent black smear
    show_ticks = num_classes <= 80
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names if show_ticks else False,
        yticklabels=target_names if show_ticks else False,
        cbar_kws={'label': 'Prediction Count'},
        ax=ax,
    )
    ax.set_xlabel('Predicted Class Index', fontsize=12)
    ax.set_ylabel('True Class Index', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix ({num_classes}-Class Identification)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")

    # -----------------------------------------------------------------
    # 4. Machine-Readable Metrics (JSON)
    # -----------------------------------------------------------------
    json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {json_path}")

    # -----------------------------------------------------------------
    # 5. Markdown Results Summary
    # -----------------------------------------------------------------
    md_content = f"""# {model_name} — Evaluation Results ({num_classes} Classes)

## 1. Task Configuration
- **Task**: Closed-set user identification ({num_classes}-class classification)
- **Random Chance Baseline**: `{100.0/num_classes:.4f}%` (1 / {num_classes})
- **Test Samples**: `{total_samples:,}` sessions
- **Training Samples**: `{train_samples:,}` sessions
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `{precision_macro:.4f}` | `{precision_weighted:.4f}` |
| **Recall** | `{recall_macro:.4f}` | `{recall_weighted:.4f}` |
| **F1-Score** | `{f1_macro:.4f}` | `{f1_weighted:.4f}` |

- **Overall Accuracy**: `{accuracy*100:.2f}%` ({correct:,}/{total_samples:,} correct)

## 3. Visual Artifacts
- **Confusion Matrix**: [`{model_name}_confusion_matrix.png`]({model_name}_confusion_matrix.png)
- **Full Per-Class Report**: [`{model_name}_classification_report.txt`]({model_name}_classification_report.txt)
- **Machine-Readable Metrics**: [`{model_name}_metrics.json`]({model_name}_metrics.json)
"""
    md_path = os.path.join(output_dir, f"{model_name}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved results summary: {md_path}")

    return metrics
