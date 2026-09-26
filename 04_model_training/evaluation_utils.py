"""
evaluation_utils.py — Shared Evaluation Module for Stage 04 Model Training
===========================================================================
Provides a single reusable function `evaluate_predictions()` that computes
accuracy, macro/weighted precision/recall/F1, generates a confusion matrix
heatmap (saved as PNG), writes a full sklearn classification report, and
saves all metrics to a {model_name}_results.md file.

This module is imported identically by train_random_forest.py, train_cnn.py,
and train_lstm.py so that evaluation is directly comparable across models.

Usage:
    from evaluation_utils import evaluate_predictions
    evaluate_predictions(y_true, y_pred, label_encoder, "Random_Forest", output_dir)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/Colab environments
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


def evaluate_predictions(y_true, y_pred, label_encoder, model_name, output_dir):
    """
    Evaluate model predictions and save all results to disk.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth integer labels (encoded via LabelEncoder).
    y_pred : array-like of shape (n_samples,)
        Predicted integer labels from the model.
    label_encoder : sklearn.preprocessing.LabelEncoder
        The fitted LabelEncoder used in Stage 03 (maps integer labels back
        to original PARTICIPANT_IDs for human-readable reports).
    model_name : str
        A short identifier for the model (e.g. "Random_Forest", "1D_CNN",
        "LSTM"). Used in filenames and report titles.
    output_dir : str
        Directory path where all output files will be saved.

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metric values.

    Side Effects
    ------------
    Writes the following files into `output_dir`:
        - {model_name}_confusion_matrix.png  (high-res heatmap)
        - {model_name}_classification_report.txt  (full per-class report)
        - {model_name}_results.md  (formatted markdown summary)
        - {model_name}_metrics.json  (machine-readable metrics)
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
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:            {accuracy:.4f}  ({correct}/{total_samples})")
    print(f"  Precision (macro):   {precision_macro:.4f}")
    print(f"  Recall    (macro):   {recall_macro:.4f}")
    print(f"  F1-Score  (macro):   {f1_macro:.4f}")
    print(f"  Precision (weighted):{precision_weighted:.4f}")
    print(f"  Recall    (weighted):{recall_weighted:.4f}")
    print(f"  F1-Score  (weighted):{f1_weighted:.4f}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # 2. Full Per-Class Classification Report
    # -----------------------------------------------------------------
    target_names = [str(c) for c in label_encoder.classes_]
    report_str = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name} — Full Classification Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(report_str)
    print(f"Saved classification report: {report_path}")

    # -----------------------------------------------------------------
    # 3. Confusion Matrix Heatmap
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm,
        annot=False,        # 60x60 matrix is too dense for annotations
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={'label': 'Prediction Count'},
        ax=ax,
    )
    ax.set_xlabel('Predicted Participant ID', fontsize=12)
    ax.set_ylabel('True Participant ID', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix (60-Class Identification)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
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
    md_content = f"""# {model_name} — Evaluation Results

## 1. Task Configuration
- **Task**: Closed-set user identification (60-class classification)
- **Test Samples**: `{total_samples}` sessions (3 per participant x 60 participants)
- **Training Samples**: `720` sessions (12 per participant x 60 participants)
- **Feature Count**: `124` biometric features per session

## 2. Aggregate Performance Metrics

| Metric | Macro Average | Weighted Average |
|---|:---:|:---:|
| **Precision** | `{precision_macro:.4f}` | `{precision_weighted:.4f}` |
| **Recall** | `{recall_macro:.4f}` | `{recall_weighted:.4f}` |
| **F1-Score** | `{f1_macro:.4f}` | `{f1_weighted:.4f}` |

- **Overall Accuracy**: `{accuracy:.4f}` ({correct}/{total_samples} correct)

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
