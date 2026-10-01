# Consolidated Benchmark Report: 1,000-User Keystroke Authentication

## 1. Experimental Overview
- **Task**: Closed-set user identification across **$C = 1,000$ participants**
- **Evaluation Set**: Held-out **3,000 test sessions** (3 sessions per participant across all 1,000 users)
- **Training Set**: **12,000 sessions** (12 sessions per participant across all 1,000 users)
- **Feature Space**: **124 numeric biometric features** per session vector
- **Random Chance Baseline**: **$1 / 1{,}000 = 0.1000\%$**

---

## 2. Multi-Model Benchmark Comparison (1,000 Classes)

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Correct / 3,000 | Factor Over Random ($0.1\%$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🌲 **Random Forest** | **97.00%** | **0.9753** | **0.9700** | **0.9687** | **2,910 / 3,000** | **$970\times$** |
| 🧠 **Bidirectional LSTM** | **86.97%** | **0.8948** | **0.8697** | **0.8649** | **2,609 / 3,000** | **$870\times$** |
| ⚡ **1D-CNN** | **49.07%** | **0.5041** | **0.4907** | **0.4606** | **1,472 / 3,000** | **$490\times$** |
| 🎲 *Random Guess Baseline* | *0.10%* | — | — | — | *3 / 3,000* | *$1\times$* |

---

## 3. Scaling Comparison: 60 Users vs. 1,000 Users ($16.6\times$ Expansion)

| Model Architecture | 60-User Baseline Accuracy | **1,000-User Scale Accuracy** | Impact of $16.6\times$ Scale |
|---|:---:|:---:|:---:|
| **Random Forest** | 99.44% | **97.00%** | Drops only **-2.44%** despite 940 additional competing classes! |
| **Bidirectional LSTM** | 86.67% | **86.97%** | Gains **+0.30%** (improved generalization with 12,000 training samples). |
| **1D-CNN** | 65.56% | **49.07%** | Drops **-16.49%** (convolutional filters struggle with extreme class density). |

---

## 4. Top 10 Most Discriminative Biometric Features (1,000 Typists)

Ranked by Gini feature importance from the Random Forest model:

| Rank | Feature Name | Category | Gini Importance | Biometric Interpretation |
|:---:|---|---|:---:|---|
| **1** | `dg_BKSP_BKSP_mean_HT` | Error Recovery | **0.025558** | Hold time when double-tapping backspace to delete typos. |
| **2** | `dg_SPACE_SHIFT_mean_HT` | Capitalization | **0.024400** | Hold time of the Shift key following a space (sentence/word starts). |
| **3** | `dg_SPACE_SHIFT_mean_UD` | Capitalization | **0.022876** | Travel flight time from spacebar to Shift modifier. |
| **4** | `dg_SPACE_SHIFT_mean_DD` | Capitalization | **0.022537** | Down-to-down latency from spacebar to Shift modifier. |
| **5** | `dg_BKSP_BKSP_mean_UD` | Error Recovery | **0.021607** | Flight time between consecutive backspace presses. |
| **6** | `dg_BKSP_BKSP_mean_DD` | Error Recovery | **0.021438** | Down-to-down repetition frequency of backspacing. |
| **7** | `dg_o_SPACE_mean_HT` | Word Boundary | **0.019683** | Dwell duration when finishing words ending in 'o'. |
| **8** | `dg_o_SPACE_mean_UD` | Word Boundary | **0.019653** | Flight time from 'o' to the spacebar. |
| **9** | `dg_o_u_mean_UD` | Digraph Cadence | **0.019124** | Flight time between 'o' and 'u' keys. |
| **10** | `dg_i_s_mean_HT` | Digraph Cadence | **0.017521** | Dwell duration when typing the high-frequency 'is' transition. |
