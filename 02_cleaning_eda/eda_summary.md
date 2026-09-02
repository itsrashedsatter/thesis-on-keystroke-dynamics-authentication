# Exploratory Data Analysis (EDA) Summary

## 1. Overview of Key Findings
This document provides exploratory findings on keystroke biometric features extracted from the cleaned dataset of 60 participants (total `42,929` keystrokes). These insights support the foundational hypothesis that keystroke dynamics carry distinct, measurable biometric fingerprints for continuous user authentication.

---

## 2. Detailed Findings & Academic Synthesis

### Finding 1: Distinct Intra-User vs. Inter-User Timing Distributions
- **Observation**: As shown in the overlay density plots ([`plots/participant_overlay_distributions.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/participant_overlay_distributions.png)), individual users exhibit distinct, sharp probability density peaks for both Hold Time ($HT$) and Flight Time ($UD$). 
- **Biometric Significance**: While the population Hold Time clusters around a mean of `112.0` ms (std: `70.1` ms), individual participants maintain distinct mean hold times ranging from ~70 ms (fast, light touch typists) to >180 ms (deliberate typists). This variance confirms that neuromotor timing patterns are user-specific.

### Finding 2: Natural Key Rollover in Skilled Typing
- **Observation**: Over `11,166` keystroke transitions (approx. `26.0%` of valid transitions) exhibit negative Up-to-Down flight times ($UD < 0$).
- **Biometric Significance**: Negative UD times represent physical key rollover—where a typist initiates the next key press prior to releasing the current key during fluid digram execution (e.g., 'th', 'er'). Preserving these negative flight times rather than clipping them to zero retains critical typing fluidity signals.

### Finding 3: Mathematical & Empirical Relationships Between Timing Features
- **Observation**: The correlation heatmap ([`plots/timing_correlation_heatmap.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/timing_correlation_heatmap.png)) shows a very strong linear correlation between $UD$ (Flight Time) and $DD$ (Down-to-Down Latency) ($r = 0.943$), while Hold Time exhibits weak correlation with flight times ($r = -0.067$).
- **Biometric Significance**: Down-to-Down latency is mathematically $DD_i = UD_i + HT_{i-1}$. Because Hold Time and Flight Time capture distinct physical phenomena (finger pressure duration vs. hand movement travel time), both features provide complementary information for biometric classifier models.

### Finding 4: Class Balance & Session Regularity
- **Observation**: As illustrated in [`plots/participant_session_balance.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/participant_session_balance.png), all 60 participants completed exactly 15 test sentences, producing between 500 and 1,150 keystrokes per user (mean `715.5` keystrokes).
- **Biometric Significance**: The uniform number of test sessions provides balanced multi-class training and testing partitions, preventing majority-class bias during closed-set user identification and open-set authentication benchmark experiments.

---

## 3. Visual Artifacts Generated
1. **Overall Timing Distributions**: [`02_cleaning_eda/plots/overall_timing_distributions.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/overall_timing_distributions.png)
2. **Participant Overlay Distributions**: [`02_cleaning_eda/plots/participant_overlay_distributions.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/participant_overlay_distributions.png)
3. **Timing Correlation Heatmap**: [`02_cleaning_eda/plots/timing_correlation_heatmap.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/timing_correlation_heatmap.png)
4. **Participant Balance**: [`02_cleaning_eda/plots/participant_session_balance.png`](file:///f:/thesis%20original/02_cleaning_eda/plots/participant_session_balance.png)
