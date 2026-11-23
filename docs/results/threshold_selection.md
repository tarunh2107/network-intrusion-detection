# Phase 6B: Validation Threshold Selection Report

**Timestamp (UTC):** `2026-10-12T19:08:03.561388+00:00`  
**Validation Flows Evaluated:** `417,395`  
**Operating Selection Rule:**  
> *"Choose the threshold with highest validation ATTACK recall among thresholds with false-positive rate <= 0.2%. Break ties by higher precision, then higher threshold. (This is a project evaluation choice, not an industry standard or a guarantee about live networks.)"*

---

## 1. Selected Decision Threshold

- **Chosen Threshold:** `0.29`
- **FPR Budget Satisfied (FPR <= 0.20%):** `True`
- **Selection Rationale:** Highest validation ATTACK recall (0.9971) among thresholds with FPR <= 0.20% (actual FPR: 0.1958%). Tie-breakers: precision=0.9914, threshold=0.29.

---

## 2. Full Validation Performance: Default (0.50) vs Selected Threshold

| Metric | Default Threshold (0.50) | Selected Threshold (0.29) | Operational Impact |
| :--- | :--- | :--- | :--- |
| **ATTACK Precision** | 99.6204% | 99.1435% | Change in positive predictive value |
| **ATTACK Recall** | 99.4645% | 99.7103% | Change in proportion of attacks caught |
| **ATTACK F1-Score** | 0.9954 | 0.9943 | Harmonic mean |
| **False Positive Rate (FPR)** | 0.0862% | 0.1958% | Must remain <= 0.20% budget |
| **False Negative Rate (FNR)** | 0.5355% | 0.2897% | Missed attacks proportion |
| **Average Precision (AP)** | 0.9995 | 0.9995 | Overall PR curve metric |

### Confusion Matrix Comparison

| Threshold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.50** | 339,785 | 293 | 414 | 76,903 | 417,395 |
| **0.29** | 339,412 | 666 | 224 | 77,093 | 417,395 |

---

## 3. Per-Category Detection Breakdown on Full Validation Set (Threshold = 0.29)

| Category | Class Type | Support in Validation | Detected Correctly | Detection Rate / Recall | Missed | Notes / Uncertainty |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | Benign | 340,078 | 339,412 | **99.80%** | 666 | Standard sample |
| `Bot` | Attack | 293 | 174 | **59.39%** | 119 | Standard sample |
| `DDoS` | Attack | 19,178 | 19,161 | **99.91%** | 17 | Standard sample |
| `DoS GoldenEye` | Attack | 1,592 | 1,581 | **99.31%** | 11 | Standard sample |
| `DoS Hulk` | Attack | 28,211 | 28,169 | **99.85%** | 42 | Standard sample |
| `DoS Slowhttptest` | Attack | 834 | 829 | **99.40%** | 5 | Standard sample |
| `DoS slowloris` | Attack | 892 | 889 | **99.66%** | 3 | Standard sample |
| `FTP-Patator` | Attack | 1,154 | 1,154 | **100.00%** | 0 | Standard sample |
| `Heartbleed` | Attack | 1 | 0 | **0.00%** | 1 | Warning: Based on 1 example(s); high estimation variance |
| `Infiltration` | Attack | 6 | 0 | **0.00%** | 6 | Standard sample |
| `PortScan` | Attack | 23,859 | 23,857 | **99.99%** | 2 | Standard sample |
| `SSH-Patator` | Attack | 967 | 964 | **99.69%** | 3 | Standard sample |
| `Web Attack - Brute Force` | Attack | 237 | 226 | **95.36%** | 11 | Standard sample |
| `Web Attack - Sql Injection` | Attack | 3 | 3 | **100.00%** | 0 | Standard sample |
| `Web Attack - XSS` | Attack | 90 | 86 | **95.56%** | 4 | Standard sample |
