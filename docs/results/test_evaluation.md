# Phase 6B: Held-Out Test Partition Evaluation Report

**Timestamp (UTC):** `2026-10-12T19:08:03.561388+00:00`  
**Model Artifact:** `models/baseline_v1/model.joblib` (SHA-256: `4339282ba6ce68b5c44c6dd6a72e7796e755a77542d65021984dd1a8b0434793`)  
**Frozen Decision Threshold:** `0.29`  
**Operating Selection Rule:** *"Choose the threshold with highest validation ATTACK recall among thresholds with false-positive rate <= 0.2%. Break ties by higher precision, then higher threshold. (This is a project evaluation choice, not an industry standard or a guarantee about live networks.)"*  
**Total Held-Out Test Flows:** `416,144` (`339,256` BENIGN, `76,888` ATTACK)  

---

## 1. Executive Summary

The held-out test partition (`data/processed/v1/test/`) was evaluated **exactly once** using the frozen model and threshold. The model parameters, preprocessing pipeline, and decision threshold were strictly locked before opening any test shards.

- **ATTACK Precision:** `99.1723%`
- **ATTACK Recall:** `99.7308%`
- **ATTACK F1-Score:** `0.9945`
- **False Positive Rate (FPR):** `0.1886%` (Budget requirement: `<= 0.20%`)
- **False Negative Rate (FNR):** `0.2692%`
- **Average Precision (AP):** `0.9996` (Scikit-learn `average_precision_score`)

---

## 2. Test Set Confusion Matrix (Threshold = 0.29)

| Actual \ Predicted | Predicted BENIGN (0) | Predicted ATTACK (1) | Total Actual | Class Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Actual BENIGN (0)** | **338,616** (TN) | **640** (FP) | 339,256 | Specificity: 99.81% (FPR: 0.1886%) |
| **Actual ATTACK (1)** | **207** (FN) | **76,681** (TP) | 76,888 | Recall: 99.73% (FNR: 0.2692%) |
| **Total Predicted** | 338,823 | 77,321 | 416,144 | Accuracy: 99.7965% |

---

## 3. Threshold Comparison on Held-Out Test Partition

| Metric | Frozen Threshold (0.29) | Default Threshold (0.50) | Difference |
| :--- | :--- | :--- | :--- |
| **ATTACK Precision** | **99.1723%** | 99.6060% | -0.4337% |
| **ATTACK Recall** | **99.7308%** | 99.2860% | +0.4448% |
| **ATTACK F1-Score** | **0.9945** | 0.9945 | +0.0001 |
| **False Positive Rate (FPR)** | **0.1886%** | 0.0890% | +0.0996% |
| **False Negative Rate (FNR)** | **0.2692%** | 0.7140% | -0.4448% |

---

## 4. Per-Category Detection Breakdown on Held-Out Test Set (Threshold = 0.29)

> [!IMPORTANT]
> This binary classifier outputs predictions strictly for `BENIGN` vs `ATTACK`. Category breakdown is computed strictly by grouping test flows by their ground-truth metadata display labels.

| Category | Class Type | Support in Test Split | Detected Correctly | Detection Rate / Recall | Missed | Notes / Uncertainty Warning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | Benign | 339,256 | 338,616 | **99.81%** | 640 | Sufficient sample |
| `Bot` | Attack | 289 | 172 | **59.52%** | 117 | Sufficient sample |
| `DDoS` | Attack | 19,305 | 19,282 | **99.88%** | 23 | Sufficient sample |
| `DoS GoldenEye` | Attack | 1,581 | 1,572 | **99.43%** | 9 | Sufficient sample |
| `DoS Hulk` | Attack | 27,679 | 27,645 | **99.88%** | 34 | Sufficient sample |
| `DoS Slowhttptest` | Attack | 812 | 808 | **99.51%** | 4 | Sufficient sample |
| `DoS slowloris` | Attack | 807 | 807 | **100.00%** | 0 | Sufficient sample |
| `FTP-Patator` | Attack | 1,416 | 1,416 | **100.00%** | 0 | Sufficient sample |
| `Heartbleed` | Attack | 1 | 0 | **0.00%** | 1 | Warning: Based on 1 example(s); high estimation variance |
| `Infiltration` | Attack | 3 | 0 | **0.00%** | 3 | Sufficient sample |
| `PortScan` | Attack | 23,897 | 23,893 | **99.98%** | 4 | Sufficient sample |
| `SSH-Patator` | Attack | 773 | 771 | **99.74%** | 2 | Sufficient sample |
| `Web Attack - Brute Force` | Attack | 230 | 223 | **96.96%** | 7 | Sufficient sample |
| `Web Attack - Sql Injection` | Attack | 2 | 0 | **0.00%** | 2 | Warning: Based on 2 example(s); high estimation variance |
| `Web Attack - XSS` | Attack | 93 | 92 | **98.92%** | 1 | Sufficient sample |

---

## 5. Methodological Boundaries & Limitations

1. **Row-Weighted Evaluation:** Repeated identical network flow vectors within the test partition remain uncollapsed and weighted equally per flow row. Repeated patterns exert proportionally higher influence on overall metrics.
2. **Unresolved Binary Conflicts:** Groups mapping to conflicting labels in the original dataset remain in their assigned folds without ad-hoc filtering.
3. **Small-Sample High Variance:** Categories with very few instances (e.g. `Heartbleed` with 1 flow, `Web Attack - Sql Injection` with 2 flows) exhibit high statistical uncertainty. Detection results for these classes must be reported with explicit variance warnings.
4. **Offline Benchmark Scope:** These results measure offline discrimination on the CICIDS2017 flow collection. They do not constitute a demonstration of effectiveness on unconstrained live-network traffic or under temporal distribution drift.
