# Phase 6A: Baseline Model Training & Comparison Report

**Timestamp (UTC):** `2026-10-12T18:52:05.485286+00:00`  
**Total Execution Time:** `116.12 seconds`  
**Random Seed:** `42`  
**Winning Model:** `Random Forest (100 trees, depth=18, min_leaf=2, jobs=2)`  

---

## 1. Executive Summary

In Phase 6A, four resource-bounded baseline models were trained and evaluated on strictly disjoint, leak-free CICIDS2017 flow data under an offline binary classification contract (`BENIGN` = 0 vs `ATTACK` = 1).

- **Training Sample:** 150,000 flows sampled deterministically across all 25 train shards.
- **Validation Sample:** 50,000 flows sampled deterministically across all 8 validation shards.
- **Test Partition Protection:** The test partition (`data/processed/v1/test`) was **strictly untouched and held out** for final testing.
- **Selected Champion:** **Random Forest (100 trees, depth=18, min_leaf=2, jobs=2)** achieved the highest validation ATTACK F1-score (**0.9953**) with high precision (**0.9954**), strong recall (**0.9951**), and a low False Positive Rate (**0.1029%**).

---

## 2. Baseline Model Comparison Table

| Model | Attack Precision | Attack Recall | Attack F1-score | Accuracy | FPR | FNR | Average Precision (AP) | Fit Time (s) | Batch Scoring (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Most Frequent)** | 0.0000 | 0.0000 | **0.0000** | 0.8162 | 0.0000% | 100.0000% | 0.1838 | 2.47s | 1.26 |
| **Logistic Regression (Balanced)** | 0.6815 | 0.9487 | **0.7932** | 0.9091 | 9.9851% | 5.1257% | 0.9101 | 11.92s | 1.80 |
| **Decision Tree (max_depth=15, min_leaf=5)** | 0.9948 | 0.9902 | **0.9925** | 0.9972 | 0.1176% | 0.9794% | 0.9951 | 9.47s | 1.59 |
| **Random Forest (100 trees, depth=18, min_leaf=2, jobs=2)** | 0.9954 | 0.9951 | **0.9953** | 0.9983 | 0.1029% | 0.4897% | 0.9995 | 38.99s | 7.49 |

> [!NOTE]
> - **Average Precision (AP):** Calculated using Scikit-learn's `average_precision_score` (the weighted mean of precisions at each threshold). This is distinct from trapezoidal area under the curve (`sklearn.metrics.auc`).
> - **Batch Scoring Throughput:** The reported timing (e.g., 7.49 ms / 1,000 samples) reflects vectorized in-memory model execution. It is **not** a measurement of end-to-end single-request REST API latency.

---

## 3. Confusion Matrix Breakdown

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Most Frequent)** | 40,811 | 0 | 9,189 | 0 | 50,000 |
| **Logistic Regression (Balanced)** | 36,736 | 4,075 | 471 | 8,718 | 50,000 |
| **Decision Tree (max_depth=15, min_leaf=5)** | 40,763 | 48 | 90 | 9,099 | 50,000 |
| **Random Forest (100 trees, depth=18, min_leaf=2, jobs=2)** | 40,769 | 42 | 45 | 9,144 | 50,000 |

---

## 4. Per-Category Detection Breakdown (Random Forest Baseline)

> [!IMPORTANT]
> This binary model outputs predictions strictly for `BENIGN` (0) vs `ATTACK` (1) and possesses **no mechanism to identify attack categories at inference**. The category detection breakdown below is computed strictly by grouping validation flows by their ground-truth metadata annotations (`display_label`).

| Category | Class Type | Total in Val Sample | Detected Correctly | Detection Rate / Recall | Missed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | Benign | 40,811 | 40,769 | **99.90%** | 42 |
| `Bot` | Attack | 35 | 13 | **37.14%** | 22 |
| `DDoS` | Attack | 2,311 | 2,311 | **100.00%** | 0 |
| `DoS GoldenEye` | Attack | 172 | 168 | **97.67%** | 4 |
| `DoS Hulk` | Attack | 3,339 | 3,327 | **99.64%** | 12 |
| `DoS Slowhttptest` | Attack | 102 | 99 | **97.06%** | 3 |
| `DoS slowloris` | Attack | 101 | 101 | **100.00%** | 0 |
| `FTP-Patator` | Attack | 131 | 131 | **100.00%** | 0 |
| `PortScan` | Attack | 2,844 | 2,844 | **100.00%** | 0 |
| `SSH-Patator` | Attack | 109 | 108 | **99.08%** | 1 |
| `Web Attack - Brute Force` | Attack | 36 | 33 | **91.67%** | 3 |
| `Web Attack - Sql Injection` | Attack | 2 | 2 | **100.00%** | 0 |
| `Web Attack - XSS` | Attack | 7 | 7 | **100.00%** | 0 |

---

## 5. Resource Utilization & Operational Profile

- **Peak Working Memory:** Maintained well within the 7 GB system budget by reading individual shards sequentially in streaming chunks and sampling rows uniformly.
- **Model Serialization:** Winning pipeline persisted to `models/baseline_v1/model.joblib` using joblib (SHA-256: `4339282ba6ce68b5c44c6dd6a72e7796e755a77542d65021984dd1a8b0434793`).
- **Batch Execution:** Vectorized in-memory inference achieved 7.49 ms per 1,000 samples. Actual single-request HTTP latency will be measured after FastAPI service integration in Phase 7.
- **Preprocessing Integration:** Imputation (`median`) and variance filtering (`threshold=0.0`) are bundled inside the Scikit-learn Pipeline, preventing runtime feature drift.

---

## 6. Observations & Considerations for Phase 6B

1. **Bot Class Detection:** Lower recall on `Bot` (37.14%) is an observed empirical weakness in this validation sample; its root cause has not been demonstrated or proven.
2. **Decision Threshold Calibration:** Evaluate thresholds to examine whether validation recall can be improved while enforcing a strict false-positive budget.
3. **Held-Out Test Set:** All test shards remain completely untouched and will be evaluated once under a frozen threshold.
