# Model Card: NIDS Baseline Random Forest Classifier (v1)

## 1. Model Details & Architecture
- **Model Identifier:** `baseline_v1`
- **Model Type:** Supervised Binary Classification Pipeline (`scikit-learn` v1.9.1)
- **Pipeline Structure:**
  1. `SimpleImputer(strategy='median')`: Imputes missing feature values using medians computed strictly on the training sample.
  2. `RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=2, n_jobs=2, random_state=42)`
- **Class Mapping:**
  - `0`: `BENIGN`
  - `1`: `ATTACK`
- **Operating Decision Threshold:** `0.29` (selected via validation ROC/PR operating rule; default is `0.50`)
- **Model Weight SHA-256 Checksum:** `4339282ba6ce68b5c44c6dd6a72e7796e755a77542d65021984dd1a8b0434793`
- **Weights File:** `models/baseline_v1/model.joblib` (6.75 MB)
- **Metadata Files:** `models/baseline_v1/metadata.json`, `models/baseline_v1/frozen_evaluation_config.json`

---

## 2. Intended Use & Scope
- **Intended Purpose:** Offline binary classification of pre-extracted network flow records (NetFlow / IPFIX style bidirectional summary metrics) for security log analysis, traffic pattern auditing, and academic benchmark evaluation.
- **Input Specification:** Exactly 77 ordered numeric features representing statistical flow properties (durations, packet counts, inter-arrival times, header lengths, flag counts, TCP window sizes, subflow stats, active/idle timers).
- **Primary Users:** Security analysts, network researchers, and security engineering students conducting post-incident or forensic flow log reviews.

---

## 3. Excluded & Out-of-Scope Uses
> [!CAUTION]
> **Explicitly Excluded Operational Scenarios:**
> - **Not an Intrusion Prevention System (IPS):** The system has no capability to terminate sockets, block IP addresses, alter firewall tables, or modify network routing. It operates strictly in passive read-only inference.
> - **Not a Live Packet Capture Engine:** The system does not interface with network interfaces, promiscuous mode packet sniffers (WinPcap/Npcap/libpcap), raw sockets, or real-time stream reassembly. It processes aggregated bidirectional flow records after capture.
> - **Sub-Millisecond Line-Rate Processing:** Inference latency (~7.5 ms per 1,000 flows in batch mode) is designed for batched log evaluation, not inline hardware-switched packet forwarding.
> - **Adversarial Robustness Guarantee:** The model has not been certified against adversarial perturbations, evasion attacks (e.g., deliberate packet-splitting or feature manipulation), or domain drift.

---

## 4. Dataset Provenance & Third-Party Mirror Limitation
- **Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick).
- **Capture Window:** 5 consecutive days (Monday July 3, 2017 to Friday July 7, 2017).
- **Original Source:** [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (requires academic registration intake).
- **Acquired Mirror:** Hugging Face repository [`bencorn/CICIDS2017`](https://huggingface.co/datasets/bencorn/CICIDS2017) (`MachineLearningCSV.zip`).
- **Provenance Limitation:** UNB does not provide cryptographic SHA-256 signatures for the CSV distributions. Although our local acquisition validated archive CRC-32 integrity and filename matching, **bit-for-bit equivalence between this third-party mirror and UNB's original generation cannot be independently verified**.

---

## 5. Feature Engineering, Cleaning & Preprocessing Contract
- **Input Dimension:** 77 features (derived from 79 raw CSV columns, which include 78 candidate features + 1 label column).
- **Redundant Column Removal:**
  - In raw CSV files, two columns carry the header `Fwd Header Length`:
    - Primary column: **0-based index 34** (1-based column 35).
    - Secondary column: **0-based index 55** (1-based column 56).
  - Our dataset audit proved that 0-based index 34 and 0-based index 55 are value-identical across all 2,830,743 raw rows (0 mismatches).
  - Secondary `Fwd Header Length` at **0-based index 55 (1-based column 56)** was removed during preprocessing.
  - *(Note: 0-based index 60 / 1-based column 61 is `Bwd Avg Packets/Bulk` and is preserved in the 77-feature schema).*
- **Missing & Infinite Value Handling:**
  - 4,376 infinite values (`Flow Bytes/s`: 1,509; `Flow Packets/s`: 2,867) converted to missing.
  - 85 negative `Flow Bytes/s` and 115 negative `Flow Packets/s` converted to missing (rate metrics cannot physically be negative).
  - Negative values in TCP window features (`Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `min_seg_size_forward`) are valid OS sentinels (-1 indicates window not advertised) and were **preserved**.
- **Imputation:** Training-set medians stored inside the model's `SimpleImputer`. During inference, JSON `null` values are automatically imputed with these frozen training statistics. Vectors with all 77 features null are strictly rejected.

---

## 6. Leakage Prevention & Data Splitting

### Feature Fingerprinting vs. Partition Assignment Hashing
The data preparation pipeline (`app/ml/preprocess.py`) strictly separates the **feature fingerprint** from the **partition assignment hash**:

1. **Feature Fingerprint Algorithm:**
   - Float Canonicalization: The 77-element `float64` vector is converted into IEEE 754 uint64 bit representations. All `NaN` representations are normalized to canonical quiet NaN (`0x7FF8000000000000`), and `-0.0` and `+0.0` are normalized to canonical zero (`0x0000000000000000`).
   - Hash Function: **SHA-256** is computed over the 616 bytes (77 floats $\times$ 8 bytes) of canonical uint64 data:
     $$\text{fingerprint} = \text{hashlib.sha256}(\text{canonical\_bytes}).\text{hexdigest}()$$
   - Emits a deterministic 64-character lowercase hex string representing the exact feature vector.

2. **Partition Assignment Hash:**
   - Salted Hash: To assign a fingerprint group to `train`, `validation`, or `test`, a salted SHA-256 hash is computed:
     $$\text{digest} = \text{hashlib.sha256}(b"\text{phase5\_split\_v1:seed=42:}" + \text{fingerprint.encode}("ascii")).\text{digest}()$$
   - Bucket Rule: The first 8 bytes of the digest are parsed as an unsigned 64-bit big-endian integer modulo 10,000:
     $$\text{bucket} = \text{int.from\_bytes}(\text{digest}[:8], \text{"big"}) \pmod{10000}$$
   - Partition Boundaries:
     - `bucket < 7000`: `train` (70.0% target)
     - `7000 <= bucket < 8500`: `validation` (15.0% target)
     - `bucket >= 8500`: `test` (15.0% target)
   - Guarantee: All rows sharing the same feature fingerprint are mathematically guaranteed to map to the exact same partition.

### Partition Distribution & Disjointness Audit
- **Total Dataset Size:** 2,830,743 rows.
- **Partition Counts:**
  - **Train Partition (70.55%):** 1,997,204 rows (1,593,763 Benign, 403,441 Attack).
  - **Validation Partition (14.75%):** 417,395 rows (340,078 Benign, 77,317 Attack).
  - **Held-Out Test Partition (14.70%):** 416,144 rows (339,256 Benign, 76,888 Attack).
- **Disjointness Audit:** Exactly **0 cross-partition collisions** verified across Train/Val, Train/Test, and Val/Test.
- **Resource-Constrained Model Sampling:**
  - **Training Subsample:** 150,000 stratified rows (119,776 Benign, 30,224 Attack).
  - **Validation Comparison Subsample:** 50,000 stratified rows (40,811 Benign, 9,189 Attack).
  - **Threshold Selection:** Evaluated across the **entire 417,395 rows** of the validation partition.
  - **Final Test Evaluation:** Evaluated across the **entire 416,144 rows** of the held-out test partition.

---

## 7. Model Selection & Comparison (Validation Subsample)

On the 50,000-row validation comparison sample, 4 candidate architectures were evaluated:

| Model Candidate | Validation ATTACK Precision | Validation ATTACK Recall | Validation ATTACK F1 | False Positive Rate (FPR) | Fit Time (s) | Inference Latency (ms / 1k flows) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Most Frequent)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.47s | 1.26 ms |
| **Logistic Regression (Balanced)** | 0.6815 | 0.9487 | 0.7932 | 0.0999 (9.99%) | 11.92s | 1.80 ms |
| **Decision Tree (max_depth=15)** | 0.9948 | 0.9902 | 0.9925 | 0.0012 (0.12%) | 9.47s | 1.59 ms |
| **Random Forest (100 trees, depth=18)** | **0.9954** | **0.9951** | **0.9953** | **0.0010 (0.10%)** | 38.99s | 7.49 ms |

**Winner Selection:** Random Forest was selected based on superior ATTACK F1-score and substantially lower False Positive Rate.

---

## 8. Threshold Selection Rationale
- **Operating Rule:** Choose the decision threshold that maximizes validation ATTACK recall while strictly bounding the False Positive Rate (FPR) at $\le 0.20\%$ ($\le 0.0020$).
- **Selection Data:** Full validation partition (417,395 rows).
- **Selected Threshold:** `0.29` (FPR = $0.1958\%$, Recall = $99.71\%$, Precision = $99.14\%$).
- **Comparison to Default (0.50):** The default 0.50 threshold produced FPR = $0.0862\%$ but lower ATTACK recall ($99.46\%$, missing 414 attacks vs. 224 at 0.29).

---

## 9. Held-Out Test Evaluation Performance

Evaluated **once** on the untouched held-out test partition (**416,144 rows**):

### Overall Binary Metrics
| Metric | Frozen Threshold (0.29) | Default Threshold (0.50) | Notes on Calculation |
| :--- | :--- | :--- | :--- |
| **Precision (ATTACK)** | **99.17%** (`0.991723`) | **99.61%** (`0.996060`) | $TP / (TP + FP)$ |
| **Recall (ATTACK)** | **99.73%** (`0.997308`) | **99.29%** (`0.992860`) | $TP / (TP + FN)$ |
| **F1-Score (ATTACK)** | **99.45%** (`0.994507`) | **99.45%** (`0.994457`) | Harmonic mean of precision & recall |
| **Accuracy** | **99.80%** (`0.997965`) | **99.80%** (`0.997955`) | $(TP + TN) / \text{Total}$ |
| **False Positive Rate (FPR)** | **0.1886%** (`0.001886`) | **0.0890%** (`0.000890`) | $FP / (FP + TN)$ |
| **False Negative Rate (FNR)** | **0.2692%** (`0.002692`) | **0.7140%** (`0.007140`) | $FN / (FN + TP)$ |
| **Average Precision (AP)** | **0.9996** (`0.999642`) | **0.9996** (`0.999642`) | Calculated via Scikit-learn `average_precision_score` (not trapezoidal PR-AUC) |
| **True Negatives (TN)** | 338,616 | 338,954 | Correctly classified BENIGN |
| **False Positives (FP)** | 640 | 302 | BENIGN misclassified as ATTACK |
| **False Negatives (FN)** | 207 | 549 | ATTACK misclassified as BENIGN |
| **True Positives (TP)** | 76,681 | 76,339 | Correctly classified ATTACK |

> [!NOTE]
> **Average Precision vs. Trapezoidal PR-AUC:** The implementation evaluates ranking quality using Scikit-learn's `average_precision_score`, which calculates the weighted mean of precisions achieved at each threshold ($AP = \sum_n (R_n - R_{n-1}) P_n$). This is mathematically distinct from trapezoidal PR-AUC ($\int P(R) dR$ via the trapezoidal rule).

### Per-Attack Category Recall Breakdown (Post-Hoc Evaluation Metadata)
*Note: The model is strictly binary at inference. Category recall is computed post-hoc using ground-truth test metadata.*

| Threat Category | Held-Out Test Total Rows | Detected at Threshold 0.29 | Recall (0.29) | Detected at Default 0.50 | Recall (0.50) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BENIGN (Specificity)** | 339,256 | 338,616 | **99.81%** | 338,954 | **99.91%** |
| **PortScan** | 23,897 | 23,893 | **99.98%** | 23,886 | 99.95% |
| **DDoS** | 19,305 | 19,282 | **99.88%** | 19,270 | 99.82% |
| **DoS Hulk** | 27,679 | 27,645 | **99.88%** | 27,543 | 99.51% |
| **SSH-Patator** | 773 | 771 | **99.74%** | 763 | 98.71% |
| **DoS Slowhttptest** | 812 | 808 | **99.51%** | 801 | 98.65% |
| **DoS GoldenEye** | 1,581 | 1,572 | **99.43%** | 1,548 | 97.91% |
| **FTP-Patator** | 1,416 | 1,416 | **100.00%** | 1,320 | 93.22% |
| **DoS slowloris** | 807 | 807 | **100.00%** | 779 | 96.53% |
| **Web Attack - XSS** | 93 | 92 | **98.92%** | 89 | 95.70% |
| **Web Attack - Brute Force** | 230 | 223 | **96.96%** | 206 | 89.57% |
| **Bot** | 289 | 172 | **59.52%** | 134 | **46.37%** |
| **Infiltration** | 3 | 0 | **0.00%** | 0 | 0.00% |
| **Web Attack - Sql Injection** | 2 | 0 | **0.00%** | 0 | 0.00% |
| **Heartbleed** | 1 | 0 | **0.00%** | 0 | 0.00% |

---

## 10. Key Limitations & Risk Analysis

### A. Bot Class Detection Weakness
The model detected only **59.52% of Bot flows** (172 / 289) on the held-out test partition (improved from 46.37% at default threshold). 
- **Cause Analysis:** In CIC-IDS2017, Bot traffic (Ares botnet) consists of stealthy C2 keep-alives and periodic polling whose packet rates, byte counts, and durations heavily overlap with benign workstation HTTP traffic. The empirical weakness is proven by test metrics, but specific causal feature attribution remains an open hypothesis.

### B. Rare-Class Statistical Uncertainty
Certain attacks in CIC-IDS2017 contain minimal occurrences in the test set:
- `Heartbleed`: 1 flow
- `Web Attack - Sql Injection`: 2 flows
- `Infiltration`: 3 flows
Due to extreme sample sparsity ($N \le 3$), zero detection on these categories does not prove general inability, but confirms that empirical recall has very high variance and cannot be considered statistically reliable.

### C. Repeated-Row Weighting
Benchmark evaluation metrics are row-weighted. In CIC-IDS2017, voluminous attacks (e.g., DoS Hulk with 27,679 rows, PortScan with 23,897 rows) dominate aggregate precision and recall. A model could achieve 99% overall accuracy while failing on less frequent, highly targeted attacks.

### D. Unresolved Label Conflicts in Benchmark
The dataset audit revealed 1,358 duplicate feature vectors that have conflicting label assignments in the source CSVs. Because identical vectors cannot be uniquely resolved by feature statistics alone without ground-truth PCAP re-inspection, residual label noise is inherent to the benchmark.

### E. Uncalibrated Output Scores
Random Forest `predict_proba` values reflect tree ensemble voting proportions, not mathematically calibrated Bayesian posterior probabilities. Output probabilities should be interpreted as ordinal ranking scores relative to the `0.29` threshold rather than exact real-world attack likelihoods.

### F. Model Loading & Deserialization Security
Python `joblib` artifacts execute arbitrary bytecode during deserialization. The application implements a cryptographic trust boundary:
- Rejects any model artifact whose streaming SHA-256 does not match `4339282ba6ce68b5c44c6dd6a72e7796e755a77542d65021984dd1a8b0434793`.
- Model paths cannot be supplied by HTTP clients or dashboard uploads.
