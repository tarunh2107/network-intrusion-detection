# CICIDS2017 Dataset Audit & Schema Inspection Report

**Audit Execution Time:** 2026-10-12T17:39:44.296051+00:00  
**Elapsed Scan Time:** 472.91 seconds  
**Total CSV Files Scanned:** 8  
**Chunk Size:** 25,000 rows  

---

## 1. Overall Row & Class Summary

- **Total Rows Scanned:** `2,830,743`
- **Benign Rows:** `2,273,097` (80.30%)
- **Attack Rows:** `557,646` (19.70%)
- **Empty / Unlabeled Rows:** `0`

### Global Label Distribution (Whitespace-Trimmed)
| Class Label | Row Count | Percentage |
| :--- | :--- | :--- |
| `BENIGN` | 2,273,097 | 80.3004% |
| `DoS Hulk` | 231,073 | 8.1630% |
| `PortScan` | 158,930 | 5.6144% |
| `DDoS` | 128,027 | 4.5227% |
| `DoS GoldenEye` | 10,293 | 0.3636% |
| `FTP-Patator` | 7,938 | 0.2804% |
| `SSH-Patator` | 5,897 | 0.2083% |
| `DoS slowloris` | 5,796 | 0.2048% |
| `DoS Slowhttptest` | 5,499 | 0.1943% |
| `Bot` | 1,966 | 0.0695% |
| `Web Attack - Brute Force` (raw: `'Web Attack � Brute Force'`) | 1,507 | 0.0532% |
| `Web Attack - XSS` (raw: `'Web Attack � XSS'`) | 652 | 0.0230% |
| `Infiltration` | 36 | 0.0013% |
| `Web Attack - Sql Injection` (raw: `'Web Attack � Sql Injection'`) | 21 | 0.0007% |
| `Heartbleed` | 11 | 0.0004% |

---

## 2. Duplicate Records & Split Leakage Analysis

- **Unique Feature Vectors:** `2,521,664`
- **Excess Duplicate Feature Vectors:** `309,079` (10.92%)
- **Excess Duplicate Feature+Label Records:** `308,381`
- **Conflicting Label Feature Vectors (Multiclass):** `698` (spanning `7,020` rows)
- **Cross-File Duplicate Vectors:** `34,208` (spanning `146,363` rows)

> [!NOTE]
> Over 300,000 duplicate rows exist in the raw dataset. Duplicate feature vectors across train and test partitions present an evaluation leakage risk. Leakage can be mitigated either by deduplicating identical feature vectors prior to splitting or by using group-aware partitioning (e.g. by time/source).
> Note: Conflicting label feature vectors reflect disagreements among multiclass labels; the breakdown between binary (BENIGN vs. ATTACK) and multiclass-only conflicts is not disaggregated in the summary.

### Within-File Duplicate Excess Rows
| CSV Filename | Total Rows | Duplicate Excess Rows |
| :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 2,633 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | 72,353 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | 6,888 |
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 26,935 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | 35,630 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 6,066 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 24,065 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 81,957 |

---

## 3. Schema & Duplicate Column Verification

- **Suspected Duplicate Columns:** `Fwd Header Length (pos 34)` and `Fwd Header Length (pos 55)`
- **Strict Equality Across All 2,830,743 Rows:** `True` (Mismatches: `0`)
- **Duplicate Column Rule:** Column 55 (`Fwd Header Length.1`) is strictly identical to Column 34 across all rows and should be dropped during preprocessing.
- **Constant Zero-Variance Columns Identified (8):** `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`
> [!NOTE]
> Zero-variance feature selection should be fitted on the training split within the evaluation pipeline rather than applied via global dataset truncation.

---

## 4. Per-File Statistics
| Filename | Rows | Benign | Attack | Missing Labels |
| :--- | :--- | :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 97,718 | 128,027 | 0 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | 127,537 | 158,930 | 0 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | 189,067 | 1,966 | 0 |
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 529,918 | 0 | 0 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | 288,566 | 36 | 0 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 168,186 | 2,180 | 0 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 432,074 | 13,835 | 0 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 440,031 | 252,672 | 0 |

---

## 5. Non-Numeric & Infinity Values in Features
| Feature Name (Pos) | Missing / Blank | NaN Strings | Invalid Non-Numeric | Positive Inf | Negative Inf | Finite Min | Finite Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Flow Bytes/s` (14) | 0 | 1,358 | 0 | 1,509 | 0 | -2.61e+08 | 2.071e+09 |
| `Flow Packets/s` (15) | 0 | 0 | 0 | 2,867 | 0 | -2e+06 | 4e+06 |