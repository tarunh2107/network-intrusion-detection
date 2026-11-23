# Phase 5: Deterministic Data Preparation & Splitting Report

**Execution Timestamp:** 2026-10-12T18:31:33.948561+00:00  
**Total Elapsed Time:** 1336.01 seconds  
**Model Features Retained:** `77`  
**Metadata Fields Attached:** `6`  

---

## 1. Overall Row Accounting & Splitting Proportions

- **Total Input Raw Rows:** `2,830,743`
- **Valid Processed Rows Emitted:** `2,830,743`
- **Explicitly Rejected Rows:** `0`
- **Row Balance Reconciled (`Input == Emitted + Rejected`):** `True`

| Partition | Row Count | Target Ratio | Actual Proportion | Binary Benign (0) | Binary Attack (1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `1,997,204` | 70.0% | `70.55%` | `1,593,763` | `403,441` |
| **Validation** | `417,395` | 15.0% | `14.75%` | `340,078` | `77,317` |
| **Test** | `416,144` | 15.0% | `14.70%` | `339,256` | `76,888` |
| **Total** | `2,830,743` | 100.0% | 100.00% | `2,273,097` | `557,646` |

---

## 2. Partition Fingerprint Disjointness Verification

- **Cross-Partition Collisions (Train <-> Validation):** `0`
- **Cross-Partition Collisions (Train <-> Test):** `0`
- **Cross-Partition Collisions (Validation <-> Test):** `0`
- **Strict Disjointness Established:** `True`

> [!NOTE]
> All rows sharing identical 77-element cleaned numeric feature vectors (fingerprints) were deterministically routed into the identical partition. No identical feature vector appears across train, validation, or test partitions.

---

## 3. Original Attack-Label Coverage Across Partitions

| Display Label | Total Flows | Train Count | Validation Count | Test Count | Present in All Folds? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | 2,273,097 | 1,593,763 | 340,078 | 339,256 | **Yes** |
| `Bot` | 1,966 | 1,384 | 293 | 289 | **Yes** |
| `DDoS` | 128,027 | 89,544 | 19,178 | 19,305 | **Yes** |
| `DoS GoldenEye` | 10,293 | 7,120 | 1,592 | 1,581 | **Yes** |
| `DoS Hulk` | 231,073 | 175,183 | 28,211 | 27,679 | **Yes** |
| `DoS Slowhttptest` | 5,499 | 3,853 | 834 | 812 | **Yes** |
| `DoS slowloris` | 5,796 | 4,097 | 892 | 807 | **Yes** |
| `FTP-Patator` | 7,938 | 5,368 | 1,154 | 1,416 | **Yes** |
| `Heartbleed` | 11 | 9 | 1 | 1 | **Yes** |
| `Infiltration` | 36 | 27 | 6 | 3 | **Yes** |
| `PortScan` | 158,930 | 111,174 | 23,859 | 23,897 | **Yes** |
| `SSH-Patator` | 5,897 | 4,157 | 967 | 773 | **Yes** |
| `Web Attack - Brute Force` | 1,507 | 1,040 | 237 | 230 | **Yes** |
| `Web Attack - Sql Injection` | 21 | 16 | 3 | 2 | **Yes** |
| `Web Attack - XSS` | 652 | 469 | 90 | 93 | **Yes** |

---

## 4. Cleaning Anomalies & Sentinel Handling

- **String NaNs (`NaN`, `null`, `none`, empty) converted to missing:** `0`
- **Infinities (`+inf`, `-inf`, string infinity) converted to missing:** `4,376` (Reconciled from an initial 8,752 double-count artifact; exactly 1,509 in `Flow Bytes/s` + 2,867 in `Flow Packets/s`, matching audit baseline)
- **Negative `Flow Bytes/s` values converted to missing:** `85`
- **Negative `Flow Packets/s` values converted to missing:** `115`
- **Invalid non-numeric float tokens converted to missing:** `1,358`
- **Sentinel preservation:** Non-rate negative values (e.g. `Init_Win_bytes_forward = -1`) were preserved intact without conversion.
- **Constant column status:** The 8 zero-variance features identified in Phase 4 were preserved in the processed shards to allow feature selection (`VarianceThreshold`) to be fitted strictly on training data in Phase 6.

---

## 5. Storage Metrics & Compressed Output Shards

- **Total Shards Generated:** `41` files
- **Compressed Disk Usage:** `382.80 MB` (`401,391,862` bytes)
- **Output Location:** `data\processed\v1`

---

## 6. Remaining Methodological Considerations for Phase 6

1. **Offline Benchmark Split:** This hash-grouped partition constitutes an offline benchmark split designed to prevent duplicate vector leakage. It does not test temporal drift or unconstrained live-network generalization.
2. **Rare Class Coverage:** Classes with very few independent flow groups (e.g. Heartbleed with 11 flows) may fall entirely or predominantly into one partition under grouped hashing.
3. **Binary Conflict Handling:** Groups mapping to conflicting labels were preserved intact in their assigned partition. Explicit conflict-resolution policies (e.g. dropping ambiguous vectors or label smoothing) remain for evaluation in Phase 6.
4. **Duplicate Weighting:** Repeated identical flows remain within their assigned partition. Training can either weight flows equally, deduplicate within-fold, or apply sample weights during model optimization.