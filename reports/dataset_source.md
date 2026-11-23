# Dataset Provenance & Acquisition Record: CICIDS2017

## 1. Official Publisher & Primary Source Information
- **Original Publisher:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB), Fredericton, Canada.
- **Official Documentation Portal:** [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Primary Gated Intake Portal:** [http://cicresearch.ca/CICDataset/CIC-IDS-2017/](http://cicresearch.ca/CICDataset/CIC-IDS-2017/) (redirects to `https://cicresearch.ca/CICDataset/CIC-IDS-2017/`, which requires an academic registration intake form).
- **Mandatory Publisher Citation:**
  > Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, *"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"*, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

---

## 2. Acquired Third-Party Mirror & Provenance Status
- **Acquisition Source:** Hugging Face Open Dataset Mirror [`bencorn/CICIDS2017`](https://huggingface.co/datasets/bencorn/CICIDS2017)
- **Direct Download URL Used:** `https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/MachineLearningCSV.zip`
- **Third-Party Status & Equivalence Disclaimer:**  
  This repository is an unofficial third-party community mirror. While the mirror description cites the University of New Brunswick as the original publisher and preserves the expected 8-file structure, **bit-for-bit equivalence to the original publisher's distribution has not been independently established and remains unverified**.
- **Evidence vs. Assumptions:**
  - *Observed Evidence:* A valid ZIP archive containing 8 CSV files with filenames matching UNB's documented capture days, valid internal CRC-32 checksums, and expected file sizes.
  - *Unverified Assumption:* That no undocumented filtering, column reordering, or sampling occurred in the mirror relative to UNB's primary files. The upcoming audit phase directly inspects the data to establish its exact raw properties.

---

## 3. Acquired Archive & Local Storage Metrics
- **Archive Filename:** `MachineLearningCSV.zip`
- **Location on Disk:** `data/raw/MachineLearningCSV.zip`
- **Measured Compressed Size:** `235,102,953` bytes (`224.21 MB`)
- **Archive SHA-256 Checksum:** `c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928`
- **Total Uncompressed Size:** `884,645,759` bytes (`843.66 MB`) across 8 CSV files
- **ZIP Integrity Test:** Passed internal CRC-32 testing across all archive members with zero errors.

---

## 4. Extracted CSV Collection (`data/raw/cicids2017/`)

All 8 CSV files were extracted with path-traversal safeguards:

| CSV Filename | Size (Bytes) | Size (MB) | Local SHA-256 Checksum | Documented Day / Threat |
| :--- | :--- | :--- | :--- | :--- |
| `Monday-WorkingHours.pcap_ISCX.csv` | 176,927,918 | 168.73 MB | `852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34` | Monday (BENIGN baseline) |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 135,078,995 | 128.82 MB | `52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851` | Tuesday (FTP/SSH Patator) |
| `Wednesday-workingHours.pcap_ISCX.csv` | 225,166,395 | 214.74 MB | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` | Wednesday (DoS, Heartbleed) |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 52,023,263 | 49.61 MB | `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d` | Thursday AM (Web Attacks) |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 83,102,436 | 79.25 MB | `6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5` | Thursday PM (Infiltration) |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 58,316,725 | 55.62 MB | `53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000` | Friday AM (Botnet) |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 76,906,168 | 73.34 MB | `ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b` | Friday PM (PortScan) |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77,123,859 | 73.55 MB | `6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee` | Friday PM (DDoS) |

---

## 5. Cryptographic Integrity Note
- Locally calculated SHA-256 hashes identify our downloaded files and ensure reproducible, tamper-evident processing across local pipeline stages.
- These hashes **do not prove publisher authenticity** because the publisher (UNB) does not host authoritative SHA-256 reference signatures on its public portal.
- Full manifest metadata is recorded in [`reports/dataset_manifest.json`](reports/dataset_manifest.json).
