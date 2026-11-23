# Project Status & Verification Checklist

This document tracks the verified implementation boundaries of the Network Intrusion Detection System (NIDS) as of **Phase 10**. In adherence to strict evidence-based engineering standards, items are explicitly partitioned by verification status.

---

## 1. Implemented & Verified

### Data Pipeline & Preprocessing
- [x] **Acquisition & Integrity Verification:** Downloaded and extracted 8 CSVs from the benchmark mirror; verified ZIP CRC-32 integrity; calculated and recorded SHA-256 digests in [`reports/dataset_manifest.json`](../reports/dataset_manifest.json).
- [x] **Comprehensive Schema & Feature Audit:** Inspected all 2,830,743 raw rows in bounded 25k chunks; identified and removed redundant second `Fwd Header Length` at **0-based index 55 (1-based column 56)** after proving value-identity to **0-based index 34 (1-based column 35)**; quantified 4,376 infinite values and 200 negative rate values; generated audit reports.
- [x] **Leakage-Aware Disjoint Partitioning:** Implemented canonical IEEE 754 uint64 float vector **SHA-256 fingerprinting** (256-bit) and **salted SHA-256 partition hashing** (64-bit integer modulo 10,000); grouped identical flows to enforce strict partition disjointness; verified **exactly 0 cross-partition collisions** across Train (1,997,204), Validation (417,395), and Held-Out Test (416,144).
- [x] **Training-Only Preprocessing Encapsulation:** Encapsulated median imputation inside a Scikit-learn Pipeline; fitted strictly on the 150,000-row training sample.

### Machine Learning & Evaluation
- [x] **Baseline Model Comparison:** Evaluated 4 candidate models (Dummy, Logistic Regression, Decision Tree, Random Forest) on a 50,000-row validation sample; verified Random Forest superiority.
- [x] **Validation Threshold Optimization:** Evaluated operating rule ($\text{FPR} \le 0.20\%$) across all 417,395 validation rows; selected and froze operating threshold `0.29` into [`models/baseline_v1/frozen_evaluation_config.json`](../models/baseline_v1/frozen_evaluation_config.json).
- [x] **One-Time Frozen Held-Out Test Evaluation:** Scored the entire 416,144-row test set once; achieved 99.17% Precision, 99.73% Recall, 99.45% F1, 0.1886% FPR, and **0.9996 Average Precision (AP via `average_precision_score`)**; documented Bot class weakness (59.52%) and rare-class sample limitations.

### Backend & Presentation Services
- [x] **FastAPI Inference Service:** Implemented lifespan model loading with streaming SHA-256 pre-verification; exposed `/health`, `/model/info`, `/predict`, `/predict/batch`, `/alerts`, `/statistics`; enforced strict Pydantic input validation (rejecting booleans, string numbers, infinities).
- [x] **Atomic SQLite Persistence:** Implemented transactional prediction logging with rollback; separated presentation alerts query from prediction ingestion.
- [x] **Decoupled Streamlit Dashboard (Service Liveness & Logic):** Built standalone UI interacting strictly through HTTP REST calls; implemented client-side CSV validation (1,000-row cap, 10 MB limit, schema checks) rejecting invalid files locally before network transmission; displayed color-coded alert tables and summary metrics.
- [x] **1,000-Flow Demonstration & Chunked Ingestion:** Generated [`data/demo/example_flows_1000.csv`](../data/demo/example_flows_1000.csv) from 1,000 distinct source rows in the training partition (0 overlap with held-out test data); added dashboard download button; implemented sequential batching ($\le 500$ flows per `POST /predict/batch` request) preserving 1:1 input order; verified that scoring 1,000 flows adds exactly 1,000 records to an isolated database; verified partial failure stops without automatic POST retries.
- [x] **Automated Test Suite:** **66 of 66 tests passing** across preflight, API, dashboard, temp DB integration, training, preprocessing, evaluation, and audit modules.
- [x] **Runtime Artifact Preflight Verification:** [`scripts/check_runtime_artifacts.py`](../scripts/check_runtime_artifacts.py) validates model SHA-256, frozen configs, 77-feature schema, evaluation report, both demo CSVs (10-flow and 1,000-flow), and database permissions without deserializing untrusted joblib files.
- [x] **Local Two-Terminal Fallback Runtime Verification:** Verified direct startup of Uvicorn and Streamlit bound strictly to `127.0.0.1`; scored 10 flows from `data/demo/example_flows.csv` directly via FastAPI HTTP POST; confirmed 5 alerts logged in SQLite; confirmed full port release upon termination. *(Note: Graphical browser interaction was not part of this programmatic check; see Unverified below).*

---

## 2. Implemented but Unverified

The following components are implemented in the codebase but have not completed live end-to-end execution verification due to environment boundaries:

- [ ] **Docker Engine Container Builds:**
  - *Current Status:* [`Dockerfile.api`](../Dockerfile.api), [`Dockerfile.dashboard`](../Dockerfile.dashboard), and [`docker-compose.yml`](../docker-compose.yml) are fully defined and passed static validation (`docker compose config` exited with code 0).
  - *Unverified Boundary:* Live container building (`docker compose build`), Linux package installation compatibility (`python:3.11-slim`), and container networking have not executed because the Docker Desktop daemon was not running.
- [ ] **Docker Volume Persistence Under Container Recreation:**
  - *Current Status:* Named volume `nids_db_data` is configured.
  - *Unverified Boundary:* Destroying and recreating containers with persistent volume retention has not been tested in a live Docker engine.
- [ ] **Multi-Step Interactive Graphical Browser Session:**
  - *Current Status:* Headless Chrome captured a static snapshot (`dashboard_snapshot.png`) showing the initial skeleton loading state. Service health endpoints (`/_stcore/health`) return HTTP 200.
  - *Unverified Boundary:* Programmatic browser-driven file drag-and-drop, tab clicking, interactive form submissions, and dynamic table rendering have not been verified via headless browser automation (no Playwright/Selenium installed in virtual environment). Live graphical UI interaction remains pending manual desktop browser review.

---

## 3. Blocked (External Environment Dependencies)

- **Docker Desktop Engine Startup:**
  - *Blocker:* Windows Docker Desktop binary exists at `C:\Program Files\Docker\Docker\resources\bin\docker.exe`, but the background daemon (`npipe:////./pipe/dockerDesktopLinuxEngine`) was stopped and did not initialize within 180 seconds.
  - *Safety Boundary:* Modifying Windows virtualization features, WSL distributions, system services, or administrative prompts was prohibited. Starting the Docker daemon requires interactive user launch.
- **Direct PowerShell Supervisor Invocation Under Default Policy:**
  - *Blocker:* Executing `.\scripts\start_local.ps1` directly fails with `PSSecurityException` under Windows default `Restricted` execution policy.
  - *Safety Boundary:* Modifying the machine execution policy is prohibited. The verified two-terminal Python commands bypass this blocker safely without altering execution policies.

---

## 4. Optional Future Work

- [ ] **High-Throughput Streaming Ingestion:** Integrate an asynchronous message queue (e.g., Apache Kafka / RabbitMQ) with high-speed flow collectors (e.g., GoFlow2 or Zeek).
- [ ] **Automated Concept Drift Detection:** Implement streaming distribution drift detectors (e.g., Page-Hinkley test, Kolmogorov-Smirnov tests) on network inter-arrival times and packet size distributions.
- [ ] **Explainable AI (XAI) Attribution:** Integrate TreeSHAP or integrated gradients into the API to return feature importance attribution per alert.
- [ ] **CI/CD Linux Pipeline:** Establish GitHub Actions workflows with Linux runners to verify Docker container builds and multi-platform packaging automatically.
