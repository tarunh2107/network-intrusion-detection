# NIDS End-to-End Demonstration Guide (5-Minute Walkthrough)

This guide provides an evidence-based, 5-minute live demonstration script for academic evaluators, interviewers, and project reviewers. It uses the verified local two-terminal startup without requiring administrative elevation, system execution policy modifications, or Docker engine dependencies.

---

## Pre-Flight Requirements & Port Verification
Before beginning, ensure that no stale services occupy ports `8000` or `8501`:

```powershell
# Verify ports 8000 and 8501 are clear
Get-NetTCPConnection -LocalPort 8000, 8501 -ErrorAction SilentlyContinue
```

Run the read-only preflight verification to validate runtime artifacts without deserializing unverified weights:

```powershell
.\.venv\Scripts\python.exe scripts/check_runtime_artifacts.py
```
*Expected Output:* `[SUCCESS] ALL RUNTIME ARTIFACTS AND CONFIGURATIONS VERIFIED SUCCESSFULLY.`

---

## Minute 1: Start Services via Two Terminals

Open two separate Windows PowerShell terminals in the project root.

> [!NOTE]
> Running the commands directly using `.\.venv\Scripts\python.exe` operates within standard user permissions and avoids Windows default `Restricted` PowerShell script execution policies.

### Terminal 1: Launch FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
- **What to observe:** Uvicorn starts on `http://127.0.0.1:8000`. The lifespan handler computes the model SHA-256 (`4339282ba6ce...`), validates it against `frozen_evaluation_config.json`, resolves the ATTACK class index dynamically via `classes_`, and initializes the SQLite schema.

### Terminal 2: Launch Streamlit Dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard/main.py --server.address 127.0.0.1 --server.port 8501
```
- **What to observe:** Streamlit opens in your browser at `http://127.0.0.1:8501`.

---

## Minute 2: Verify Service Health & Readiness

### 1. Backend Health Check
In a browser or via PowerShell:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```
```json
{
  "status": "healthy",
  "service_ready": true,
  "model_loaded": true,
  "database_connected": true
}
```

### 2. Model Contract & Feature Schema
Inspect the verified contract:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/info"
```
Explain to the evaluator:
- Exactly 77 ordered features are accepted.
- Operating threshold is frozen at `0.29` (selected for $\le 0.2\%$ False Positive Rate).
- Missing value policy uses training-set medians stored inside the Scikit-learn Pipeline.

---

## Minute 3: Batch CSV Scoring & Alert Persistence

1. Navigate to the Streamlit UI at `http://127.0.0.1:8501`.
2. Ensure the top status banner displays:
   `Backend: Online | Model: baseline_v1 | Database: Ready`
3. Click on the **Batch Flow Prediction** tab.
4. Upload the benchmark demo file located at:
   `data/demo/example_flows.csv`
   *(Or click "Download 77-Feature Header Template" to inspect the exact CSV schema).*
5. The CSV validator will confirm: `10 valid flows detected across 77 features.`
6. Click the **"Analyze Batch Flows"** button **ONCE**.

### Expected Results:
- **Scored Summary:** 10 flows analyzed.
- **Classification Output:**
  - **5 flows classified as BENIGN** (Flows 1–5: Ephemeral high-port traffic, probabilities $< 0.29$).
  - **5 flows classified as ATTACK** (Flows 6–10: Port 80 DoS Hulk/GoldenEye/Slowhttptest flows, probabilities $> 0.29$).
- **Color-Coded Badges:** Red badges for `ATTACK` (e.g., $P \approx 0.96$ to $1.00$), green badges for `BENIGN`.

> [!IMPORTANT]
> **Persistence Notice:** Each click sends a `POST /predict/batch` request. Database transactions are atomic, but HTTP requests are not idempotent. Clicking "Analyze" multiple times will add duplicate historical records to the database.

---

## Minute 4: Alerts Feed & System Statistics

1. Switch to the **Alerts Feed** tab in the dashboard.
   - The dashboard calls `GET /alerts?limit=50&offset=0`.
   - The 5 newly classified ATTACK records are displayed with UUID prediction IDs, UTC timestamps, model version `baseline_v1`, threshold `0.29`, and probability scores.
2. Switch to the **System Statistics** tab.
   - The dashboard calls `GET /statistics`.
   - View aggregate metrics: Total predictions, count of benign vs. attack, and attack proportion.
   - Point out the explicit disclaimer rendered at the bottom: *Historical counts reflect local predictions and local test submissions, not a certified live-network threat posture.*

---

## Minute 5: Error Handling & Limitations Defense

### 1. Demonstrate Negative Testing (Schema Rejection)
To prove robustness against malformed inputs:
- **Client-Side CSV Validation in Dashboard:**
  - Upload a non-compliant CSV in the Streamlit UI (e.g., missing features or extra columns).
  - **Result:** The dashboard CSV validator (`app/dashboard/csv_utils.py`) intercepts the file client-side, halting processing before issuing any network call.
  - The UI displays: `CSV Validation Error: Missing required features [...]`.
  - *Note: Because the upload is halted client-side, no network request is sent to FastAPI and no HTTP 422 is generated.*
- **Server-Side API Rejection (HTTP 422):**
  - To demonstrate the backend's independent defense, send an invalid payload directly to the API:
    ```powershell
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"features": {"Destination Port": "invalid_string"}}'
    ```
  - **Result:** The FastAPI backend returns `HTTP 422 (Unprocessable Content)` with precise Pydantic field validation errors, protecting the model and database from corrupt inputs.

### 2. Show Held-Out Benchmark Results Separately
Emphasize to the evaluator:
- The 10-flow demo CSV is a functional verification sample.
- Scientific performance was evaluated **once on 416,144 held-out test flows**:
  - ATTACK Recall: **99.73%** (76,681 / 76,888 detected).
  - False Positive Rate: **0.1886%** (640 / 339,256 benign misclassified).
  - Average Precision: **99.96%**.
- Open `reports/generated/test_evaluation.md` (or `docs/results/test_evaluation.md`) to show the full confusion matrix and per-attack breakdown.

### 3. Truthfully State Limitations
Close the demonstration with three clear technical boundaries:
1. **Passive Flow Records Only:** The system processes extracted bidirectional summary statistics; it does not capture raw Ethernet frames or sniff live packets.
2. **Bot Detection Weakness:** Ares Botnet recall is 59.52% on the test set due to behavioral overlap with periodic HTTP polling.
3. **Uncalibrated Scores:** Output probabilities represent tree ensemble voting proportions, not mathematical Bayesian posteriors.

---

## Service Teardown
When the demonstration is finished:
- In Terminal 1 (Uvicorn): Press `Ctrl + C`.
- In Terminal 2 (Streamlit): Press `Ctrl + C`.
- Confirm ports are released:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000, 8501 -ErrorAction SilentlyContinue
  ```
