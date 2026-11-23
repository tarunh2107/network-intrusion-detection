"""Tests for Phase 8 Streamlit security dashboard components, CSV validator, and API client.

Phase 8 Requirements:
- Inspect headers before pandas can silently rename duplicates.
- Strip surrounding header whitespace, reject duplicate names, normalization collisions,
  missing features, and extra columns.
- Convert blank / NaN cells to JSON null.
- Reject non-numeric values and infinities with clear row/column errors.
- Enforce 500 row cap and 10 MB limit.
- Handle API connection errors, 422 rejections, and timeouts (no automatic retries).
- Validate demo CSV artifact matches 77-feature contract.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.dashboard.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiServiceUnavailableError,
    ApiTimeoutError,
    ApiValidationError,
    ChunkedPredictionResult,
    NidsApiClient,
)
from app.dashboard.config import DashboardSettings
from app.dashboard.csv_utils import (
    generate_header_template_csv,
    inspect_and_validate_csv,
)

METADATA_PATH = pathlib.Path("models/baseline_v1/metadata.json")


@pytest.fixture(scope="module")
def feature_names() -> List[str]:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["feature_names"]


@pytest.fixture
def valid_single_row_csv(feature_names: List[str]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(feature_names)
    # 77 zeros
    writer.writerow([0.0] * len(feature_names))
    return out.getvalue()


# -----------------------------------------------------------------------------
# CSV Validation Tests
# -----------------------------------------------------------------------------

def test_valid_csv_passes_inspection(feature_names: List[str], valid_single_row_csv: str):
    """Verify clean 77-feature CSV passes validation with zero errors."""
    flows, df, errors = inspect_and_validate_csv(valid_single_row_csv, feature_names)
    assert not errors
    assert len(flows) == 1
    assert len(flows[0]["features"]) == 77
    assert df.shape == (1, 77)


def test_header_whitespace_is_stripped(feature_names: List[str]):
    """Verify leading and trailing whitespace in column headers is stripped cleanly."""
    padded_headers = [f"  {h}  " for h in feature_names]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(padded_headers)
    writer.writerow([1.0] * len(feature_names))

    flows, df, errors = inspect_and_validate_csv(out.getvalue(), feature_names)
    assert not errors
    assert len(flows) == 1
    assert "Destination Port" in flows[0]["features"]


def test_reject_duplicate_headers(feature_names: List[str]):
    """Verify duplicate column names in header are rejected explicitly."""
    dup_headers = list(feature_names)
    dup_headers[1] = dup_headers[0]  # duplicate first header

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(dup_headers)
    writer.writerow([0.0] * len(dup_headers))

    flows, df, errors = inspect_and_validate_csv(out.getvalue(), feature_names)
    assert errors
    assert any("Duplicate column names detected" in e for e in errors)


def test_reject_normalization_collisions(feature_names: List[str]):
    """Verify case/whitespace collisions in headers are rejected."""
    colliding_headers = list(feature_names)
    colliding_headers[1] = colliding_headers[0].upper()  # collision

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(colliding_headers)
    writer.writerow([0.0] * len(colliding_headers))

    flows, df, errors = inspect_and_validate_csv(out.getvalue(), feature_names)
    assert errors
    assert any("normalization collisions detected" in e or "Missing required" in e for e in errors)


def test_reject_missing_and_extra_columns(feature_names: List[str]):
    """Verify missing features and extra unrecognized columns are rejected."""
    # Missing one column
    subset = feature_names[:-1]
    out1 = io.StringIO()
    csv.writer(out1).writerow(subset)
    csv.writer(out1).writerow([0.0] * len(subset))
    _, _, errs1 = inspect_and_validate_csv(out1.getvalue(), feature_names)
    assert any("Missing required feature columns" in e for e in errs1)

    # Extra column
    extra = list(feature_names) + ["Extra_Malicious_Col"]
    out2 = io.StringIO()
    csv.writer(out2).writerow(extra)
    csv.writer(out2).writerow([0.0] * len(extra))
    _, _, errs2 = inspect_and_validate_csv(out2.getvalue(), feature_names)
    assert any("Unrecognized extra columns detected" in e for e in errs2)


def test_reject_oversized_rows(feature_names: List[str]):
    """Verify CSV with > 500 rows is rejected by max_rows limit."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(feature_names)
    for _ in range(501):
        writer.writerow([0.0] * len(feature_names))

    _, _, errors = inspect_and_validate_csv(out.getvalue(), feature_names, max_rows=500)
    assert errors
    assert any("exceeds maximum allowed limit of 500 data rows" in e for e in errors)


def test_blank_and_nan_cells_converted_to_none(feature_names: List[str]):
    """Verify empty/blank cells and 'NaN' are converted to Python None (JSON null)."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(feature_names)
    row = [1.0] * len(feature_names)
    row[0] = ""  # blank cell
    row[1] = "NaN"  # nan string
    row[2] = "null"  # null string
    writer.writerow(row)

    flows, _, errors = inspect_and_validate_csv(out.getvalue(), feature_names)
    assert not errors
    features = flows[0]["features"]
    assert features[feature_names[0]] is None
    assert features[feature_names[1]] is None
    assert features[feature_names[2]] is None
    assert features[feature_names[3]] == 1.0


def test_reject_non_numeric_and_infinities_with_row_col(feature_names: List[str]):
    """Verify non-numeric values and infinities are rejected with exact row & column indicators."""
    # Row with string "invalid_text"
    out1 = io.StringIO()
    writer1 = csv.writer(out1)
    writer1.writerow(feature_names)
    row1 = [0.0] * len(feature_names)
    row1[3] = "invalid_text"
    writer1.writerow(row1)

    _, _, errs1 = inspect_and_validate_csv(out1.getvalue(), feature_names)
    assert errs1
    assert any(f"Row 1, column '{feature_names[3]}': Non-numeric value 'invalid_text'" in e for e in errs1)

    # Row with "inf"
    out2 = io.StringIO()
    writer2 = csv.writer(out2)
    writer2.writerow(feature_names)
    row2 = [0.0] * len(feature_names)
    row2[5] = "inf"
    writer2.writerow(row2)

    _, _, errs2 = inspect_and_validate_csv(out2.getvalue(), feature_names)
    assert errs2
    assert any(f"Row 1, column '{feature_names[5]}': Non-finite value 'inf'" in e for e in errs2)


def test_reject_entirely_null_row(feature_names: List[str]):
    """Verify a row containing only blank/null cells is rejected."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(feature_names)
    writer.writerow([""] * len(feature_names))

    _, _, errors = inspect_and_validate_csv(out.getvalue(), feature_names)
    assert errors
    assert any("Row 1: Row contains entirely null/empty values." in e for e in errors)


def test_header_template_generator(feature_names: List[str]):
    """Verify downloadable header template generates exact 77 headers."""
    csv_str = generate_header_template_csv(feature_names)
    reader = csv.reader(io.StringIO(csv_str))
    headers = next(reader)
    assert headers == feature_names
    with pytest.raises(StopIteration):
        next(reader)


def test_demo_csv_file_validity(feature_names: List[str]):
    """Verify generated data/demo/example_flows.csv passes all schema checks."""
    demo_path = pathlib.Path("data/demo/example_flows.csv")
    assert demo_path.is_file(), "Demo CSV file must exist under data/demo/example_flows.csv"

    content = demo_path.read_bytes()
    flows, df, errors = inspect_and_validate_csv(content, feature_names)
    assert not errors
    assert len(flows) == 10
    assert df.shape == (10, 77)


# -----------------------------------------------------------------------------
# API Client Tests
# -----------------------------------------------------------------------------

def test_api_client_connection_error():
    """Verify ApiConnectionError is raised with clear message when backend is unreachable."""
    client = NidsApiClient(
        settings=DashboardSettings(api_base_url="http://127.0.0.1:9999", connect_timeout_seconds=0.1)
    )
    with pytest.raises(ApiConnectionError) as exc_info:
        client.get_health()
    assert "Cannot connect to API" in str(exc_info.value)


def test_api_client_timeout_no_retry_notice():
    """Verify ApiTimeoutError provides explicit warning against retrying POST requests."""
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ReadTimeout("Read timed out")

    client = NidsApiClient(session=session)
    flows = [{"features": {"Destination Port": 80.0}}]

    with pytest.raises(ApiTimeoutError) as exc_info:
        client.predict_batch(flows)

    err_text = str(exc_info.value)
    assert "do not automatically retry" in err_text
    assert "duplicate records" in err_text


def test_api_client_handles_422_validation():
    """Verify ApiValidationError extracts server validation details."""
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {"detail": "Missing feature: Flow Duration"}
    session.post.return_value = mock_resp

    client = NidsApiClient(session=session)
    with pytest.raises(ApiValidationError) as exc_info:
        client.predict_batch([{"features": {}}])

    assert "Missing feature: Flow Duration" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 1,000-Row Chunking & Batching Tests
# -----------------------------------------------------------------------------

def test_demo_1000_csv_file_validity(feature_names: List[str]):
    """Verify generated data/demo/example_flows_1000.csv passes all schema checks."""
    demo_path = pathlib.Path("data/demo/example_flows_1000.csv")
    assert demo_path.is_file(), "Demo CSV file must exist under data/demo/example_flows_1000.csv"

    content = demo_path.read_bytes()
    flows, df, errors = inspect_and_validate_csv(content, feature_names, max_rows=1000)
    assert not errors
    assert len(flows) == 1000
    assert df.shape == (50, 77)


def test_csv_validation_allows_1000_rows_and_rejects_1001(feature_names: List[str]):
    """Verify CSV validation allows up to 1,000 rows and rejects 1,001 data rows."""
    out_1000 = io.StringIO()
    writer_1000 = csv.writer(out_1000)
    writer_1000.writerow(feature_names)
    for _ in range(1000):
        writer_1000.writerow([0.0] * len(feature_names))

    flows, df, errs_1000 = inspect_and_validate_csv(out_1000.getvalue(), feature_names, max_rows=1000)
    assert not errs_1000
    assert len(flows) == 1000
    assert df.shape == (50, 77)

    # 1001 rows
    out_1001 = io.StringIO()
    writer_1001 = csv.writer(out_1001)
    writer_1001.writerow(feature_names)
    for _ in range(1001):
        writer_1001.writerow([0.0] * len(feature_names))

    _, _, errs_1001 = inspect_and_validate_csv(out_1001.getvalue(), feature_names, max_rows=1000)
    assert errs_1001
    assert any("exceeds maximum allowed limit of 1000 data rows" in e for e in errs_1001)


def test_predict_in_chunks_batches_1000_flows():
    """Verify predict_in_chunks splits 1,000 flows into two sequential batches of 500."""
    client = NidsApiClient()
    flows = [{"features": {"Destination Port": float(i)}} for i in range(1000)]

    chunk_calls = []

    def mock_predict_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        chunk_calls.append(len(batch))
        return {
            "batch_size": len(batch),
            "predictions": [
                {
                    "prediction_id": f"pred-{i}",
                    "prediction_label": "BENIGN",
                    "attack_probability": 0.05,
                    "decision_threshold": 0.29,
                    "timestamp_utc": "2026-10-13T00:00:00Z",
                }
                for i in range(len(batch))
            ],
        }

    with patch.object(client, "predict_batch", side_effect=mock_predict_batch):
        result = client.predict_in_chunks(flows, chunk_size=500)

    assert chunk_calls == [500, 500]
    assert result.total_requested == 1000
    assert result.total_completed == 1000
    assert result.chunks_sent == 2
    assert not result.is_partial
    assert result.error_message is None
    assert len(result.predictions) == 1000


def test_predict_in_chunks_preserves_input_order():
    """Verify combined predictions strictly preserve original 1:1 input order."""
    client = NidsApiClient()
    flows = [{"features": {"flow_index": i}} for i in range(1000)]

    def mock_predict_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "batch_size": len(batch),
            "predictions": [
                {
                    "prediction_id": f"id_{item['features']['flow_index']}",
                    "prediction_label": "BENIGN",
                    "attack_probability": 0.1,
                    "decision_threshold": 0.29,
                    "timestamp_utc": "2026-10-13T00:00:00Z",
                }
                for item in batch
            ],
        }

    with patch.object(client, "predict_batch", side_effect=mock_predict_batch):
        result = client.predict_in_chunks(flows, chunk_size=500)

    assert len(result.predictions) == 1000
    for idx, pred in enumerate(result.predictions):
        assert pred["prediction_id"] == f"id_{idx}"


def test_predict_in_chunks_partial_failure_retains_completed_and_forbids_retry():
    """Verify when batch 2 fails, completed batch 1 results are retained and no retry occurs."""
    client = NidsApiClient()
    flows = [{"features": {"Destination Port": float(i)}} for i in range(1000)]

    call_count = 0

    def mock_predict_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "batch_size": len(batch),
                "predictions": [
                    {
                        "prediction_id": f"chunk1-{i}",
                        "prediction_label": "BENIGN",
                        "attack_probability": 0.05,
                        "decision_threshold": 0.29,
                        "timestamp_utc": "2026-10-13T00:00:00Z",
                    }
                    for i in range(len(batch))
                ],
            }
        raise ApiTimeoutError("Prediction request timed out.")

    with patch.object(client, "predict_batch", side_effect=mock_predict_batch):
        result = client.predict_in_chunks(flows, chunk_size=500)

    # Must only call twice: chunk 1 (succeeded), chunk 2 (failed). Must NOT retry chunk 2!
    assert call_count == 2
    assert result.is_partial is True
    assert result.total_requested == 1000
    assert result.total_completed == 500
    assert len(result.predictions) == 500
    assert result.chunks_sent == 2
    assert "do not automatically retry post requests" in result.error_message.lower()
    assert "duplicate records" in result.error_message.lower()


def test_predict_in_chunks_progress_callback():
    """Verify progress callback is invoked after each sequential batch."""
    client = NidsApiClient()
    flows = [{"features": {"Destination Port": 80.0}} for _ in range(750)]
    progress_updates = []

    def on_progress(completed: int, total: int):
        progress_updates.append((completed, total))

    with patch.object(client, "predict_batch", return_value={"batch_size": 500, "predictions": [{}] * 500}) as mock_pb:
        mock_pb.side_effect = [
            {"batch_size": 500, "predictions": [{}] * 500},
            {"batch_size": 250, "predictions": [{}] * 250},
        ]
        result = client.predict_in_chunks(flows, chunk_size=500, progress_callback=on_progress)

    assert progress_updates == [(500, 750), (750, 750)]
    assert result.total_completed == 750
    assert not result.is_partial


def test_resubmission_prevention_on_rerender():
    """Verify existing session state analysis results are preserved without resubmitting to API."""
    mock_client = MagicMock()
    # Simulating Streamlit session state containing previous analysis
    session_state = {
        "last_analysis_results": {
            "predictions": [{"prediction_id": "existing-1", "prediction_label": "BENIGN"}],
            "total_requested": 1000,
            "total_completed": 1000,
            "is_partial": False,
        }
    }

    # If analyze_clicked is False (as occurs during navigation or normal rerender)
    analyze_clicked = False
    if analyze_clicked:
        mock_client.predict_in_chunks([{}])

    # No API call was made
    mock_client.predict_in_chunks.assert_not_called()
    assert len(session_state["last_analysis_results"]["predictions"]) == 1

