"""Verification test for Phase 10: 1,000-row demo batch analysis with an isolated temporary database.

Requirements:
- Validates that data/demo/example_flows_1000.csv is parsed and validated cleanly.
- Executes chunked scoring through NidsApiClient.predict_in_chunks (2 batches of 500 flows).
- Uses an isolated temporary SQLite database, leaving data/predictions.db untouched.
- Verifies that SELECT COUNT(*) FROM predictions in the temporary database is EXACTLY 1,000.
- Verifies that partial failure retains completed batch without automatic retries.
"""

from __future__ import annotations

import pathlib
import sqlite3
from typing import Any, Dict, List
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest
import requests

from app.api.main import create_app
from app.config import Settings
from app.dashboard.api_client import NidsApiClient
from app.dashboard.config import DashboardSettings
from app.dashboard.csv_utils import inspect_and_validate_csv


@pytest.fixture(scope="module")
def example_1000_flows() -> List[Dict[str, Any]]:
    csv_path = pathlib.Path("data/demo/example_flows_1000.csv")
    assert csv_path.is_file(), "example_flows_1000.csv must exist."
    with open("models/baseline_v1/metadata.json", "r", encoding="utf-8") as f:
        import json
        meta = json.load(f)
    feature_names = meta["feature_names"]

    content = csv_path.read_bytes()
    flows, _, errors = inspect_and_validate_csv(content, feature_names, max_rows=1000)
    assert not errors, f"Validation errors found: {errors}"
    assert len(flows) == 1000
    return flows


def test_isolated_temp_db_receives_exactly_1000_records(tmp_path: pathlib.Path, example_1000_flows: List[Dict[str, Any]]):
    """Verify that scoring 1,000 flows adds exactly 1,000 records to an isolated database."""
    temp_db = tmp_path / "isolated_predictions_1000.db"
    real_db = pathlib.Path("data/predictions.db")
    real_db_mtime_before = real_db.stat().st_mtime if real_db.exists() else None

    # Custom settings pointing to temporary database
    custom_settings = Settings(
        database_path=temp_db,
        model_path=pathlib.Path("models/baseline_v1/model.joblib"),
        frozen_config_path=pathlib.Path("models/baseline_v1/frozen_evaluation_config.json"),
    )

    app = create_app(custom_settings=custom_settings)

    with TestClient(app) as test_client:
        # Check health
        health_resp = test_client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["service_ready"] is True

        # Create session adapter delegating to test_client
        session = requests.Session()

        def mock_post(url: str, json: Any = None, timeout: Any = None):
            path = url.replace("http://127.0.0.1:8000", "")
            tc_resp = test_client.post(path, json=json)
            r = requests.Response()
            r.status_code = tc_resp.status_code
            r._content = tc_resp.content
            r.headers.update(tc_resp.headers)
            return r

        session.post = mock_post  # type: ignore[assignment]

        dashboard_settings = DashboardSettings(api_base_url="http://127.0.0.1:8000")
        client = NidsApiClient(settings=dashboard_settings, session=session)

        # Execute chunked prediction across 1,000 flows
        chunk_progress: List[int] = []

        def on_progress(completed: int, total: int):
            chunk_progress.append(completed)

        result = client.predict_in_chunks(example_1000_flows, chunk_size=500, progress_callback=on_progress)

        assert result.total_requested == 1000
        assert result.total_completed == 1000
        assert result.chunks_sent == 2
        assert not result.is_partial
        assert result.error_message is None
        assert len(result.predictions) == 1000
        assert chunk_progress == [500, 1000]

    # Verify directly in SQLite
    assert temp_db.is_file(), "Temporary database must have been created."
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_count = cursor.fetchone()[0]
    assert total_count == 1000, f"Expected exactly 1,000 records in temporary database, got {total_count}."

    cursor.execute("SELECT COUNT(DISTINCT id) FROM predictions")
    distinct_ids = cursor.fetchone()[0]
    assert distinct_ids == 1000, "All 1,000 prediction records must have distinct IDs."

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label IN ('BENIGN', 'ATTACK')")
    valid_labels = cursor.fetchone()[0]
    assert valid_labels == 1000

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE threshold = 0.29")
    valid_thresholds = cursor.fetchone()[0]
    assert valid_thresholds == 1000

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE model_version = 'baseline_v1'")
    valid_models = cursor.fetchone()[0]
    assert valid_models == 1000

    conn.close()

    # Verify real database was NOT modified
    if real_db.exists():
        real_db_mtime_after = real_db.stat().st_mtime
        assert real_db_mtime_before == real_db_mtime_after, "data/predictions.db must remain untouched."


def test_isolated_temp_db_partial_failure_stops_without_retry(tmp_path: pathlib.Path, example_1000_flows: List[Dict[str, Any]]):
    """Verify that if batch 2 fails, exactly 500 records remain and no retries are performed."""
    temp_db = tmp_path / "isolated_partial_failure.db"

    custom_settings = Settings(
        database_path=temp_db,
        model_path=pathlib.Path("models/baseline_v1/model.joblib"),
        frozen_config_path=pathlib.Path("models/baseline_v1/frozen_evaluation_config.json"),
    )

    app = create_app(custom_settings=custom_settings)

    with TestClient(app) as test_client:
        call_count = 0
        session = requests.Session()

        def mock_post(url: str, json: Any = None, timeout: Any = None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                path = url.replace("http://127.0.0.1:8000", "")
                tc_resp = test_client.post(path, json=json)
                r = requests.Response()
                r.status_code = tc_resp.status_code
                r._content = tc_resp.content
                r.headers.update(tc_resp.headers)
                return r
            # Chunk 2 fails with timeout
            raise requests.exceptions.ReadTimeout("Mocked timeout on batch 2")

        session.post = mock_post  # type: ignore[assignment]

        dashboard_settings = DashboardSettings(api_base_url="http://127.0.0.1:8000")
        client = NidsApiClient(settings=dashboard_settings, session=session)

        result = client.predict_in_chunks(example_1000_flows, chunk_size=500)

        assert call_count == 2, "Must NOT attempt automatic retry on chunk 2."
        assert result.is_partial is True
        assert result.total_requested == 1000
        assert result.total_completed == 500
        assert len(result.predictions) == 500
        assert result.chunks_sent == 2

    # Verify database has exactly 500 records from chunk 1
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    assert count == 500, f"Expected exactly 500 records from the completed batch, got {count}."
    conn.close()
