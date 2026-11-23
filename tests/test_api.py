"""Comprehensive automated tests for Phase 7 FastAPI inference backend and SQLite persistence."""

import datetime
import pathlib
import uuid
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings, load_verified_model
from app.ml.predict import validate_and_extract_feature_vector
from app.ml.preprocess import MODEL_FEATURE_NAMES
from app.services.storage import (
    check_database_readiness,
    get_alerts,
    get_statistics,
    init_database,
    record_prediction,
)


@pytest.fixture
def temp_db_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Fixture providing an initialized temporary SQLite database path."""
    db_file = tmp_path / "test_predictions.db"
    init_database(db_file)
    return db_file


@pytest.fixture
def valid_features_dict() -> Dict[str, Any]:
    """Fixture providing a valid 77-feature dictionary with standard finite values."""
    return {name: float(i + 1) for i, name in enumerate(MODEL_FEATURE_NAMES)}


@pytest.fixture
def client(temp_db_path: pathlib.Path) -> TestClient:
    """Fixture providing a TestClient configured with real baseline model and temporary DB."""
    settings = Settings(
        api_host="127.0.0.1",
        api_port=8000,
        model_path=pathlib.Path("models/baseline_v1/model.joblib"),
        frozen_config_path=pathlib.Path("models/baseline_v1/frozen_evaluation_config.json"),
        database_path=temp_db_path,
        log_level="INFO",
    )
    app = create_app(custom_settings=settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_healthy(client: TestClient):
    """Verify /health returns 200 when model and database are ready."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service_ready"] is True
    assert data["model_loaded"] is True
    assert data["database_connected"] is True


def test_model_info_endpoint(client: TestClient):
    """Verify /model/info returns frozen contract, threshold, and feature schema."""
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "baseline_v1"
    assert data["feature_count"] == 77
    assert data["feature_names"] == MODEL_FEATURE_NAMES
    assert data["decision_threshold"] == 0.29
    assert data["classes"] == {"0": "BENIGN", "1": "ATTACK"}
    assert "offline" in data["offline_use_limitations"].lower()


def test_predict_single_flow_success(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify single flow prediction returns required schema and persists record."""
    resp = client.post("/predict", json={"features": valid_features_dict})
    assert resp.status_code == 200
    data = resp.json()

    assert "prediction_id" in data
    assert "timestamp_utc" in data
    assert data["prediction_label"] in ("BENIGN", "ATTACK")
    assert isinstance(data["attack_probability"], float)
    assert 0.0 <= data["attack_probability"] <= 1.0
    assert data["decision_threshold"] == 0.29
    assert data["model_version"] == "baseline_v1"

    # Verify persisted in database
    stats = client.get("/statistics").json()
    assert stats["total_predictions"] == 1


def test_feature_ordering_independence(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify JSON key order does not alter predicted vector or result."""
    # Reverse key order
    reversed_dict = {k: valid_features_dict[k] for k in reversed(list(valid_features_dict.keys()))}

    resp1 = client.post("/predict", json={"features": valid_features_dict})
    resp2 = client.post("/predict", json={"features": reversed_dict})

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert pytest.approx(resp1.json()["attack_probability"], abs=1e-6) == resp2.json()["attack_probability"]
    assert resp1.json()["prediction_label"] == resp2.json()["prediction_label"]


def test_feature_validation_rejections(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify strict rejection of missing keys, unknown keys, strings, booleans, and non-finite values."""
    # 1. Missing feature
    missing_dict = dict(valid_features_dict)
    del missing_dict["Flow Duration"]
    r_missing = client.post("/predict", json={"features": missing_dict})
    assert r_missing.status_code == 422
    assert "Missing 1 required features" in r_missing.json()["detail"]

    # 2. Unknown extra feature
    extra_dict = dict(valid_features_dict)
    extra_dict["Unexpected_Column"] = 123.0
    r_extra = client.post("/predict", json={"features": extra_dict})
    assert r_extra.status_code == 422
    assert "Unexpected 1 unknown features" in r_extra.json()["detail"]

    # 3. String value
    str_dict = dict(valid_features_dict)
    str_dict["Flow Duration"] = "1000"
    r_str = client.post("/predict", json={"features": str_dict})
    assert r_str.status_code == 422
    assert "invalid type str" in r_str.json()["detail"]

    # 4. Boolean value (must not be silently coerced to 1.0/0.0)
    bool_dict = dict(valid_features_dict)
    bool_dict["Fwd PSH Flags"] = True
    r_bool = client.post("/predict", json={"features": bool_dict})
    assert r_bool.status_code == 422
    assert "boolean value True" in r_bool.json()["detail"]

    # 5. Non-finite values
    inf_dict = dict(valid_features_dict)
    inf_dict["Flow Packets/s"] = float("inf")
    with pytest.raises(ValueError, match="non-finite value"):
        validate_and_extract_feature_vector(inf_dict, MODEL_FEATURE_NAMES)

    nan_dict = dict(valid_features_dict)
    nan_dict["Flow Packets/s"] = float("nan")
    with pytest.raises(ValueError, match="non-finite value"):
        validate_and_extract_feature_vector(nan_dict, MODEL_FEATURE_NAMES)

    # Rejection of string infinity over HTTP
    str_inf_dict = dict(valid_features_dict)
    str_inf_dict["Flow Packets/s"] = "Infinity"
    r_str_inf = client.post("/predict", json={"features": str_inf_dict})
    assert r_str_inf.status_code == 422

    # 6. Entirely null vector
    all_null_dict = {name: None for name in MODEL_FEATURE_NAMES}
    r_null = client.post("/predict", json={"features": all_null_dict})
    assert r_null.status_code == 422
    assert "entirely null" in r_null.json()["detail"]


def test_explicit_null_and_negative_sentinel_handling(valid_features_dict: Dict[str, Any]):
    """Verify explicit nulls become NaN, and negative rate cleaning matches training."""
    test_dict = dict(valid_features_dict)
    test_dict["Flow Bytes/s"] = None  # explicit null -> nan
    test_dict["Flow Packets/s"] = -500.0  # negative rate -> nan
    test_dict["Init_Win_bytes_forward"] = -1.0  # negative sentinel -> preserved

    vec = validate_and_extract_feature_vector(test_dict, MODEL_FEATURE_NAMES)
    bytes_idx = MODEL_FEATURE_NAMES.index("Flow Bytes/s")
    packets_idx = MODEL_FEATURE_NAMES.index("Flow Packets/s")
    win_idx = MODEL_FEATURE_NAMES.index("Init_Win_bytes_forward")

    assert np.isnan(vec[bytes_idx])
    assert np.isnan(vec[packets_idx])
    assert vec[win_idx] == -1.0


def test_predict_batch_success_and_limits(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify batch inference handles 1-500 flows, preserves order, and rejects invalid sizes."""
    # 1. Valid batch of 3 flows
    batch_req = {"flows": [{"features": valid_features_dict} for _ in range(3)]}
    r_batch = client.post("/predict/batch", json=batch_req)
    assert r_batch.status_code == 200
    data = r_batch.json()
    assert data["batch_size"] == 3
    assert len(data["predictions"]) == 3

    # Check persistence
    stats = client.get("/statistics").json()
    assert stats["total_predictions"] == 3

    # 2. Empty batch rejection
    r_empty = client.post("/predict/batch", json={"flows": []})
    assert r_empty.status_code == 422

    # 3. Oversized batch (> 500) rejection
    oversized = {"flows": [{"features": valid_features_dict} for _ in range(501)]}
    r_over = client.post("/predict/batch", json=oversized)
    assert r_over.status_code == 422
    assert "exceeds maximum limit of 500" in r_over.json()["detail"]


def test_threshold_boundary_classification(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify classification decision boundary: score >= threshold -> ATTACK, else BENIGN."""
    app_model = client.app.state.model

    # Score right below threshold (0.2899)
    with patch.object(app_model.pipeline, "predict_proba", return_value=np.array([[0.7101, 0.2899]])):
        r_below = client.post("/predict", json={"features": valid_features_dict})
        assert r_below.json()["attack_probability"] == 0.2899
        assert r_below.json()["prediction_label"] == "BENIGN"

    # Score right at threshold (0.2900)
    with patch.object(app_model.pipeline, "predict_proba", return_value=np.array([[0.7100, 0.2900]])):
        r_at = client.post("/predict", json={"features": valid_features_dict})
        assert r_at.json()["attack_probability"] == 0.2900
        assert r_at.json()["prediction_label"] == "ATTACK"

    # Score above threshold (0.2901)
    with patch.object(app_model.pipeline, "predict_proba", return_value=np.array([[0.7099, 0.2901]])):
        r_above = client.post("/predict", json={"features": valid_features_dict})
        assert r_above.json()["attack_probability"] == 0.2901
        assert r_above.json()["prediction_label"] == "ATTACK"


def test_api_predictions_match_direct_pipeline(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify API prediction score exactly matches direct pipeline evaluation."""
    app_model = client.app.state.model
    vec = validate_and_extract_feature_vector(valid_features_dict, app_model.feature_names)
    expected_prob = float(app_model.pipeline.predict_proba(vec.reshape(1, -1))[0, app_model.attack_col_idx])
    expected_label = "ATTACK" if expected_prob >= app_model.frozen_threshold else "BENIGN"

    resp = client.post("/predict", json={"features": valid_features_dict})
    data = resp.json()
    assert pytest.approx(data["attack_probability"], abs=1e-6) == expected_prob
    assert data["prediction_label"] == expected_label


def test_atomic_batch_rollback_on_failure(client: TestClient, valid_features_dict: Dict[str, Any]):
    """Verify batch persistence failure rolls back all records atomically and returns 500."""
    initial_stats = client.get("/statistics").json()
    assert initial_stats["total_predictions"] == 0

    batch_payload = {"flows": [{"features": valid_features_dict} for _ in range(4)]}

    with patch("app.api.main.record_predictions_batch", side_effect=RuntimeError("Simulated DB Crash")):
        resp = client.post("/predict/batch", json=batch_payload)
        assert resp.status_code == 500
        assert "Batch persistence error" in resp.json()["detail"]

    # Verify no records were partially written
    after_stats = client.get("/statistics").json()
    assert after_stats["total_predictions"] == 0


def test_alerts_filtering_pagination_and_empty_stats(temp_db_path: pathlib.Path, client: TestClient):
    """Verify empty statistics, alerts pagination, and invalid date range handling."""
    # 1. Empty database statistics
    stats_empty = client.get("/statistics").json()
    assert stats_empty["total_predictions"] == 0
    assert stats_empty["benign_predictions"] == 0
    assert stats_empty["attack_predictions"] == 0
    assert stats_empty["attack_proportion"] == 0.0

    # 2. Insert 3 attack records and 2 benign records directly into DB
    t1 = "2026-10-13T01:00:00+00:00"
    t2 = "2026-10-13T02:00:00+00:00"
    t3 = "2026-10-13T03:00:00+00:00"
    record_prediction(temp_db_path, "atk_1", t1, "v1", 0.29, 0.85, "ATTACK")
    record_prediction(temp_db_path, "atk_2", t2, "v1", 0.29, 0.90, "ATTACK")
    record_prediction(temp_db_path, "atk_3", t3, "v1", 0.29, 0.95, "ATTACK")
    record_prediction(temp_db_path, "ben_1", t1, "v1", 0.29, 0.02, "BENIGN")
    record_prediction(temp_db_path, "ben_2", t2, "v1", 0.29, 0.04, "BENIGN")

    # 3. Query alerts pagination (limit 2, offset 0)
    r_p1 = client.get("/alerts?limit=2&offset=0")
    assert r_p1.status_code == 200
    p1 = r_p1.json()
    assert p1["total_alerts"] == 3
    assert len(p1["alerts"]) == 2
    # Ordered descending by timestamp: atk_3 (03:00) first, then atk_2 (02:00)
    assert p1["alerts"][0]["prediction_id"] == "atk_3"
    assert p1["alerts"][1]["prediction_id"] == "atk_2"

    # Query offset 2
    r_p2 = client.get("/alerts?limit=2&offset=2")
    assert len(r_p2.json()["alerts"]) == 1
    assert r_p2.json()["alerts"][0]["prediction_id"] == "atk_1"

    # 4. Date filtering
    r_date = client.get(f"/alerts?start_time={t2}&end_time={t3}")
    assert r_date.status_code == 200
    assert r_date.json()["total_alerts"] == 2

    # 5. Invalid date filter (start_time > end_time)
    r_inv = client.get(f"/alerts?start_time={t3}&end_time={t1}")
    assert r_inv.status_code == 400
    assert "cannot be later than" in r_inv.json()["detail"]


def test_restart_persistence(temp_db_path: pathlib.Path, valid_features_dict: Dict[str, Any]):
    """Verify stored prediction history survives a simulated server restart."""
    settings = Settings(
        api_host="127.0.0.1",
        api_port=8000,
        model_path=pathlib.Path("models/baseline_v1/model.joblib"),
        frozen_config_path=pathlib.Path("models/baseline_v1/frozen_evaluation_config.json"),
        database_path=temp_db_path,
        log_level="INFO",
    )

    # Server instance 1: record 2 predictions
    app1 = create_app(custom_settings=settings)
    with TestClient(app1) as c1:
        c1.post("/predict", json={"features": valid_features_dict})
        c1.post("/predict", json={"features": valid_features_dict})
        assert c1.get("/statistics").json()["total_predictions"] == 2

    # Server instance 2 (simulating restart using same database path)
    app2 = create_app(custom_settings=settings)
    with TestClient(app2) as c2:
        stats = c2.get("/statistics").json()
        assert stats["total_predictions"] == 2


def test_health_check_unhealthy_on_missing_model_or_db(tmp_path: pathlib.Path):
    """Verify /health returns HTTP 503 when dependencies are unavailable."""
    bad_settings = Settings(
        api_host="127.0.0.1",
        api_port=8000,
        model_path=tmp_path / "non_existent_model.joblib",
        frozen_config_path=tmp_path / "non_existent_config.json",
        database_path=tmp_path / "test.db",
        log_level="INFO",
    )
    app = create_app(custom_settings=bad_settings)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["service_ready"] is False
        assert data["model_loaded"] is False

        # Endpoints requiring model return 503
        r_pred = client.post("/predict", json={"features": {}})
        assert r_pred.status_code == 503
