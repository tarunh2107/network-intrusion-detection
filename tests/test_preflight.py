"""Tests for Phase 9 runtime artifact preflight integrity checker.

Phase 9 Requirements:
- Validates model binary, streaming SHA-256 checksum, and metadata schema.
- Verifies frozen threshold configuration and evaluation report.
- Detects missing files and checksum mismatches without deserializing weights.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import pytest

from scripts.check_runtime_artifacts import check_artifacts, compute_file_sha256, EXPECTED_MODEL_SHA256

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_preflight_passes_on_current_workspace():
    """Verify preflight check passes completely on current verified workspace artifacts."""
    model_path = WORKSPACE_ROOT / "models" / "baseline_v1" / "model.joblib"
    frozen_config_path = WORKSPACE_ROOT / "models" / "baseline_v1" / "frozen_evaluation_config.json"
    metadata_path = WORKSPACE_ROOT / "models" / "baseline_v1" / "metadata.json"
    eval_path = WORKSPACE_ROOT / "reports" / "generated" / "test_evaluation.json"
    demo_path = WORKSPACE_ROOT / "data" / "demo" / "example_flows.csv"
    db_path = WORKSPACE_ROOT / "data" / "predictions.db"

    success, errors = check_artifacts(
        model_path=model_path,
        frozen_config_path=frozen_config_path,
        metadata_path=metadata_path,
        evaluation_report_path=eval_path,
        demo_csv_path=demo_path,
        database_path=db_path,
    )
    assert success is True
    assert not errors


def test_preflight_detects_missing_model_file(tmp_path: pathlib.Path):
    """Verify preflight detects missing model file and fails cleanly."""
    non_existent = tmp_path / "missing_model.joblib"
    success, errors = check_artifacts(
        model_path=non_existent,
        frozen_config_path=WORKSPACE_ROOT / "models" / "baseline_v1" / "frozen_evaluation_config.json",
        metadata_path=WORKSPACE_ROOT / "models" / "baseline_v1" / "metadata.json",
        evaluation_report_path=WORKSPACE_ROOT / "reports" / "generated" / "test_evaluation.json",
        demo_csv_path=WORKSPACE_ROOT / "data" / "demo" / "example_flows.csv",
        database_path=tmp_path / "test.db",
    )
    assert success is False
    assert any("Model artifact not found" in e for e in errors)


def test_preflight_detects_checksum_mismatch(tmp_path: pathlib.Path):
    """Verify preflight rejects corrupted/tampered model binary without deserialization."""
    bad_model = tmp_path / "model.joblib"
    bad_model.write_bytes(b"corrupted_or_tampered_model_bytes" * 100)

    success, errors = check_artifacts(
        model_path=bad_model,
        frozen_config_path=WORKSPACE_ROOT / "models" / "baseline_v1" / "frozen_evaluation_config.json",
        metadata_path=WORKSPACE_ROOT / "models" / "baseline_v1" / "metadata.json",
        evaluation_report_path=WORKSPACE_ROOT / "reports" / "generated" / "test_evaluation.json",
        demo_csv_path=WORKSPACE_ROOT / "data" / "demo" / "example_flows.csv",
        database_path=tmp_path / "test.db",
    )
    assert success is False
    assert any("SHA-256 checksum mismatch" in e for e in errors)


def test_preflight_detects_corrupt_config_json(tmp_path: pathlib.Path):
    """Verify preflight detects malformed JSON in frozen evaluation configuration."""
    bad_config = tmp_path / "frozen_config.json"
    bad_config.write_text("{malformed_json: true", encoding="utf-8")

    success, errors = check_artifacts(
        model_path=WORKSPACE_ROOT / "models" / "baseline_v1" / "model.joblib",
        frozen_config_path=bad_config,
        metadata_path=WORKSPACE_ROOT / "models" / "baseline_v1" / "metadata.json",
        evaluation_report_path=WORKSPACE_ROOT / "reports" / "generated" / "test_evaluation.json",
        demo_csv_path=WORKSPACE_ROOT / "data" / "demo" / "example_flows.csv",
        database_path=tmp_path / "test.db",
    )
    assert success is False
    assert any("Failed parsing frozen config" in e for e in errors)
