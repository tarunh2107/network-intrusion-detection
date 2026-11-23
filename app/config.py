"""Application configuration, environment management, and verified model loading.

Phase 7 Requirements:
- Reads configuration from environment variables with documented defaults.
- Verifies model SHA-256 against frozen local metadata before deserialization.
- Rejects missing, corrupted, or contradictory metadata (no silent fallbacks).
- Loads model pipeline once during application lifecycle.
- Resolves ATTACK probability dynamically via classes_.
- Prohibits client-supplied model paths or upload endpoints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline

from app.ml.evaluate import get_attack_class_index
from app.ml.preprocess import MODEL_FEATURE_COUNT

logger = logging.getLogger(__name__)

# Base workspace directory
WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_dotenv_file(env_path: Optional[pathlib.Path] = None) -> None:
    """Load key-value pairs from .env file into os.environ if the file exists.

    Does not overwrite existing environment variables.
    """
    path = env_path or (WORKSPACE_ROOT / ".env")
    if not path.is_file():
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        logger.warning("Failed reading .env file %s: %s", path, exc)


@dataclasses.dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from environment variables."""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    model_path: pathlib.Path = pathlib.Path("models/baseline_v1/model.joblib")
    frozen_config_path: pathlib.Path = pathlib.Path("models/baseline_v1/frozen_evaluation_config.json")
    database_path: pathlib.Path = pathlib.Path("data/predictions.db")
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Resolve settings from environment variables with documented defaults."""
    load_dotenv_file()

    def resolve_path(val: str, default: pathlib.Path) -> pathlib.Path:
        if not val:
            return default if default.is_absolute() else (WORKSPACE_ROOT / default)
        p = pathlib.Path(val)
        return p if p.is_absolute() else (WORKSPACE_ROOT / p)

    return Settings(
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        model_path=resolve_path(
            os.getenv("MODEL_PATH", ""),
            pathlib.Path("models/baseline_v1/model.joblib"),
        ),
        frozen_config_path=resolve_path(
            os.getenv("FROZEN_CONFIG_PATH", ""),
            pathlib.Path("models/baseline_v1/frozen_evaluation_config.json"),
        ),
        database_path=resolve_path(
            os.getenv("DATABASE_PATH", ""),
            pathlib.Path("data/predictions.db"),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def compute_file_sha256(filepath: pathlib.Path) -> str:
    """Calculate streaming SHA-256 checksum of a file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found for checksum verification: {filepath}")
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


@dataclasses.dataclass
class VerifiedModel:
    """Container for verified pipeline and frozen metadata loaded once during lifespan."""
    pipeline: Pipeline
    model_version: str
    feature_names: List[str]
    frozen_threshold: float
    attack_col_idx: int
    sha256: str
    classes: List[int]
    operating_rule: str


def load_verified_model(settings: Optional[Settings] = None) -> VerifiedModel:
    """Load only the configured, locally generated model and verify consistency.

    Validates:
    - Existence of model artifact and frozen configuration JSON.
    - Checksum matches recorded SHA-256 in trusted local metadata.
    - Consistency of schema, feature order, classes, and threshold.
    - Resolves ATTACK column index dynamically from classifier classes.

    Raises:
        FileNotFoundError: If model or configuration file is missing.
        ValueError: If checksum or metadata contradicts the model contract.
    """
    cfg = settings or get_settings()

    if not cfg.frozen_config_path.is_file():
        raise FileNotFoundError(
            f"Frozen configuration file missing at: {cfg.frozen_config_path}"
        )
    if not cfg.model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact file missing at: {cfg.model_path}"
        )

    # 1. Parse frozen metadata
    try:
        with open(cfg.frozen_config_path, "r", encoding="utf-8") as f:
            frozen_data = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse frozen configuration at {cfg.frozen_config_path}: {exc}"
        ) from exc

    expected_sha256 = frozen_data.get("model_sha256")
    if not expected_sha256:
        raise ValueError(
            f"Corrupted configuration: 'model_sha256' missing in {cfg.frozen_config_path}"
        )

    frozen_threshold = frozen_data.get("selected_threshold")
    if frozen_threshold is None or not (0.0 < float(frozen_threshold) < 1.0):
        raise ValueError(
            f"Corrupted configuration: invalid 'selected_threshold': {frozen_threshold}"
        )
    frozen_threshold = float(frozen_threshold)

    feature_names = frozen_data.get("feature_names", [])
    if len(feature_names) != MODEL_FEATURE_COUNT:
        raise ValueError(
            f"Corrupted configuration: expected {MODEL_FEATURE_COUNT} features, got {len(feature_names)}"
        )

    operating_rule = frozen_data.get("operating_rule", "Unspecified operating rule")

    # 2. Checksum verification against trusted local metadata
    actual_sha256 = compute_file_sha256(cfg.model_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Model checksum mismatch: expected {expected_sha256}, got {actual_sha256}. "
            "Model file may be corrupted or replaced."
        )

    # 3. Load Scikit-learn Pipeline
    try:
        pipeline = joblib.load(cfg.model_path)
    except Exception as exc:
        raise ValueError(f"Failed to deserialize model at {cfg.model_path}: {exc}") from exc

    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected scikit-learn Pipeline, got {type(pipeline)}")

    # 4. Validate classifier classes and locate ATTACK probability index
    clf = pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "classes_"):
        raise ValueError("Model pipeline must contain a 'clf' step with a 'classes_' attribute.")

    attack_idx = get_attack_class_index(clf.classes_)
    classes_list = [int(c) for c in clf.classes_]

    model_version = "baseline_v1"

    logger.info(
        "Successfully verified and loaded model %s (SHA-256: %s, Threshold: %.2f)",
        model_version,
        actual_sha256[:8],
        frozen_threshold,
    )

    return VerifiedModel(
        pipeline=pipeline,
        model_version=model_version,
        feature_names=feature_names,
        frozen_threshold=frozen_threshold,
        attack_col_idx=attack_idx,
        sha256=actual_sha256,
        classes=classes_list,
        operating_rule=operating_rule,
    )
