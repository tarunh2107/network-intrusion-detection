"""Prediction service and strict inference validation.

Phase 7 Requirements:
- Strictly validates 77 named features matching the frozen schema.
- Feature ordering independence: maps JSON keys to saved feature order.
- Accepts finite numbers and explicit JSON null for missing measurements.
- Strictly rejects strings, booleans, NaN/Infinity literals, missing keys, and unknown keys.
- Rejects entirely null/missing feature vectors.
- Reuses training cleaning rules: negative rate features converted to NaN, negative sentinels preserved.
- Never fits or refits the saved model pipeline.
- Returns uncalibrated attack_probability (score >= threshold -> ATTACK, else BENIGN).
- Never returns an inferred multiclass attack category.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from app.config import VerifiedModel
from app.ml.preprocess import (
    FLOW_BYTES_S_CLEANED_INDEX,
    FLOW_PACKETS_S_CLEANED_INDEX,
    MODEL_FEATURE_COUNT,
)


@dataclasses.dataclass(frozen=True)
class PredictionResult:
    """Prediction output for a single network flow."""
    prediction_label: str  # "BENIGN" or "ATTACK"
    attack_probability: float  # Uncalibrated model probability for class 1
    decision_threshold: float
    model_version: str


def validate_and_extract_feature_vector(
    raw_features: Dict[str, Any],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Validate raw feature dictionary and extract ordered float64 vector.

    Args:
        raw_features: Dictionary mapping feature name to numeric value or None.
        feature_names: Ordered list of 77 expected feature names.

    Returns:
        1D float64 array of length 77.

    Raises:
        ValueError: On schema mismatch, invalid data types, or entirely null vector.
    """
    if not isinstance(raw_features, dict):
        raise ValueError(f"Features must be a JSON object (dictionary), got {type(raw_features).__name__}")

    input_keys = set(raw_features.keys())
    expected_keys = set(feature_names)

    # 1. Schema key validation
    missing = expected_keys - input_keys
    if missing:
        raise ValueError(f"Missing {len(missing)} required features: {sorted(list(missing))}")

    extra = input_keys - expected_keys
    if extra:
        raise ValueError(f"Unexpected {len(extra)} unknown features: {sorted(list(extra))}")

    # 2. Value type and finiteness validation
    vec = np.empty(len(feature_names), dtype=np.float64)
    all_missing = True

    for idx, name in enumerate(feature_names):
        val = raw_features[name]

        # Explicit JSON null is acceptable as missing measurement
        if val is None:
            vec[idx] = np.nan
            continue

        # Reject booleans (bool is a subclass of int in Python)
        if isinstance(val, bool):
            raise ValueError(f"Feature '{name}' has boolean value {val!r}; booleans are invalid numeric features.")

        # Accept finite numbers
        if isinstance(val, (int, float)):
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(
                    f"Feature '{name}' has non-finite value {val!r}. "
                    "NaN and Infinity literals are rejected; pass JSON null for missing values."
                )
            vec[idx] = f_val
            all_missing = False
        else:
            raise ValueError(
                f"Feature '{name}' has invalid type {type(val).__name__} (value: {val!r}); "
                "must be finite number or null."
            )

    # 3. Reject entirely null vector
    if all_missing:
        raise ValueError("Feature vector cannot be entirely null/missing.")

    # 4. Negative rate feature sanitization (same rule as Phase 5 preprocessing)
    if not np.isnan(vec[FLOW_BYTES_S_CLEANED_INDEX]) and vec[FLOW_BYTES_S_CLEANED_INDEX] < 0.0:
        vec[FLOW_BYTES_S_CLEANED_INDEX] = np.nan

    if not np.isnan(vec[FLOW_PACKETS_S_CLEANED_INDEX]) and vec[FLOW_PACKETS_S_CLEANED_INDEX] < 0.0:
        vec[FLOW_PACKETS_S_CLEANED_INDEX] = np.nan

    return vec


def predict_single_flow(
    raw_features: Dict[str, Any],
    model: VerifiedModel,
) -> PredictionResult:
    """Execute prediction for a single validated feature dictionary.

    Args:
        raw_features: 77-feature dictionary.
        model: VerifiedModel instance loaded during lifespan.

    Returns:
        PredictionResult with label, uncalibrated score, and threshold.
    """
    vec = validate_and_extract_feature_vector(raw_features, model.feature_names)
    X = vec.reshape(1, -1)

    # In-memory inference: pipeline applies SimpleImputer and VarianceThreshold
    probs = model.pipeline.predict_proba(X)
    score = float(probs[0, model.attack_col_idx])

    label = "ATTACK" if score >= model.frozen_threshold else "BENIGN"

    return PredictionResult(
        prediction_label=label,
        attack_probability=score,
        decision_threshold=model.frozen_threshold,
        model_version=model.model_version,
    )


def predict_batch_flows(
    flows_data: List[Dict[str, Any]],
    model: VerifiedModel,
) -> List[PredictionResult]:
    """Execute batch prediction for 1-500 flows with pre-validation.

    Args:
        flows_data: List of raw feature dictionaries (or dicts containing a 'features' key).
        model: VerifiedModel instance loaded during lifespan.

    Returns:
        List of PredictionResult matching the exact input order.

    Raises:
        ValueError: If batch size is invalid or any flow fails validation.
    """
    n_flows = len(flows_data)
    if n_flows == 0:
        raise ValueError("Batch request cannot be empty (must contain between 1 and 500 flows).")
    if n_flows > 500:
        raise ValueError(f"Batch size {n_flows} exceeds maximum limit of 500 flows.")

    # 1. Validate the entire batch upfront before inference
    vectors: List[np.ndarray] = []
    for i, item in enumerate(flows_data):
        feat_dict = item.get("features", item) if isinstance(item, dict) else item
        if not isinstance(feat_dict, dict):
            raise ValueError(f"Item at index {i} is not a valid JSON object.")
        try:
            vec = validate_and_extract_feature_vector(feat_dict, model.feature_names)
            vectors.append(vec)
        except ValueError as err:
            raise ValueError(f"Validation failure at batch index {i}: {err}") from err

    # 2. Vectorized batch inference
    X_batch = np.vstack(vectors)
    probs_batch = model.pipeline.predict_proba(X_batch)
    scores = probs_batch[:, model.attack_col_idx].astype(float)

    # 3. Assemble results preserving input order
    results: List[PredictionResult] = []
    for score in scores:
        label = "ATTACK" if score >= model.frozen_threshold else "BENIGN"
        results.append(
            PredictionResult(
                prediction_label=label,
                attack_probability=float(score),
                decision_threshold=model.frozen_threshold,
                model_version=model.model_version,
            )
        )

    return results
