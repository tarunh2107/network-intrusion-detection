"""Baseline model training, pipeline construction, sampling, and artifact management.

Implements Phase 6A requirements:
- Deterministic streaming sampling across all shards for train and validation.
- Strict isolation of test partition.
- Sklearn Pipeline with SimpleImputer, VarianceThreshold, optional StandardScaler, and classifier.
- Joblib serialization and metadata persistence.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.preprocess import MODEL_FEATURE_NAMES

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_SAMPLE_SIZE: int = 150_000
DEFAULT_VAL_SAMPLE_SIZE: int = 50_000
DEFAULT_SEED: int = 42

METADATA_COLUMNS: List[str] = [
    "source_file",
    "source_row_number",
    "raw_label",
    "display_label",
    "binary_target",
    "fingerprint",
]


def sample_partition_shards(
    partition_dir: pathlib.Path,
    sample_size: int = 150_000,
    seed: int = 42,
    manifest_path: Optional[pathlib.Path] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Deterministically sample rows across all shards in a partition.

    Args:
        partition_dir: Directory containing shard files (e.g. data/processed/v1/train).
        sample_size: Maximum number of rows to sample across all shards.
        seed: Random seed for deterministic reproducibility.
        manifest_path: Optional path to manifest.json containing shard row counts.
        feature_names: Expected list of feature column names (defaults to 77 model features).

    Returns:
        Tuple of (X, y, metadata_df):
            X: float64 array of shape (N, num_features)
            y: int64 array of binary targets (N,)
            metadata_df: DataFrame of metadata columns (N, 6)

    Raises:
        ValueError: If partition_dir is the held-out test split, or if required columns are missing.
        FileNotFoundError: If no shard files are found.
    """
    partition_name = partition_dir.name.lower()
    if partition_name == "test":
        raise ValueError(
            "Test partition cannot be sampled or evaluated during baseline training. "
            "The test split is held out strictly for final evaluation."
        )

    if feature_names is None:
        feature_names = MODEL_FEATURE_NAMES

    # Discover and sort shard files deterministically
    shard_files = sorted(list(partition_dir.glob("*.csv.gz")))
    if not shard_files:
        raise FileNotFoundError(f"No .csv.gz shard files found in {partition_dir}")

    # Determine row counts per shard
    shard_row_counts: List[int] = []
    manifest_lookup: Dict[str, int] = {}

    resolved_manifest = manifest_path or (partition_dir.parent / "manifest.json")
    if resolved_manifest.is_file():
        try:
            with open(resolved_manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for item in manifest_data.get("shards", []):
                manifest_lookup[item["filename"]] = int(item["rows"])
        except Exception as exc:
            logger.warning("Failed to parse manifest %s: %s", resolved_manifest, exc)

    for sf in shard_files:
        if sf.name in manifest_lookup:
            shard_row_counts.append(manifest_lookup[sf.name])
        else:
            # Fallback: count rows by reading shard
            df_temp = pd.read_csv(sf, usecols=["binary_target"])
            shard_row_counts.append(len(df_temp))

    total_rows = sum(shard_row_counts)
    actual_sample_size = min(sample_size, total_rows)

    if actual_sample_size <= 0:
        raise ValueError(f"No rows available to sample in {partition_dir}")

    # Deterministic global index selection across all shards
    rng = np.random.default_rng(seed)
    selected_global_indices = np.sort(
        rng.choice(total_rows, size=actual_sample_size, replace=False)
    )

    sampled_dfs: List[pd.DataFrame] = []
    current_offset = 0

    expected_cols = METADATA_COLUMNS + feature_names

    for sf, row_count in zip(shard_files, shard_row_counts):
        shard_start = current_offset
        shard_end = current_offset + row_count
        current_offset = shard_end

        # Find global indices belonging to this shard
        mask = (selected_global_indices >= shard_start) & (selected_global_indices < shard_end)
        shard_global_indices = selected_global_indices[mask]

        if len(shard_global_indices) == 0:
            continue

        shard_local_indices = shard_global_indices - shard_start

        # Read shard and extract sampled local rows
        shard_df = pd.read_csv(sf)
        missing_cols = set(expected_cols) - set(shard_df.columns)
        if missing_cols:
            raise ValueError(f"Shard {sf.name} is missing expected columns: {missing_cols}")

        sampled_shard_df = shard_df.iloc[shard_local_indices]
        sampled_dfs.append(sampled_shard_df)

    combined_df = pd.concat(sampled_dfs, ignore_index=True)

    metadata_df = combined_df[METADATA_COLUMNS].copy()
    y = combined_df["binary_target"].values.astype(np.int64)
    X = combined_df[feature_names].values.astype(np.float64)

    return X, y, metadata_df


def build_pipeline(
    model_type: str,
    random_state: int = DEFAULT_SEED,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessing and classifier.

    Supported model_type values:
    - 'dummy': DummyClassifier (most frequent)
    - 'logistic_regression': SimpleImputer + VarianceThreshold + StandardScaler + LogisticRegression
    - 'decision_tree': SimpleImputer + VarianceThreshold + DecisionTreeClassifier
    - 'random_forest': SimpleImputer + VarianceThreshold + RandomForestClassifier

    Args:
        model_type: Identifier of baseline model.
        random_state: Seed for reproducible initialization.
        **kwargs: Model-specific hyperparameter overrides.

    Returns:
        Configured scikit-learn Pipeline.

    Raises:
        ValueError: If model_type is unknown.
    """
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    var_thresh = VarianceThreshold(threshold=0.0)

    m_lower = model_type.lower()
    if m_lower == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
        return Pipeline([
            ("imputer", imputer),
            ("var_thresh", var_thresh),
            ("clf", clf),
        ])

    elif m_lower in ("logistic_regression", "lr"):
        scaler = StandardScaler()
        max_iter = kwargs.get("max_iter", 1000)
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        return Pipeline([
            ("imputer", imputer),
            ("var_thresh", var_thresh),
            ("scaler", scaler),
            ("clf", clf),
        ])

    elif m_lower in ("decision_tree", "dt"):
        max_depth = kwargs.get("max_depth", 15)
        min_samples_leaf = kwargs.get("min_samples_leaf", 5)
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", imputer),
            ("var_thresh", var_thresh),
            ("clf", clf),
        ])

    elif m_lower in ("random_forest", "rf"):
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 18)
        min_samples_leaf = kwargs.get("min_samples_leaf", 2)
        n_jobs = kwargs.get("n_jobs", 2)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", imputer),
            ("var_thresh", var_thresh),
            ("clf", clf),
        ])

    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Supported types: "
            f"'dummy', 'logistic_regression', 'decision_tree', 'random_forest'."
        )


def save_model_artifact(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    output_dir: pathlib.Path,
) -> None:
    """Save fitted pipeline and JSON metadata to the specified directory.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        metadata: Dictionary containing training parameters, metrics, feature list, etc.
        output_dir: Target directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    meta_path = output_dir / "metadata.json"

    joblib.dump(pipeline, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model_artifact(model_dir: pathlib.Path) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load fitted pipeline and metadata from the specified directory.

    Args:
        model_dir: Directory containing model.joblib and metadata.json.

    Returns:
        Tuple of (fitted_pipeline, metadata_dict).
    """
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata
