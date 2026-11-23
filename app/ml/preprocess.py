"""Deterministic data preprocessing, cleaning, fingerprinting, and partition assignment.

Reusable across batch offline dataset preparation and future real-time REST API inference.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import math
import pathlib
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Schema Definitions & Constants
# -----------------------------------------------------------------------------

RAW_COLUMN_COUNT: int = 79
MODEL_FEATURE_COUNT: int = 77
RAW_DUPLICATE_COL_PRIMARY: int = 34
RAW_DUPLICATE_COL_SECONDARY: int = 55
RAW_LABEL_COL_INDEX: int = 78

# Canonical quiet NaN and canonical zero bit patterns (IEEE 754 little-endian uint64)
CANONICAL_NAN_UINT64: int = 0x7FF8000000000000
CANONICAL_ZERO_UINT64: int = 0x0000000000000000

# Metadata fields emitted alongside cleaned feature shards
METADATA_COLUMNS: List[str] = [
    "source_file",
    "source_row_number",
    "raw_label",
    "display_label",
    "binary_target",
    "fingerprint",
]

# Canonical 79 stripped raw column names
EXPECTED_RAW_COLUMNS: List[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]

# Canonical 77 model features after dropping redundant column 55
MODEL_FEATURE_NAMES: List[str] = (
    EXPECTED_RAW_COLUMNS[:55] + EXPECTED_RAW_COLUMNS[56:78]
)
assert len(MODEL_FEATURE_NAMES) == MODEL_FEATURE_COUNT, (
    f"Expected {MODEL_FEATURE_COUNT} features, got {len(MODEL_FEATURE_NAMES)}"
)

# Position of rate features in the 77-feature cleaned schema
FLOW_BYTES_S_CLEANED_INDEX: int = 14
FLOW_PACKETS_S_CLEANED_INDEX: int = 15
assert MODEL_FEATURE_NAMES[FLOW_BYTES_S_CLEANED_INDEX] == "Flow Bytes/s"
assert MODEL_FEATURE_NAMES[FLOW_PACKETS_S_CLEANED_INDEX] == "Flow Packets/s"

# Documented bidirectional label mapping
# Maps raw string (including UTF-8 byte replacement variations) to (display_label, binary_target)
LABEL_MAPPING: Dict[str, Tuple[str, int]] = {
    # Benign
    "BENIGN": ("BENIGN", 0),
    # Attacks (DoS / DDoS)
    "DDoS": ("DDoS", 1),
    "PortScan": ("PortScan", 1),
    "Bot": ("Bot", 1),
    "Infiltration": ("Infiltration", 1),
    "FTP-Patator": ("FTP-Patator", 1),
    "SSH-Patator": ("SSH-Patator", 1),
    "DoS slowloris": ("DoS slowloris", 1),
    "DoS Slowhttptest": ("DoS Slowhttptest", 1),
    "DoS Hulk": ("DoS Hulk", 1),
    "DoS GoldenEye": ("DoS GoldenEye", 1),
    "Heartbleed": ("Heartbleed", 1),
    # Web attacks with raw replacement character U+FFFD
    "Web Attack \ufffd Brute Force": ("Web Attack - Brute Force", 1),
    "Web Attack \ufffd XSS": ("Web Attack - XSS", 1),
    "Web Attack \ufffd Sql Injection": ("Web Attack - Sql Injection", 1),
    # Sanitized alternatives (for compatibility with clean inputs)
    "Web Attack - Brute Force": ("Web Attack - Brute Force", 1),
    "Web Attack - XSS": ("Web Attack - XSS", 1),
    "Web Attack - Sql Injection": ("Web Attack - Sql Injection", 1),
}


# -----------------------------------------------------------------------------
# 2. Data Classes for Tracking & Streaming
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class CleaningStats:
    """Tracks counts of cleaning transformations and anomalies."""
    total_rows_inspected: int = 0
    valid_rows_emitted: int = 0
    rejected_rows_count: int = 0
    nan_strings_converted: int = 0
    infinities_converted: int = 0
    negative_flow_bytes_converted: int = 0
    negative_flow_packets_converted: int = 0
    invalid_numerics_converted: int = 0
    label_counts_raw: Dict[str, int] = dataclasses.field(default_factory=dict)
    label_counts_display: Dict[str, int] = dataclasses.field(default_factory=dict)
    binary_counts: Dict[int, int] = dataclasses.field(default_factory=dict)

    def merge(self, other: CleaningStats) -> None:
        self.total_rows_inspected += other.total_rows_inspected
        self.valid_rows_emitted += other.valid_rows_emitted
        self.rejected_rows_count += other.rejected_rows_count
        self.nan_strings_converted += other.nan_strings_converted
        self.infinities_converted += other.infinities_converted
        self.negative_flow_bytes_converted += other.negative_flow_bytes_converted
        self.negative_flow_packets_converted += other.negative_flow_packets_converted
        self.invalid_numerics_converted += other.invalid_numerics_converted
        for k, v in other.label_counts_raw.items():
            self.label_counts_raw[k] = self.label_counts_raw.get(k, 0) + v
        for k, v in other.label_counts_display.items():
            self.label_counts_display[k] = self.label_counts_display.get(k, 0) + v
        for k, v in other.binary_counts.items():
            self.binary_counts[k] = self.binary_counts.get(k, 0) + v


@dataclasses.dataclass
class CleanedChunkResult:
    """Holds cleaned feature array, associated metadata, and anomaly counts."""
    features: np.ndarray  # shape (N, 77), dtype float64
    metadata: List[Dict[str, Any]]
    stats: CleaningStats
    rejected_rows: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# 3. Label Mapping & Validation
# -----------------------------------------------------------------------------

def validate_and_map_label(raw_label: str) -> Tuple[str, int]:
    """Validate raw label string and return (display_label, binary_target).

    Raises:
        ValueError: If label is missing, blank, or not in known schema.
    """
    stripped = raw_label.strip() if isinstance(raw_label, str) else ""
    if not stripped:
        raise ValueError("Missing or blank class label encountered.")

    mapping = LABEL_MAPPING.get(stripped)
    if mapping is None:
        # Check case-insensitive benign
        if stripped.upper() == "BENIGN":
            return ("BENIGN", 0)
        raise ValueError(f"Unexpected or unauthorized attack label: {raw_label!r}")

    return mapping


# -----------------------------------------------------------------------------
# 4. Header & Schema Inspection
# -----------------------------------------------------------------------------

def inspect_and_validate_header(header_row: Sequence[str]) -> List[str]:
    """Inspect positional raw header, strip whitespace, and validate 79-column schema.

    Raises:
        ValueError: If column count or expected duplicate header positions differ.
    """
    if len(header_row) != RAW_COLUMN_COUNT:
        raise ValueError(
            f"Schema mismatch: expected {RAW_COLUMN_COUNT} columns, found {len(header_row)}"
        )

    stripped = [c.strip() for c in header_row]

    # Verify duplicate column positions
    if stripped[RAW_DUPLICATE_COL_PRIMARY] != "Fwd Header Length":
        raise ValueError(
            f"Expected 'Fwd Header Length' at position {RAW_DUPLICATE_COL_PRIMARY}, "
            f"got {stripped[RAW_DUPLICATE_COL_PRIMARY]!r}"
        )
    if stripped[RAW_DUPLICATE_COL_SECONDARY] != "Fwd Header Length":
        raise ValueError(
            f"Expected 'Fwd Header Length' at position {RAW_DUPLICATE_COL_SECONDARY}, "
            f"got {stripped[RAW_DUPLICATE_COL_SECONDARY]!r}"
        )
    if stripped[RAW_LABEL_COL_INDEX] != "Label":
        raise ValueError(
            f"Expected 'Label' at position {RAW_LABEL_COL_INDEX}, "
            f"got {stripped[RAW_LABEL_COL_INDEX]!r}"
        )

    return stripped


# -----------------------------------------------------------------------------
# 5. Deterministic Numeric Fingerprinting & Partition Assignment
# -----------------------------------------------------------------------------

def canonicalize_feature_matrix_uint64(feature_matrix: np.ndarray) -> np.ndarray:
    """Canonicalize float64 array into deterministic uint64 bit representations.

    Normalizes:
    - All NaNs to CANONICAL_NAN_UINT64 (0x7FF8000000000000).
    - -0.0, +0.0 to CANONICAL_ZERO_UINT64 (0x0000000000000000).
    - Finite floats preserve exact IEEE 754 64-bit little-endian bit patterns.
    """
    assert feature_matrix.dtype == np.float64, "Input must be float64"
    u = np.copy(feature_matrix).view(np.uint64)
    nan_mask = np.isnan(feature_matrix)
    zero_mask = (feature_matrix == 0.0)
    u[nan_mask] = CANONICAL_NAN_UINT64
    u[zero_mask] = CANONICAL_ZERO_UINT64
    return u


def compute_fingerprint_for_row(feature_vector: np.ndarray) -> str:
    """Compute deterministic SHA-256 fingerprint for a single 77-element float vector."""
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
    u = canonicalize_feature_matrix_uint64(feature_vector)
    return hashlib.sha256(u[0].tobytes()).hexdigest()


def compute_fingerprints_batch(feature_matrix: np.ndarray) -> List[str]:
    """Compute deterministic SHA-256 fingerprints for a batch of feature vectors."""
    u = canonicalize_feature_matrix_uint64(feature_matrix)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in u]


def assign_partition(
    fingerprint: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> str:
    """Deterministically assign a fingerprint to 'train', 'validation', or 'test'.

    Uses a salted SHA-256 hash modulo 10,000 bucket rule.
    Target proportions:
    - [0, 7000): train (70.0%)
    - [7000, 8500): validation (15.0%)
    - [8500, 10000): test (15.0%)

    All rows sharing the same fingerprint are guaranteed to map to the identical partition.
    """
    train_thresh = int(train_ratio * 10000)
    val_thresh = train_thresh + int(val_ratio * 10000)

    salt = f"phase5_split_v1:seed={seed}:".encode("ascii")
    digest = hashlib.sha256(salt + fingerprint.encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 10000

    if bucket < train_thresh:
        return "train"
    elif bucket < val_thresh:
        return "validation"
    else:
        return "test"


# -----------------------------------------------------------------------------
# 6. Chunk Cleaning & Feature Extraction
# -----------------------------------------------------------------------------

def clean_chunk_dataframe(
    df_raw: pd.DataFrame,
    source_file: str,
    start_row_number: int,
    validate_column_equality: bool = True,
) -> CleanedChunkResult:
    """Clean a pandas DataFrame chunk (79 columns, positional index 0..78).

    Steps:
    1. Check column 34 == column 55 across all chunk rows (fails if mismatch).
    2. Drop redundant column 55.
    3. Validate and map labels (column 78).
    4. Parse 77 features to float64, converting NaN/inf and negative rates to np.nan.
    5. Compute deterministic fingerprints and assign metadata.
    """
    stats = CleaningStats(total_rows_inspected=len(df_raw))
    num_rows = len(df_raw)

    if num_rows == 0:
        return CleanedChunkResult(
            features=np.empty((0, MODEL_FEATURE_COUNT), dtype=np.float64),
            metadata=[],
            stats=stats,
            rejected_rows=[],
        )

    # 1. Verify equality of column 34 vs 55
    if validate_column_equality:
        col34 = df_raw.iloc[:, RAW_DUPLICATE_COL_PRIMARY].astype(str).str.strip()
        col55 = df_raw.iloc[:, RAW_DUPLICATE_COL_SECONDARY].astype(str).str.strip()
        mismatches = (col34 != col55).sum()
        if mismatches > 0:
            raise ValueError(
                f"Validation failed in {source_file} (rows {start_row_number}..{start_row_number + num_rows - 1}): "
                f"Column {RAW_DUPLICATE_COL_PRIMARY} and Column {RAW_DUPLICATE_COL_SECONDARY} "
                f"have {mismatches} mismatching values."
            )

    # 2. Extract raw labels (column 78) and validate
    raw_labels = df_raw.iloc[:, RAW_LABEL_COL_INDEX].astype(str).tolist()
    valid_mask = np.ones(num_rows, dtype=bool)
    display_labels: List[str] = []
    binary_targets: List[int] = []
    rejected_records: List[Dict[str, Any]] = []

    for i, raw_lbl in enumerate(raw_labels):
        try:
            disp_lbl, bin_tgt = validate_and_map_label(raw_lbl)
            display_labels.append(disp_lbl)
            binary_targets.append(bin_tgt)
            stats.label_counts_raw[raw_lbl.strip()] = stats.label_counts_raw.get(raw_lbl.strip(), 0) + 1
            stats.label_counts_display[disp_lbl] = stats.label_counts_display.get(disp_lbl, 0) + 1
            stats.binary_counts[bin_tgt] = stats.binary_counts.get(bin_tgt, 0) + 1
        except ValueError as err:
            valid_mask[i] = False
            display_labels.append("")
            binary_targets.append(-1)
            stats.rejected_rows_count += 1
            rejected_records.append({
                "source_file": source_file,
                "source_row_number": start_row_number + i,
                "raw_label": raw_lbl,
                "reason": str(err),
            })

    # 3. Extract 77 feature columns (drop col 55 and col 78)
    feat_cols_indices = list(range(0, 55)) + list(range(56, 78))
    feat_df = df_raw.iloc[:, feat_cols_indices].copy()
    assert feat_df.shape[1] == MODEL_FEATURE_COUNT

    # 4. Parse features to numeric float64
    feat_matrix = np.empty((num_rows, MODEL_FEATURE_COUNT), dtype=np.float64)

    for col_idx in range(MODEL_FEATURE_COUNT):
        col_series = feat_df.iloc[:, col_idx].astype(str).str.strip()

        # Count string NaNs
        lower_series = col_series.str.lower()
        nan_str_mask = lower_series.isin(["nan", "null", "none", ""]) | col_series.isna()
        stats.nan_strings_converted += int(nan_str_mask.sum())

        inf_str_mask = lower_series.isin(["infinity", "+infinity", "inf", "+inf", "-infinity", "-inf"])

        # Numeric conversion with coerce
        num_vals = pd.to_numeric(col_series, errors="coerce").to_numpy(dtype=np.float64, copy=True)

        # Handle numeric infinities (count each infinity cell exactly once)
        inf_mask = np.isinf(num_vals) | inf_str_mask.to_numpy()
        stats.infinities_converted += int(inf_mask.sum())
        num_vals[inf_mask] = np.nan

        # Count values that became NaN but were not explicit NaN/inf strings
        coerced_nan_mask = np.isnan(num_vals) & (~nan_str_mask.to_numpy()) & (~inf_mask)
        stats.invalid_numerics_converted += int(coerced_nan_mask.sum())

        # Handle rate features negative values
        if col_idx == FLOW_BYTES_S_CLEANED_INDEX:
            neg_bytes_mask = (num_vals < 0) & (~np.isnan(num_vals))
            stats.negative_flow_bytes_converted += int(neg_bytes_mask.sum())
            num_vals[neg_bytes_mask] = np.nan
        elif col_idx == FLOW_PACKETS_S_CLEANED_INDEX:
            neg_pkts_mask = (num_vals < 0) & (~np.isnan(num_vals))
            stats.negative_flow_packets_converted += int(neg_pkts_mask.sum())
            num_vals[neg_pkts_mask] = np.nan

        feat_matrix[:, col_idx] = num_vals

    # Filter out rejected rows if any
    if not valid_mask.all():
        feat_matrix = feat_matrix[valid_mask]
        display_labels = [lbl for i, lbl in enumerate(display_labels) if valid_mask[i]]
        binary_targets = [tgt for i, tgt in enumerate(binary_targets) if valid_mask[i]]
        raw_labels = [lbl for i, lbl in enumerate(raw_labels) if valid_mask[i]]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
    else:
        valid_indices = list(range(num_rows))

    stats.valid_rows_emitted = len(valid_indices)

    # 5. Compute deterministic fingerprints
    fingerprints = compute_fingerprints_batch(feat_matrix) if len(feat_matrix) > 0 else []

    # Build metadata list
    metadata: List[Dict[str, Any]] = []
    for idx_in_valid, orig_idx in enumerate(valid_indices):
        metadata.append({
            "source_file": source_file,
            "source_row_number": start_row_number + orig_idx,
            "raw_label": raw_labels[idx_in_valid],
            "display_label": display_labels[idx_in_valid],
            "binary_target": binary_targets[idx_in_valid],
            "fingerprint": fingerprints[idx_in_valid],
        })

    return CleanedChunkResult(
        features=feat_matrix,
        metadata=metadata,
        stats=stats,
        rejected_rows=rejected_records,
    )


# -----------------------------------------------------------------------------
# 7. Inference Helper for Future REST API
# -----------------------------------------------------------------------------

def clean_inference_vector(
    raw_vector: Union[Dict[str, Any], Sequence[Any], np.ndarray],
) -> np.ndarray:
    """Clean a single inference flow record into a canonical 77-element float64 vector.

    Accepts:
    - Dict mapping feature names to values (names matching MODEL_FEATURE_NAMES or 78/79 raw names).
    - Sequence of 77 numeric values (or 79 raw values with redundant col 55 dropped).

    Returns:
        np.ndarray of shape (77,) dtype float64 with canonicalized missing values.
    """
    if isinstance(raw_vector, dict):
        # Extract features in MODEL_FEATURE_NAMES order
        cleaned = np.empty(MODEL_FEATURE_COUNT, dtype=np.float64)
        for i, name in enumerate(MODEL_FEATURE_NAMES):
            val = raw_vector.get(name, raw_vector.get(f" {name}", np.nan))
            try:
                num = float(val)
                if math.isinf(num) or math.isnan(num):
                    num = np.nan
                elif i in (FLOW_BYTES_S_CLEANED_INDEX, FLOW_PACKETS_S_CLEANED_INDEX) and num < 0:
                    num = np.nan
                cleaned[i] = num
            except (ValueError, TypeError):
                cleaned[i] = np.nan
        return cleaned

    arr = np.asarray(raw_vector, dtype=object)
    if arr.ndim == 1 and len(arr) == RAW_COLUMN_COUNT:
        # Raw 79-column format: verify col 34 == 55 and drop 55 & 78
        if str(arr[34]).strip() != str(arr[55]).strip():
            raise ValueError("Inference input has mismatched duplicate Fwd Header Length values.")
        indices = list(range(0, 55)) + list(range(56, 78))
        arr = arr[indices]

    if len(arr) != MODEL_FEATURE_COUNT:
        raise ValueError(f"Expected {MODEL_FEATURE_COUNT} features for inference, got {len(arr)}")

    cleaned = np.empty(MODEL_FEATURE_COUNT, dtype=np.float64)
    for i, val in enumerate(arr):
        try:
            num = float(val)
            if math.isinf(num) or math.isnan(num):
                num = np.nan
            elif i in (FLOW_BYTES_S_CLEANED_INDEX, FLOW_PACKETS_S_CLEANED_INDEX) and num < 0:
                num = np.nan
            cleaned[i] = num
        except (ValueError, TypeError):
            cleaned[i] = np.nan
    return cleaned
