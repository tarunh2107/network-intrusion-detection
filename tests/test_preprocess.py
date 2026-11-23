"""Unit tests for Phase 5 data preprocessing, cleaning, fingerprinting, and splitting."""

import csv
import json
import pathlib
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from app.ml.preprocess import (
    CANONICAL_NAN_UINT64,
    CANONICAL_ZERO_UINT64,
    EXPECTED_RAW_COLUMNS,
    FLOW_BYTES_S_CLEANED_INDEX,
    FLOW_PACKETS_S_CLEANED_INDEX,
    MODEL_FEATURE_COUNT,
    MODEL_FEATURE_NAMES,
    RAW_COLUMN_COUNT,
    assign_partition,
    clean_chunk_dataframe,
    clean_inference_vector,
    compute_fingerprint_for_row,
    compute_fingerprints_batch,
    inspect_and_validate_header,
    validate_and_map_label,
)


def make_dummy_raw_row(
    label: str = "BENIGN",
    fwd_header_len: str = "32",
    flow_bytes_s: str = "100.5",
    flow_packets_s: str = "50.0",
    init_win_bytes_fwd: str = "-1",
    fwd_header_len_dup: str = "32",
) -> List[str]:
    """Create a valid 79-column dummy row with configurable key features."""
    row = ["1.0"] * RAW_COLUMN_COUNT
    row[14] = str(flow_bytes_s)
    row[15] = str(flow_packets_s)
    row[34] = str(fwd_header_len)
    row[55] = str(fwd_header_len_dup)
    row[66] = str(init_win_bytes_fwd)  # Init_Win_bytes_forward
    row[78] = label
    return row


def test_numeric_canonicalization():
    """Verify that numerically equivalent strings and zero representations yield identical fingerprints."""
    # Test zero representations
    vec_zero1 = np.full(MODEL_FEATURE_COUNT, 0.0, dtype=np.float64)
    vec_zero2 = np.full(MODEL_FEATURE_COUNT, -0.0, dtype=np.float64)

    fp_zero1 = compute_fingerprint_for_row(vec_zero1)
    fp_zero2 = compute_fingerprint_for_row(vec_zero2)
    assert fp_zero1 == fp_zero2, "0.0 and -0.0 must produce identical fingerprints"

    # Test NaNs
    vec_nan1 = np.full(MODEL_FEATURE_COUNT, np.nan, dtype=np.float64)
    vec_nan2 = np.full(MODEL_FEATURE_COUNT, float("nan"), dtype=np.float64)
    fp_nan1 = compute_fingerprint_for_row(vec_nan1)
    fp_nan2 = compute_fingerprint_for_row(vec_nan2)
    assert fp_nan1 == fp_nan2, "All NaN variants must produce identical canonical fingerprints"

    # String variations through clean_inference_vector
    v1 = clean_inference_vector(["0"] * MODEL_FEATURE_COUNT)
    v2 = clean_inference_vector(["0.0"] * MODEL_FEATURE_COUNT)
    v3 = clean_inference_vector(["0e0"] * MODEL_FEATURE_COUNT)
    assert compute_fingerprint_for_row(v1) == compute_fingerprint_for_row(v2)
    assert compute_fingerprint_for_row(v2) == compute_fingerprint_for_row(v3)


def test_zero_vs_missing_fingerprint_distinction():
    """Verify that a zero-valued vector and an otherwise identical vector containing a missing value
    produce strictly DIFFERENT fingerprints.
    """
    vec_zero = np.full(MODEL_FEATURE_COUNT, 0.0, dtype=np.float64)
    fp_zero = compute_fingerprint_for_row(vec_zero)

    # 1. Entirely missing vs entirely zero
    vec_all_nan = np.full(MODEL_FEATURE_COUNT, np.nan, dtype=np.float64)
    fp_all_nan = compute_fingerprint_for_row(vec_all_nan)
    assert fp_zero != fp_all_nan, "All-zero and all-NaN vectors must have different fingerprints"

    # 2. Identical zero vector except for a single missing value at critical feature positions
    for idx in [0, FLOW_BYTES_S_CLEANED_INDEX, FLOW_PACKETS_S_CLEANED_INDEX, MODEL_FEATURE_COUNT - 1]:
        vec_single_nan = np.full(MODEL_FEATURE_COUNT, 0.0, dtype=np.float64)
        vec_single_nan[idx] = np.nan
        fp_single_nan = compute_fingerprint_for_row(vec_single_nan)
        assert fp_zero != fp_single_nan, (
            f"Zero vector and vector with NaN at feature index {idx} must produce different fingerprints"
        )



def test_missing_and_infinite_values():
    """Verify that string NaNs and infinities are converted to np.nan and counted."""
    rows = [
        make_dummy_raw_row(flow_bytes_s="NaN", flow_packets_s="Infinity"),
        make_dummy_raw_row(flow_bytes_s="-Infinity", flow_packets_s="null"),
        make_dummy_raw_row(flow_bytes_s="   ", flow_packets_s="corrupted_text"),
    ]
    df = pd.DataFrame(rows)
    result = clean_chunk_dataframe(df, source_file="test.csv", start_row_number=1)

    assert result.stats.nan_strings_converted >= 2
    assert result.stats.infinities_converted >= 2
    assert result.stats.invalid_numerics_converted >= 1

    # Check resulting matrix has NaNs in those positions
    assert np.isnan(result.features[0, FLOW_BYTES_S_CLEANED_INDEX])
    assert np.isnan(result.features[0, FLOW_PACKETS_S_CLEANED_INDEX])
    assert np.isnan(result.features[1, FLOW_BYTES_S_CLEANED_INDEX])
    assert np.isnan(result.features[1, FLOW_PACKETS_S_CLEANED_INDEX])
    assert np.isnan(result.features[2, FLOW_BYTES_S_CLEANED_INDEX])
    assert np.isnan(result.features[2, FLOW_PACKETS_S_CLEANED_INDEX])


def test_negative_rate_handling_vs_sentinel_preservation():
    """Verify that negative rate features are converted to NaN while other negative values are preserved."""
    # Row with negative rates and negative TCP window sentinel (-1)
    row = make_dummy_raw_row(
        flow_bytes_s="-261000.0",
        flow_packets_s="-150.0",
        init_win_bytes_fwd="-1",
    )
    df = pd.DataFrame([row])
    result = clean_chunk_dataframe(df, source_file="test.csv", start_row_number=1)

    # Rates should be converted to np.nan
    assert np.isnan(result.features[0, FLOW_BYTES_S_CLEANED_INDEX])
    assert np.isnan(result.features[0, FLOW_PACKETS_S_CLEANED_INDEX])
    assert result.stats.negative_flow_bytes_converted == 1
    assert result.stats.negative_flow_packets_converted == 1

    # Init_Win_bytes_forward (pos 65 in 77-feature schema, raw pos 66) must be preserved as -1.0
    init_win_idx = MODEL_FEATURE_NAMES.index("Init_Win_bytes_forward")
    assert result.features[0, init_win_idx] == -1.0


def test_duplicate_headers_and_equality_check():
    """Verify duplicate column equality check and redundant column dropping."""
    # 1. Matching columns 34 and 55: succeeds and drops column 55
    matching_row = make_dummy_raw_row(fwd_header_len="40", fwd_header_len_dup="40")
    df = pd.DataFrame([matching_row])
    result = clean_chunk_dataframe(df, source_file="test.csv", start_row_number=1)
    assert result.features.shape == (1, 77)

    # 2. Mismatched columns 34 and 55: raises ValueError
    mismatch_row = make_dummy_raw_row(fwd_header_len="40", fwd_header_len_dup="99")
    df_bad = pd.DataFrame([mismatch_row])
    with pytest.raises(ValueError, match="have 1 mismatching values"):
        clean_chunk_dataframe(df_bad, source_file="test.csv", start_row_number=1)

    # 3. Header validation
    valid_header = list(EXPECTED_RAW_COLUMNS)
    assert len(inspect_and_validate_header(valid_header)) == RAW_COLUMN_COUNT

    bad_header = valid_header[:-1]  # 78 cols
    with pytest.raises(ValueError, match="Schema mismatch"):
        inspect_and_validate_header(bad_header)


def test_label_validation_and_rejection():
    """Verify known label mapping, byte corruption handling, and rejection of unknown labels."""
    # Known labels
    disp, bin_tgt = validate_and_map_label("BENIGN")
    assert disp == "BENIGN" and bin_tgt == 0

    disp, bin_tgt = validate_and_map_label("DDoS")
    assert disp == "DDoS" and bin_tgt == 1

    # Byte replacement label
    disp, bin_tgt = validate_and_map_label("Web Attack \ufffd Brute Force")
    assert disp == "Web Attack - Brute Force" and bin_tgt == 1

    # Unknown label rejection
    with pytest.raises(ValueError, match="Unexpected or unauthorized attack label"):
        validate_and_map_label("UnknownTrojan")

    # Blank label rejection
    with pytest.raises(ValueError, match="Missing or blank class label"):
        validate_and_map_label("   ")

    # Rejection inside clean_chunk_dataframe
    rows = [
        make_dummy_raw_row(label="BENIGN"),
        make_dummy_raw_row(label="RansomwareUnknown"),
        make_dummy_raw_row(label=""),
    ]
    df = pd.DataFrame(rows)
    result = clean_chunk_dataframe(df, source_file="test.csv", start_row_number=1)
    assert result.stats.total_rows_inspected == 3
    assert result.stats.valid_rows_emitted == 1
    assert result.stats.rejected_rows_count == 2
    assert len(result.rejected_rows) == 2


def test_deterministic_partition_assignment_and_grouping():
    """Verify that identical features map to the same partition, across files and labels."""
    # Row A with label BENIGN
    row_benign = make_dummy_raw_row(label="BENIGN", flow_bytes_s="1234.5")
    # Row B with identical features but label DDoS (conflicting label)
    row_attack = make_dummy_raw_row(label="DDoS", flow_bytes_s="1234.5")

    df_a = pd.DataFrame([row_benign])
    df_b = pd.DataFrame([row_attack])

    res_a = clean_chunk_dataframe(df_a, source_file="file1.csv", start_row_number=1)
    res_b = clean_chunk_dataframe(df_b, source_file="file2.csv", start_row_number=50)

    fp_a = res_a.metadata[0]["fingerprint"]
    fp_b = res_b.metadata[0]["fingerprint"]

    # Fingerprints must be strictly identical (labels and provenance excluded)
    assert fp_a == fp_b

    # Partition assignments must be identical
    part_a = assign_partition(fp_a, seed=42)
    part_b = assign_partition(fp_b, seed=42)
    assert part_a == part_b
    assert part_a in ("train", "validation", "test")


def test_inference_vector_cleaner():
    """Verify clean_inference_vector with dictionary input."""
    data_dict = {name: 10.0 for name in MODEL_FEATURE_NAMES}
    data_dict["Flow Bytes/s"] = -500.0  # negative rate -> np.nan
    data_dict["Flow Packets/s"] = "Infinity"  # inf -> np.nan

    vec = clean_inference_vector(data_dict)
    assert vec.shape == (77,)
    assert np.isnan(vec[FLOW_BYTES_S_CLEANED_INDEX])
    assert np.isnan(vec[FLOW_PACKETS_S_CLEANED_INDEX])
    assert vec[0] == 10.0


def test_end_to_end_preparer_pipeline_and_safe_restart(tmp_path: pathlib.Path):
    """Verify DatasetPreparer end-to-end sharding, verification, and safe restart from checkpoint."""
    from scripts.prepare_dataset import DatasetPreparer

    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir(parents=True)

    header = list(EXPECTED_RAW_COLUMNS)

    # File 1: 5 rows
    file1 = raw_dir / "sample1.csv"
    with open(file1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(5):
            w.writerow(make_dummy_raw_row(label="BENIGN", flow_bytes_s=str(10.0 + i)))

    # File 2: 5 rows (including 1 with negative flow bytes, 1 with DDoS)
    file2 = raw_dir / "sample2.csv"
    with open(file2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(make_dummy_raw_row(label="DDoS", flow_bytes_s="10.0"))  # cross-file duplicate of sample1 row 0!
        w.writerow(make_dummy_raw_row(label="PortScan", flow_bytes_s="-50.0"))
        for i in range(3):
            w.writerow(make_dummy_raw_row(label="BENIGN", flow_bytes_s=str(100.0 + i)))

    preparer = DatasetPreparer(
        raw_dir=raw_dir,
        output_dir=out_dir,
        reports_dir=reports_dir,
        chunk_size=3,
        shard_size=4,
        seed=42,
    )

    verif = preparer.run()
    assert verif["total_emitted_rows"] == 10
    assert verif["is_strictly_disjoint"] is True

    # Check generated files
    assert (out_dir / "features.json").exists()
    assert (out_dir / "metadata_columns.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "checkpoint.json").exists()
    assert (reports_dir / "preprocessing_report.json").exists()
    assert (reports_dir / "preprocessing_report.md").exists()

    # Safe restart test: running again without --force should reuse checkpoint cleanly
    preparer_restart = DatasetPreparer(
        raw_dir=raw_dir,
        output_dir=out_dir,
        reports_dir=reports_dir,
        chunk_size=3,
        shard_size=4,
        seed=42,
        force=False,
    )
    verif_restart = preparer_restart.run()
    assert verif_restart["total_emitted_rows"] == 10
    assert verif_restart["is_strictly_disjoint"] is True

